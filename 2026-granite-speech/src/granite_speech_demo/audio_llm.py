"""Pipecat service: send the user's audio straight to the single audio-enabled
Granite Switch model and stream the spoken answer.

This is the "one model" path (no separate STT, no Mellea): the model transcribes
the audio internally and generates the response in one request. Adapted from
``hosted_stt.py`` — same audio buffering + VAD turn handling + barge-in
cancellation — but it POSTs to the Granite Switch chat endpoint with an
``input_audio`` content part and emits the answer as LLM frames (which the TTS
stage speaks), instead of emitting a transcript.
"""

import asyncio
import base64
import io
import json
import os
import time
import wave
from pathlib import Path
from typing import AsyncGenerator

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import SegmentedSTTService

from loguru import logger

# Single audio-enabled Granite Switch model (our checkpoint). Same env var the
# old Mellea LLM used, so the notebook only sets one endpoint now.
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "granite-switch-audio")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MAX_TOKENS = int(os.environ.get("AUDIO_LLM_MAX_TOKENS", "256"))


def _chat_endpoint(url: str) -> str:
    base = url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def _load_system_prompt() -> str:
    prompt_file = os.environ.get("PROMPT_FILE", "")
    if prompt_file:
        path = Path(prompt_file)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if path.is_file():
            text = path.read_text().strip()
            if text:
                logger.info("Loaded system prompt from {} ({} chars)", path, len(text))
                return text
    return (
        "You are Granite, IBM's real-time voice assistant. Answer the user's "
        "spoken question concisely and conversationally."
    )


SYSTEM_PROMPT = _load_system_prompt()


class AudioLLMService(SegmentedSTTService):
    """Audio in -> spoken answer out, via the single Granite Switch audio model.

    Subclasses SegmentedSTTService purely to reuse its audio-buffering + VAD turn
    boundaries; ``run_stt`` is repurposed to stream the LLM answer rather than a
    transcript.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session: aiohttp.ClientSession | None = None
        self._endpoint = _chat_endpoint(LLM_URL)
        self._active_task: asyncio.Task | None = None
        self._utterance_epoch: int = 0

    async def start(self, frame):
        await super().start(frame)
        self._session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_KEY}",
            }
        )

    async def stop(self, frame):
        await self._cancel_active()
        if self._session:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame):
        await self._cancel_active()
        await super().cancel(frame)

    async def _cancel_active(self):
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass
        self._active_task = None

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        # Bump epoch so any in-flight response is treated as stale, and cancel it
        # (barge-in). Pipeline interruption frames stop the TTS already in flight.
        self._utterance_epoch += 1
        await self._cancel_active()
        await super()._handle_user_started_speaking(frame)

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        self._user_speaking = False

        # Assemble the buffered PCM into a WAV (same as the old STT service).
        content = io.BytesIO()
        wav = wave.open(content, "wb")
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.setframerate(self.sample_rate)
        wav.writeframes(self._audio_buffer)
        wav.close()
        content.seek(0)
        self._audio_buffer.clear()
        audio_bytes = content.read()

        await self._cancel_active()
        epoch = self._utterance_epoch
        self._active_task = asyncio.create_task(
            self.process_generator(self.run_stt(audio_bytes, epoch)),
            name=f"{self.name}::audio_llm",
        )

    async def run_stt(
        self, audio: bytes, epoch: int | None = None
    ) -> AsyncGenerator[Frame, None]:
        if epoch is None:
            epoch = self._utterance_epoch
        if not self._session:
            yield ErrorFrame(error="audio-llm session not initialized")
            return

        audio_b64 = base64.b64encode(audio).decode("utf-8")
        payload = {
            "model": LLM_MODEL,
            "stream": True,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        }
                    ],
                },
            ],
        }

        started = False
        try:
            t0 = time.monotonic()
            async with self._session.post(self._endpoint, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("audio-llm request failed: {} {}", resp.status, error_text)
                    yield ErrorFrame(error=f"audio-llm failed: {resp.status}")
                    return

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or line_str == "data: [DONE]":
                        continue
                    if line_str.startswith("data: "):
                        line_str = line_str[6:]
                    try:
                        chunk = json.loads(line_str)
                        content_piece = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
                    if not content_piece:
                        continue
                    # Drop stale stream if the user already started a new turn.
                    if epoch != self._utterance_epoch:
                        return
                    if not started:
                        started = True
                        yield LLMFullResponseStartFrame()
                    yield LLMTextFrame(content_piece)

                if started and epoch == self._utterance_epoch:
                    yield LLMFullResponseEndFrame()
                    logger.info("audio-llm turn done ({:.3f}s)", time.monotonic() - t0)

        except asyncio.CancelledError:
            logger.debug("audio-llm cancelled (user resumed speaking)")
            raise
        except Exception as e:
            logger.exception("audio-llm error")
            yield ErrorFrame(error=f"audio-llm error: {e}")
