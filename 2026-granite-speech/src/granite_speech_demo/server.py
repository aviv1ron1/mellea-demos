"""FastAPI server with SmallWebRTC transport and pipecat pipeline."""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
import numpy as np
import uvicorn
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from granite_speech_demo.hosted_tts import HostedTTSService
from granite_speech_demo.audio_llm import AudioLLMService

from loguru import logger as loguru_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="INFO")
loguru_logger.add("logs/mellea_{time:YYYY-MM-DD}.log", level="DEBUG", rotation="1 day", retention="7 days")

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


HOST = os.environ.get("HOST", "localhost")
PORT = int(os.environ.get("PORT", "7860"))
TTS_BACKEND = os.environ.get("TTS_BACKEND", "kokoro")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_aoede")
# Playback tempo for Kokoro. Upstream hardcodes speed=1.0; >1.0 speaks faster.
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.25"))


class _FastKokoroTTSService(KokoroTTSService):
    """KokoroTTSService with a configurable speaking tempo.

    Upstream ``run_tts`` calls ``create_stream(..., speed=1.0)`` with the tempo
    hardcoded and exposes no setting for it, so we override the method to pass
    ``TTS_SPEED`` instead. Everything else (streaming, resampling, metrics) is
    identical to the upstream implementation.
    """

    @traced_tts
    async def run_tts(self, text, context_id):
        loguru_logger.debug(f"{self}: Generating TTS [{text}] @ speed={TTS_SPEED}")
        try:
            await self.start_tts_usage_metrics(text)
            stream = self._kokoro.create_stream(
                text, voice=self._settings.voice, lang=self._settings.language, speed=TTS_SPEED
            )
            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()
                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()

pcs_map: Dict[str, SmallWebRTCConnection] = {}
active_sessions: Dict[str, Dict[str, Any]] = {}

# Default ICE servers. STUN alone only works when both peers can hole-punch;
# behind a symmetric NAT (e.g. a cloud/cluster egress) a TURN relay both peers
# can reach is required. openrelay.metered.ca is a free public TURN service.
# Override wholesale with the ICE_SERVERS env var (JSON list of RTCIceServer
# dicts) to point at your own TURN — recommended for anything beyond a demo.
DEFAULT_ICE_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turns:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]


async def _mint_ice_servers() -> list[dict]:
    """Return the ICE servers for one session.

    Precedence: an explicit ICE_SERVERS env var (JSON) wins; else mint per-session
    TURN credentials from fastrtc when HF_TOKEN is set and reachable; else fall
    back to DEFAULT_ICE_SERVERS (public STUN + TURN)."""
    env_ice = os.environ.get("ICE_SERVERS")
    if env_ice:
        try:
            return json.loads(env_ice)
        except Exception as e:
            logger.warning(f"ICE_SERVERS env is not valid JSON, ignoring: {e}")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://turn.fastrtc.org/credentials",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    params={"ttl": 3600},
                )
                r.raise_for_status()
                return r.json()["iceServers"]
        except Exception as e:
            logger.warning(f"fastrtc TURN mint failed, using default ICE servers: {e}")

    return DEFAULT_ICE_SERVERS


def _ice_servers_from_dicts(servers: list[dict]) -> list[IceServer]:
    """Convert the dict shape used in the /start response into the aiortc
    dataclass shape SmallWebRTCConnection expects."""
    out: list[IceServer] = []
    for s in servers:
        urls = s.get("urls") or s.get("url")
        username = s.get("username")
        credential = s.get("credential")
        out.append(IceServer(urls=urls, username=username, credential=credential))
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


app = FastAPI(lifespan=lifespan)


async def run_bot(webrtc_connection: SmallWebRTCConnection, session_config: dict | None = None):
    logger.info("Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # SmallWebRTCTransport ignores vad_analyzer in TransportParams, so we add an
    # explicit VADProcessor: it emits VADUserStarted/StoppedSpeakingFrame, which
    # is what SegmentedSTTService (our AudioLLMService) needs to detect a turn
    # and fire run_stt. (The original demo got these frames via the
    # LLMContextAggregatorPair's vad_analyzer; we no longer use that aggregator.)
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4)))

    # R3: one audio-enabled Granite Switch model does audio -> answer in a single
    # request. No separate STT server, no Mellea LLM stage.
    audio_llm = AudioLLMService()
    if TTS_BACKEND == "hosted":
        tts = HostedTTSService(text_aggregation_mode=TextAggregationMode.SENTENCE)
    else:
        tts = _FastKokoroTTSService(
            settings=KokoroTTSService.Settings(voice=TTS_VOICE),
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            audio_llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/api/ivr/config")
async def ivr_config():
    """Static config the frontend needs before any turn runs — lets it pre-render
    the validation grid with the real requirement labels and sample count."""
    # R3 single-model path has no Best-of-N / IVR validation — report none so
    # the frontend renders no validation grid.
    return {"requirements": [], "nSamples": 1}


@app.post("/api/offer")
async def offer(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    return await _handle_offer(data, background_tasks)


@app.post("/start")
async def rtvi_start(request: Request):
    """RTVI /start endpoint — creates a session ID for the prebuilt UI."""
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    session_id = str(uuid.uuid4())
    session_config = request_data.get("body", {})
    if "ivrValidation" in request_data:
        session_config["ivr_validation"] = request_data["ivrValidation"]

    ice_servers_dicts = await _mint_ice_servers()
    session_config["_ice_servers"] = ice_servers_dicts
    active_sessions[session_id] = session_config

    result = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {"iceServers": ice_servers_dicts}
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_request(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    """RTVI session proxy — routes /sessions/{id}/api/offer to the offer handler."""
    if session_id not in active_sessions:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        data = await request.json()
        if request.method == "POST":
            session_config = active_sessions.get(session_id, {})
            return await _handle_offer(data, background_tasks, session_config=session_config)
        elif request.method == "PATCH":
            return await _handle_ice_candidate(data)

    return Response(status_code=200)


async def _handle_offer(data: dict, background_tasks: BackgroundTasks, session_config: dict | None = None):
    pc_id = data.get("pc_id")

    if pc_id and pc_id in pcs_map:
        conn = pcs_map[pc_id]
        await conn.renegotiate(
            sdp=data["sdp"],
            type=data["type"],
            restart_pc=data.get("restart_pc", False),
        )
    else:
        ice_servers_dicts = (session_config or {}).get("_ice_servers")
        if not ice_servers_dicts:
            ice_servers_dicts = await _mint_ice_servers()
        ice_servers = _ice_servers_from_dicts(ice_servers_dicts)
        conn = SmallWebRTCConnection(ice_servers)
        await conn.initialize(sdp=data["sdp"], type=data["type"])

        @conn.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, conn, session_config)

    answer = conn.get_answer()
    pcs_map[answer["pc_id"]] = conn
    return answer


async def _handle_ice_candidate(data: dict):
    from aiortc.sdp import candidate_from_sdp

    pc_id = data.get("pc_id")
    conn = pcs_map.get(pc_id)
    if not conn:
        return Response(content="Peer connection not found", status_code=404)

    for c in data.get("candidates", []):
        candidate = candidate_from_sdp(c["candidate"])
        candidate.sdpMid = c["sdp_mid"]
        candidate.sdpMLineIndex = c["sdp_mline_index"]
        await conn.add_ice_candidate(candidate)

    return {"status": "success"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mellea Pipecat Voice Server")
    parser.add_argument("--host", default=HOST, help=f"Host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
