# granite-speech-demo

A real-time voice agent demo built on [Pipecat](https://github.com/pipecat-ai/pipecat). The user speaks; an audio-capable Granite Switch model transcribes and answers in a single request; Kokoro speaks the reply back.

```
Browser mic → WebRTC → Silero VAD → Granite Switch Audio (transcription + generation) → Kokoro TTS → WebRTC → Browser speaker
```

A single vLLM-served model handles both transcription and generation via the OpenAI audio input API (`input_audio` content part). No separate STT server is required.

Barge-in (interrupting the bot mid-response) is handled by Pipecat's `InterruptionFrame` propagation.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 18+ (for the Next.js frontend)
- One vLLM server serving an audio-capable Granite Switch checkpoint (requires an NVIDIA GPU). See [below](#serving-the-model).

## Setup

```bash
cp .env.example .env   # edit LLM_URL / LLM_MODEL if your server differs
uv sync
```

## Serving the model

The demo needs one vLLM server hosting an audio-capable Granite Switch checkpoint.

### Locally

```bash
vllm serve <your-granite-switch-audio-checkpoint> --port 8000
```

Then confirm `.env` points at it:

```
LLM_URL=http://localhost:8000/v1
LLM_MODEL=granite-switch-audio   # or whatever model ID vLLM reports
```

### Via OpenShift Route (recommended)

If the model is deployed on an OpenShift cluster, expose it as a Route — this is more stable than port-forwarding.

Find the service name for your vLLM pod, then create a route:

```bash
oc get svc                                    # find the service name
oc expose svc/<service-name> --port=8000      # create a route
oc get route <service-name> -o jsonpath='{.spec.host}'  # get the hostname
```

Set `LLM_URL` in `.env` to the route hostname:

```
LLM_URL=http://<route-hostname>/v1
```

If the route is TLS-terminated use `https://` instead.

### Via OpenShift port-forward (less stable)

Port-forwarding is handy for a quick test but the tunnel can drop mid-demo. Prefer the Route approach above for anything beyond a one-off check.

```bash
oc port-forward <pod-name> 8000:8000
```

Keep that terminal open. Your `.env` defaults already point at the forwarded port:

```
LLM_URL=http://localhost:8000/v1
```

## Run

Open two terminals in the repo root:

```bash
# Terminal 1 — Pipecat backend (http://localhost:7860)
./run-backend.sh

# Terminal 2 — Next.js frontend (http://localhost:3000)
./run-frontend.sh
```

Open **http://localhost:3000**, click **Connect**, and start talking. Ctrl+C in either terminal shuts down that process cleanly.

`run-backend.sh` exits immediately if no `.env` is found and prints a reminder to copy `.env.example`.  
`run-frontend.sh` runs `npm install` automatically on first use.

### Backend only (built-in Pipecat UI)

```bash
uv run python -m granite_speech_demo.server
```

Open http://localhost:7860 — redirects to the built-in Pipecat prebuilt UI at `/client/`.

To serve over HTTPS (required for microphone access from non-localhost origins):

```bash
uv run python -m granite_speech_demo.server --ssl-certfile cert.pem --ssl-keyfile key.pem
```

## Configuration

All settings are in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `HOST` | `localhost` | Server bind address |
| `PORT` | `7860` | Server port |
| `LLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint for the audio LLM |
| `LLM_MODEL` | `granite-switch-audio` | Model ID passed in each request |
| `LLM_API_KEY` | `EMPTY` | Bearer token for the LLM endpoint (not required for local vLLM) |
| `AUDIO_LLM_MAX_TOKENS` | `256` | Max tokens per LLM response |
| `TTS_BACKEND` | `kokoro` | TTS backend: `kokoro` (local) or `hosted` (remote HTTP server) |
| `TTS_VOICE` | `af_aoede` | Kokoro voice ID |
| `TTS_SPEED` | `1.25` | Kokoro speaking tempo (1.0 = normal speed) |
| `HOSTED_TTS_URL` | `http://localhost:8086` | Base URL of a remote TTS server (used when `TTS_BACKEND=hosted`) |
| `HOSTED_TTS_PATH` | `/synth` | Path that accepts `POST {"text": "..."}` and streams raw PCM back |
| `HOSTED_TTS_SAMPLE_RATE` | `24000` | PCM sample rate returned by the hosted TTS server |
| `PROMPT_FILE` | _(unset)_ | Path to a `.txt` file that replaces the default system prompt. See `prompts/granite.txt` for the THINK 2026 demo persona. |
| `DOCUMENTS_DIR` | _(unset)_ | Directory of `.txt` files loaded at startup and injected into the system prompt as grounding documents. |
| `HF_TOKEN` | _(unset)_ | HuggingFace read token. When set, the backend mints per-session TURN credentials from `turn.fastrtc.org` so WebRTC works across NAT. Optional for localhost dev. |

## Make it your own

Two levers, in increasing order of invasiveness:

- **Persona — `PROMPT_FILE`.** A `.txt` file whose contents replace the default system prompt. `prompts/granite.txt` is the THINK 2026 example persona.
- **Grounding — `DOCUMENTS_DIR`.** A folder of `.txt` files loaded at startup and embedded in the system prompt inside `<documents>` tags. Use it to anchor answers to your own product docs, FAQ, or knowledge base.

## Project structure

```
src/granite_speech_demo/
├── server.py        # FastAPI + SmallWebRTC signaling + pipeline wiring
├── audio_llm.py     # AudioLLMService — buffers audio, sends input_audio request to Granite Switch, streams answer frames
└── hosted_tts.py    # HostedTTSService — POSTs text to a remote TTS server, streams PCM back

frontend/            # Next.js app (Carbon Design System, IBM Plex fonts)
├── app/
│   ├── page.tsx             # Single-page demo: top bar + embedded voice UI
│   ├── layout.tsx           # Root layout (fonts, global styles)
│   ├── components/
│   │   └── GraniteSpeechDemo.tsx  # Pipecat voice-ui-kit embed
│   └── api/                 # Next.js proxy routes to the Pipecat backend
│       ├── pipecat/start/
│       ├── offer/[sessionId]/
│       └── ivr/config/
├── config.ts        # PIPECAT_BACKEND_URL setting
└── package.json
```

**server.py** wires the Pipecat pipeline and the FastAPI endpoints (RTVI protocol: `/start`, `/sessions/{id}/api/offer`). Each WebRTC connection spawns its own pipeline:

```
transport.input → VADProcessor (Silero) → AudioLLMService → Kokoro TTS → transport.output
```

**audio_llm.py** subclasses Pipecat's `SegmentedSTTService` to reuse its audio-buffering and VAD turn boundaries. On each user turn it base64-encodes the buffered WAV, posts it to the Granite Switch audio endpoint with an `input_audio` content part, and streams the answer as `LLMTextFrame`s to TTS. It also implements barge-in: a new `VADUserStartedSpeakingFrame` cancels any in-flight generation.

## Dependencies

- **[Pipecat AI](https://github.com/pipecat-ai/pipecat)** — pipeline orchestration (WebRTC, Silero VAD, STT/TTS services, SmartTurn)
- **[IBM Granite Switch Audio](https://huggingface.co/ibm-granite)** — single model checkpoint for transcription + chat
