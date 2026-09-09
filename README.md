# Voice Agent POC (showcase excerpt)

A proof-of-concept real-time voice AI agent built on [Pipecat](https://www.pipecat.ai/), with a
custom ElevenLabs v3 TTS integration. This repo shows the **voice pipeline wiring** — how audio
streams through STT → LLM → a processing stage → TTS in real time, and the React client that
connects to it — as a demonstration of real-time voice AI engineering, not the emotional processing
stage itself.

This is a curated excerpt: the pipeline references an `EnhancedEmotionalProcessor` stage that lives
in the private codebase, so this repo is for reading, not running.

## What's included here

- **`server/bot.py`** — the Pipecat pipeline construction: wiring STT, LLM, an emotional-processing
  stage, and TTS together into a real-time voice agent, plus session lifecycle handling.
- **`server/server.py`** — the server entry point.
- **`server/elevenlabs_v3_tts.py`** — a custom TTS service integration for ElevenLabs' v3 API.
- **`client/`** — the Pipecat React voice client (audio capture/playback UI).

## What was built but isn't shown here

- **The emotional processing stage** (`emotional_evaluator.py`, `emotional_processor.py`,
  `enhanced_emotional_processor.py`, `emotional_frame_processor.py`) — real-time emotion detection
  and response shaping applied to the voice conversation. `bot.py` instantiates and wires this stage
  into the pipeline but doesn't contain its logic.
- Deployment infrastructure (EC2/CloudFormation, Cloudflare Tunnel, nginx config) for this POC.

I'm happy to walk through the design of the emotional processing stage in conversation — it's just
not published as code.

## Stack

Python (Pipecat) real-time voice pipeline, ElevenLabs TTS, React voice client.
