# XScribe

Cloud-powered speech-to-text, transcript chat, and voice conversations.

Upload or record audio (or transcribe live from your microphone), get a transcript with
approximate word seek positions and a hallucination guard, export SRT/VTT/JSON/TXT, ask questions about it with
a streaming LLM, or have a real-time voice conversation about it.

## Features

| Area | What it does |
| --- | --- |
| File transcription | OpenAI GPT-Transcribe runs behind a server-side Next.js route. Uploads are limited to 4 MB on Vercel. |
| Word seek positions | GPT-Transcribe does not return word timestamps; XScribe estimates positions across the audio duration. Click a word to seek approximately. |
| Live transcription | Microphone audio streams through an AudioWorklet to OpenAI Realtime using an ephemeral session token. |
| Hallucination guard | Segments are scored for common ASR artifacts ("Thank you for watching…"), n-gram loops, impossible speech rates and dead air. Flagged segments are visibly marked and can be hidden or stripped on export. |
| Exports | SRT, WebVTT, JSON (per-word) and plain text. |
| Ask the transcript | Streaming Q&A over the Responses API (`gpt-6-luna`, prompt caching, `reasoning.effort: low`), grounded strictly in the transcript text. |
| Voice chat | GPT-Live over WebRTC: one stateless edge function exchanges SDP, then your mic and the model's speech stream peer-to-peer. Live captions render alongside the audio. |
| Robot-inspired dashboard | The landing screen puts file transcription, microphone recording, and live transcription up front, with chat and voice features previewed. |
| Transcript chat | Chat opens as the first result view. Ask questions grounded in the transcript, then switch to word-timed text, exports, or voice chat. |
| State handling | A strict `useReducer` state machine (`src/app/machine.ts`) owns every phase — the UI is a pure function of state, with exhaustive event checking at compile time. |
| Privacy | Audio is sent to OpenAI for transcription. The API key stays server-side; avoid uploading audio you are not authorized to share. |

## Architecture

```
src/
  app/machine.ts        state machine: AppPhase / AppState / AppEvent (exhaustive union)
  lib/asr/              transcript hallucination guard
  lib/audio/            AudioWorklet live capture
  lib/export/           SRT / VTT / JSON / TXT renderers
  lib/openai/           transcription, Realtime live stream, chat, and WebRTC voice
  components/           one component per screen, no business logic
  app/api/               Next.js route handlers (transcribe, chat, live SDP exchange)
```

- **File uploads** use `gpt-transcribe` via `/api/transcribe`; the route keeps `OPENAI_API_KEY` server-side. GPT-Transcribe returns text but not word timestamps, so the UI estimates word positions across the audio duration ([OpenAI transcription guide](https://developers.openai.com/api/docs/guides/transcription)).
- **Upload size** is capped at 4 MB so multipart requests fit Vercel Functions' 4.5 MB body limit. OpenAI accepts larger files, but routing them through a Vercel Function exceeds Vercel's payload limit ([Vercel function limits](https://vercel.com/docs/functions/limitations)).
- **Live transcription** streams 24 kHz PCM over a persistent browser WebSocket. A Next.js route mints a short-lived Realtime client secret; the permanent API key remains server-side. An AudioWorklet performs lightweight energy-based turn detection and commits audio turns.
- **Model/VAD compatibility:** `gpt-live-transcribe` does not support server VAD and requires the client to commit each turn. XScribe keeps this low-cost detection in the AudioWorklet rather than loading an ONNX model or running MicVAD on the main thread. See the [Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription).
- **Rate limiting** for API routes is a per-instance token bucket
  (`src/lib/server/rateLimit.ts`) — good enough for a demo, not a substitute for a real quota system.

## Development

```bash
npm install
npm run dev        # http://localhost:3000
npm run typecheck  # tsc --noEmit
npm test           # vitest (machine, guard, exports)
npm run lint       # eslint
npm run build      # typecheck + production build
```

Transcription, live mode, transcript chat, and voice all require an OpenAI API key:

```bash
# .env.local (never commit; the API key is only read by server routes)
OPENAI_API_KEY=sk-...
# optional overrides
CHAT_MODEL=gpt-6-luna
LIVE_MODEL=gpt-live-1
LIVE_VOICE=marin
```

The API key is only read server-side in `src/app/api/` and is never shipped to the browser. Transcription requires network access and may incur OpenAI API charges. Browser GPU availability no longer affects transcription speed; inference runs on OpenAI's service.

## Deploy (Vercel)

```bash
vercel
```

Set `OPENAI_API_KEY` (and optionally the chat/voice model overrides) in the project's Environment
Variables. File transcription uses `gpt-transcribe`; live transcription uses the Realtime API.

Keep the OpenAI spend cap low (≈ $3) — voice sessions bill per second while connected.

## Notes & trade-offs

- Translation was deliberately dropped: language detection added surface area without
  strengthening the core transcript → answer → voice story.
- Live transcription no longer loads ONNX Runtime, Silero VAD, or browser ASR models.
- Font Awesome and Google Fonts still load from CDNs in `src/app/layout.tsx`; self-hosting them is
  an easy follow-up hardening step.
