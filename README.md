# XScribe

Speech → text → answers → voice, entirely in your browser.

Upload or record audio (or transcribe live from your microphone), get a word-timestamped
transcript with a hallucination guard, export SRT/VTT/JSON/TXT, ask questions about it with
a streaming LLM, or have a real-time voice conversation about it.

## Features

| Area | What it does |
| --- | --- |
| File transcription | Whisper (`tiny`/`base`/`small`.en) runs locally via `@huggingface/transformers` — WebGPU when available, WASM fallback. Nothing audio-related leaves the machine. |
| Word timestamps | Every word carries start/end times. Click a word in the transcript to seek the audio player; the current word highlights during playback. |
| Live transcription | Mic audio is segmented by Silero VAD (`@ricky0123/vad-web`); each utterance is transcribed with a rolling offset and streamed into the transcript as you speak. |
| Hallucination guard | Segments are scored for known Whisper artifacts ("Thank you for watching…"), n-gram loops, impossible speech rates and dead air. Flagged segments are visibly marked and can be hidden or stripped on export. |
| Exports | SRT, WebVTT, JSON (per-word) and plain text. |
| Ask the transcript | Streaming Q&A over the Responses API (`gpt-6-luna`, prompt caching, `reasoning.effort: low`), grounded strictly in the transcript text. |
| Voice chat | GPT-Live over WebRTC: one stateless edge function exchanges SDP, then your mic and the model's speech stream peer-to-peer. Live captions render alongside the audio. |
| State handling | A strict `useReducer` state machine (`src/app/machine.ts`) owns every phase — the UI is a pure function of state, with exhaustive event checking at compile time. |
| Worker protocol | The ASR worker and the app talk over typed messages only (`src/protocol/messages.ts`); queued jobs, per-request cancellation and WebGPU→WASM auto-fallback live in the worker. |

## Architecture

```
src/
  app/machine.ts        state machine: AppPhase / AppState / AppEvent (exhaustive union)
  protocol/messages.ts  typed worker protocol (AsrRequest / AsrResponse)
  utils/asr.worker.ts   model registry, job queue, streaming partials, cancellation
  lib/asr/              model registry, device detection, hallucination guard
  lib/audio/            decode/resample, VAD capture (live mode)
  lib/export/           SRT / VTT / JSON / TXT renderers
  lib/openai/           streaming chat client, WebRTC Live session
  components/           one component per screen, no business logic
  app/api/               Next.js route handlers (chat, live SDP exchange, rate-limited)
```

- **Model downloads** stream from the Hugging Face Hub into the browser (first run only,
  cached by the browser). Model size and device are shown in the UI.
- **Rate limiting** for the edge functions is a per-instance token bucket
  (`src/lib/server/rateLimit.ts`) — good enough for a demo, not a substitute for a real quota system.

## Development

```bash
npm install
npm run dev        # http://localhost:3000 — transcription works without any API key
npm run typecheck  # tsc --noEmit
npm test           # vitest (machine, guard, exports)
npm run lint       # eslint
npm run build      # typecheck + production build
```

Transcription, live mode, exports and the guard are fully offline. The **Ask** and
**Voice** tabs need OpenAI credentials:

```bash
# .env.local (never commit; the API key is only read by server routes)
OPENAI_API_KEY=sk-...
# optional overrides
CHAT_MODEL=gpt-6-luna
LIVE_MODEL=gpt-live-1
LIVE_VOICE=marin
```

The API key is only ever read server-side in `src/app/api/` — it is never shipped to the browser.

## Deploy (Vercel)

```bash
vercel
```

Set `OPENAI_API_KEY` (and optionally the model overrides) in the project's Environment
Variables. The API endpoints are Next.js route handlers under `src/app/api/`.

Keep the OpenAI spend cap low (≈ $3) — voice sessions bill per second while connected.

## Notes & trade-offs

- Translation was deliberately dropped: language detection added surface area without
  strengthening the core transcript → answer → voice story.
- `public/` self-hosts the VAD worklet + Silero model so live mode does not depend on a CDN.
- Font Awesome and Google Fonts still load from CDNs in `src/app/layout.tsx`; self-hosting them is
  an easy follow-up hardening step.
