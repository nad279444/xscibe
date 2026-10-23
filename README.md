# XScribe

Cloud-powered speech-to-text, transcript chat, and voice conversations.

Upload or record audio (or transcribe live from your microphone), get a transcript with
approximate word seek positions and a hallucination guard, export SRT/VTT/JSON/TXT, ask questions about it with
a streaming LLM, or have a real-time voice conversation about it.

## Features

| Area | What it does |
| --- | --- |
| File transcription | Audio up to 20 MB uploads directly to a private Google Cloud Storage bucket, then the Next.js server sends it to OpenAI GPT-Transcribe. |
| Word seek positions | GPT-Transcribe does not return word timestamps; XScribe estimates positions across the audio duration. Click a word to seek approximately. |
| Live transcription | Microphone audio streams through an AudioWorklet to OpenAI Realtime using an ephemeral session token. |
| Hallucination guard | Segments are scored for common ASR artifacts ("Thank you for watching…"), n-gram loops, impossible speech rates and dead air. Flagged segments are visibly marked and can be hidden or stripped on export. |
| Exports | SRT, WebVTT, JSON (per-word) and plain text. |
| Ask the transcript | Streaming Q&A over the Responses API (`gpt-6-luna`, prompt caching, `reasoning.effort: low`), grounded strictly in the transcript text. |
| Voice chat | GPT-Live over WebRTC: one stateless edge function exchanges SDP, then your mic and the model's speech stream peer-to-peer. Live captions render alongside the audio. |
| Robot-inspired dashboard | The landing screen puts file transcription, microphone recording, and live transcription up front, with chat and voice features previewed. |
| Transcript chat | Chat opens as the first result view. Ask questions grounded in the transcript, then switch to word-timed text, exports, or voice chat. |
| State handling | A strict `useReducer` state machine (`src/app/machine.ts`) owns every phase — the UI is a pure function of state, with exhaustive event checking at compile time. |
| Privacy | Audio is temporarily stored in a private GCS bucket and sent to OpenAI for transcription. API credentials remain server-side. |

## Architecture

```
src/
  app/machine.ts        state machine: AppPhase / AppState / AppEvent (exhaustive union)
  lib/asr/              transcript hallucination guard
  lib/audio/            upload limits and AudioWorklet live capture
  lib/export/           SRT / VTT / JSON / TXT renderers
  lib/openai/           browser transcription client, Realtime, chat, and voice clients
  lib/server/           GCS bucket client and API rate limiting
  components/           one component per screen, no business logic
  app/api/              GCS upload tickets, transcription, chat, and live sessions
```

- **File transcription flow:** the browser requests an upload ticket from `POST /api/transcribe/upload` with file metadata. The route validates the size/type, rate-limits ticket requests, and returns a random object path plus a 10-minute GCS signed `PUT` URL. The browser uploads directly to the private bucket, then posts the object path to `POST /api/transcribe`. The server validates the path and stored size, downloads and deletes the object, forwards the audio to OpenAI, normalizes the response, and applies the hallucination guard. Audio is capped at 20 MB. Direct multipart requests to `/api/transcribe` remain limited to 4 MB; the app uses GCS for its normal upload flow ([Vercel function limits](https://vercel.com/docs/functions/limitations), [GCS signed URLs](https://cloud.google.com/storage/docs/access-control/signed-urls)).
- **Transcription runtime:** `/api/transcribe` allows up to 300 seconds for server-side transcription, subject to the hosting platform's function-duration limits. GCS objects are deleted during normal processing; configure a bucket lifecycle rule to remove old `audio-uploads/` objects as a safety net for abandoned or interrupted uploads.
- **Chat flow:** the browser posts the transcript, question, and recent history to `/api/chat`. The route bounds transcript/question/history sizes, rate-limits requests, and streams text deltas from the OpenAI Responses API to the UI. Chat is grounded in the supplied transcript and has a configured maximum runtime of 300 seconds.
- **Live transcription** streams 24 kHz PCM over a persistent browser WebSocket. A Next.js route mints a short-lived Realtime client secret; the permanent API key remains server-side. An AudioWorklet performs lightweight energy-based turn detection and commits audio turns.
- **Model/VAD compatibility:** `gpt-live-transcribe` uses client-side turn commits, with lightweight energy detection in the AudioWorklet rather than ONNX inference or MicVAD on the main thread. See the [Realtime transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription).
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

File transcription, live mode, transcript chat, and voice require an OpenAI API key. GCS-backed file uploads also require a Google Cloud service account and private bucket:

```bash
# .env.local (never commit; credentials are only read by server routes)
OPENAI_API_KEY=sk-...
GCS_BUCKET_NAME=your-private-audio-bucket
GCS_PROJECT_ID=your-gcp-project-id
CLIENT_EMAIL=your-service-account@your-gcp-project-id.iam.gserviceaccount.com
PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
# optional overrides
CHAT_MODEL=gpt-6-luna
LIVE_MODEL=gpt-live-1
LIVE_VOICE=marin
```

OpenAI and GCS credentials are only read server-side and are never shipped to the browser. Transcription requires network access and may incur OpenAI API and GCS storage/operation charges. Browser GPU availability no longer affects file-transcription speed; inference runs on OpenAI's service.

## Deploy (Vercel)

```bash
vercel
```

Set `OPENAI_API_KEY`, `GCS_BUCKET_NAME`, `GCS_PROJECT_ID`, `CLIENT_EMAIL`, and `PRIVATE_KEY` in the project's Environment Variables (Production, Preview, and Development as needed). `PRIVATE_KEY` must be the PEM private key from the service-account JSON. The service account needs permission to create, read, and delete objects in the bucket. Configure bucket CORS to allow `PUT` from your deployed app origin (and `localhost` for local development), with the `Content-Type` request header. Use a bucket lifecycle rule to remove abandoned uploads. File transcription uses `gpt-transcribe`; live transcription uses the Realtime API.

Keep the OpenAI spend cap low (≈ $3) — voice sessions bill per second while connected.

## Notes & trade-offs

- Translation was deliberately dropped: language detection added surface area without
  strengthening the core transcript → answer → voice story.
- Live transcription no longer loads ONNX Runtime, Silero VAD, or browser ASR models.
- Font Awesome and Google Fonts still load from CDNs in `src/app/layout.tsx`; self-hosting them is
  an easy follow-up hardening step.
