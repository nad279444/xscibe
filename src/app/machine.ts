import type {
  AudioSource,
  Segment,
  Transcript,
  VoiceCaption,
  VoiceStatus,
} from './types'

export type AppPhase =
  | { status: 'idle' }
  | { status: 'ready' }
  | { status: 'decoding' }
  | { status: 'transcribing' }
  | { status: 'done' }
  | { status: 'liveLoading' }
  | { status: 'listening' }
  | { status: 'liveProcessing' }
  | { status: 'liveStopping' }
  | { status: 'voiceConnecting' }
  | { status: 'voiceActive' }
  | { status: 'error' }

export interface AppState {
  phase: AppPhase
  source: AudioSource | null
  transcript: Transcript | null
  partialText: string
  error: string | null
  liveSegments: Segment[]
  voiceStatus: VoiceStatus
  voiceCaptions: VoiceCaption[]
  requestId: number
}

export const initialState: AppState = {
  phase: { status: 'idle' },
  source: null,
  transcript: null,
  partialText: '',
  error: null,
  liveSegments: [],
  voiceStatus: 'idle',
  voiceCaptions: [],
  requestId: 0,
}

export type AppEvent =
  | { type: 'SELECT_SOURCE'; source: AudioSource }
  | { type: 'RESET' }
  | { type: 'SUBMIT_FILE' }
  | { type: 'TRANSCRIBE_STARTED' }
  | { type: 'TRANSCRIPTION_FINAL'; transcript: Transcript }
  | { type: 'TRANSCRIPTION_ERROR'; message: string }
  | { type: 'START_LIVE' }
  | { type: 'LIVE_READY' }
  | { type: 'LIVE_LISTENING' }
  | { type: 'LIVE_PARTIAL'; text: string }
  | { type: 'LIVE_SEGMENT'; segment: Segment }
  | { type: 'STOP_LIVE' }
  | { type: 'LIVE_TRANSCRIPT'; transcript: Transcript }
  | { type: 'LIVE_ERROR'; message: string }
  | { type: 'VOICE_CONNECTING' }
  | { type: 'VOICE_ACTIVE' }
  | { type: 'VOICE_CAPTION'; caption: VoiceCaption }
  | { type: 'VOICE_END' }
  | { type: 'VOICE_ERROR'; message: string }

function withRequestId(state: AppState): AppState {
  return { ...state, requestId: state.requestId + 1 }
}

export function reducer(state: AppState, event: AppEvent): AppState {
  switch (event.type) {
    case 'SELECT_SOURCE':
      return {
        ...initialState,
        source: event.source,
        phase: { status: 'ready' },
      }

    case 'RESET':
      return initialState

    case 'SUBMIT_FILE':
      if (!state.source || state.phase.status !== 'ready') return state
      return withRequestId({ ...state, phase: { status: 'decoding' }, error: null })

    case 'TRANSCRIBE_STARTED': {
      if (
        state.phase.status === 'decoding' ||
        state.phase.status === 'transcribing'
      ) {
        return { ...state, phase: { status: 'transcribing' } }
      }
      return state
    }

    case 'TRANSCRIPTION_FINAL':
      if (state.phase.status !== 'transcribing') return state
      return {
        ...state,
        phase: { status: 'done' },
        transcript: event.transcript,
        partialText: '',
      }

    case 'TRANSCRIPTION_ERROR':
    case 'LIVE_ERROR':
      return {
        ...state,
        phase: { status: 'error' },
        error: event.message,
        partialText: '',
      }

    case 'START_LIVE':
      return withRequestId({
        ...state,
        phase: { status: 'liveLoading' },
        transcript: null,
        partialText: '',
        liveSegments: [],
        error: null,
      })

    case 'LIVE_READY':
      if (state.phase.status !== 'liveLoading') return state
      return { ...state, phase: { status: 'listening' } }

    case 'LIVE_LISTENING':
      if (state.phase.status !== 'listening' && state.phase.status !== 'liveProcessing') return state
      return { ...state, phase: { status: 'listening' }, partialText: '' }

    case 'LIVE_PARTIAL':
      if (state.phase.status !== 'listening' && state.phase.status !== 'liveProcessing') return state
      return { ...state, phase: { status: 'liveProcessing' }, partialText: event.text }

    case 'LIVE_SEGMENT': {
      if (
        state.phase.status !== 'listening' &&
        state.phase.status !== 'liveProcessing' &&
        state.phase.status !== 'liveStopping'
      ) {
        return state
      }
      return {
        ...state,
        phase:
          state.phase.status === 'liveStopping'
            ? state.phase
            : { status: 'listening' as const },
        liveSegments: [
          ...state.liveSegments,
          { ...event.segment, id: state.liveSegments.length },
        ],
        partialText: '',
      }
    }

    case 'STOP_LIVE':
      if (state.phase.status !== 'listening' && state.phase.status !== 'liveProcessing') {
        return state
      }
      return { ...state, phase: { status: 'liveStopping' }, partialText: '' }

    case 'LIVE_TRANSCRIPT':
      if (state.phase.status !== 'liveStopping' && state.phase.status !== 'listening' && state.phase.status !== 'liveProcessing') {
        return state
      }
      return {
        ...state,
        phase: { status: 'done' },
        transcript: event.transcript,
        partialText: '',
      }

    case 'VOICE_CONNECTING':
      return {
        ...state,
        phase: { status: 'voiceConnecting' },
        voiceStatus: 'connecting',
        voiceCaptions: [],
        error: null,
      }

    case 'VOICE_ACTIVE':
      if (state.phase.status !== 'voiceConnecting') return state
      return { ...state, phase: { status: 'voiceActive' }, voiceStatus: 'active' }

    case 'VOICE_CAPTION': {
      if (state.phase.status !== 'voiceActive' && state.phase.status !== 'voiceConnecting') return state
      const captions = [...state.voiceCaptions]
      const last = captions[captions.length - 1]
      if (last && last.role === event.caption.role && !last.final) {
        captions[captions.length - 1] = event.caption
      } else if (!(event.caption.final && event.caption.text === '')) {
        captions.push(event.caption)
      }
      return { ...state, voiceCaptions: captions }
    }

    case 'VOICE_END':
      return {
        ...state,
        phase: state.transcript ? { status: 'done' } : { status: 'idle' },
        voiceStatus: 'ended',
      }

    case 'VOICE_ERROR':
      return {
        ...state,
        phase: { status: 'error' },
        voiceStatus: 'error',
        error: event.message,
      }

    default: {
      const exhaustive: never = event
      throw new Error(`Unhandled event: ${JSON.stringify(exhaustive)}`)
    }
  }
}
