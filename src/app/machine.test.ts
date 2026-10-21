import { describe, expect, it } from 'vitest'
import { initialState, reducer, type AppEvent, type AppState } from './machine'
import type { Segment, Transcript } from './types'

function source() {
  const file = new File(['x'], 'meeting.wav')
  return { kind: 'file' as const, file, name: 'meeting.wav' }
}

function transcript(): Transcript {
  const segment: Segment = {
    id: 0,
    text: 'Hello world.',
    start: 0,
    end: 2,
    words: [{ text: 'Hello', start: 0, end: 2 }],
    flagged: false,
    flagReasons: [],
    confidence: 1,
  }
  return { text: 'Hello world.', duration: 2, segments: [segment] }
}

function ready(): AppState {
  return reducer(initialState, { type: 'SELECT_SOURCE', source: source() })
}

describe('file transcription flow', () => {
  it('starts idle', () => {
    expect(initialState.phase).toEqual({ status: 'idle' })
  })

  it('moves to ready when a source is selected', () => {
    const state = ready()
    expect(state.phase).toEqual({ status: 'ready' })
    expect(state.source?.name).toBe('meeting.wav')
  })

  it('decodes on SUBMIT_FILE and bumps the request id', () => {
    const state = ready()
    const next = reducer(state, { type: 'SUBMIT_FILE' })
    expect(next.phase).toEqual({ status: 'decoding' })
    expect(next.requestId).toBe(state.requestId + 1)
  })

  it('ignores SUBMIT_FILE without a source', () => {
    const next = reducer(initialState, { type: 'SUBMIT_FILE' })
    expect(next.phase).toEqual({ status: 'idle' })
  })

  it('moves decoding → loadingModel → transcribing on progress', () => {
    let state = reducer(ready(), { type: 'SUBMIT_FILE' })
    state = reducer(state, {
      type: 'WORKER_PROGRESS',
      progress: { phase: 'download', progress: 42 },
    })
    expect(state.phase).toEqual({ status: 'loadingModel' })
    state = reducer(state, { type: 'TRANSCRIBE_STARTED' })
    state = reducer(state, {
      type: 'WORKER_PROGRESS',
      progress: { phase: 'warmup' },
    })
    expect(state.phase).toEqual({ status: 'transcribing' })
  })

  it('stores partial text only while transcribing', () => {
    let state = reducer(ready(), { type: 'SUBMIT_FILE' })
    state = reducer(state, { type: 'WORKER_PARTIAL', text: 'ignored' })
    expect(state.partialText).toBe('')
    state = reducer(state, { type: 'TRANSCRIBE_STARTED' })
    state = reducer(state, { type: 'WORKER_PARTIAL', text: 'Hello' })
    expect(state.partialText).toBe('Hello')
  })

  it('completes on WORKER_FINAL and clears transient state', () => {
    let state = reducer(ready(), { type: 'SUBMIT_FILE' })
    state = reducer(state, { type: 'TRANSCRIBE_STARTED' })
    state = reducer(state, { type: 'WORKER_PARTIAL', text: 'partial' })
    state = reducer(state, { type: 'WORKER_FINAL', transcript: transcript() })
    expect(state.phase).toEqual({ status: 'done' })
    expect(state.transcript?.segments).toHaveLength(1)
    expect(state.partialText).toBe('')
    expect(state.progress).toBeNull()
  })

  it('surfaces worker errors', () => {
    let state = reducer(ready(), { type: 'SUBMIT_FILE' })
    state = reducer(state, { type: 'WORKER_ERROR', message: 'boom' })
    expect(state.phase).toEqual({ status: 'error' })
    expect(state.error).toBe('boom')
  })
})

describe('reset and model selection', () => {
  it('RESET returns to idle but keeps the model choice', () => {
    let state = reducer(initialState, { type: 'SET_MODEL', modelKey: 'tiny' })
    state = reducer(state, { type: 'SELECT_SOURCE', source: source() })
    state = reducer(state, { type: 'RESET' })
    expect(state.phase).toEqual({ status: 'idle' })
    expect(state.modelKey).toBe('tiny')
    expect(state.source).toBeNull()
  })

  it('records the detected device', () => {
    const state = reducer(initialState, { type: 'DEVICE_DETECTED', device: 'webgpu' })
    expect(state.device).toBe('webgpu')
  })
})

describe('live transcription flow', () => {
  function listening(): AppState {
    let state = reducer(initialState, { type: 'START_LIVE' })
    expect(state.phase).toEqual({ status: 'liveLoading' })
    state = reducer(state, { type: 'LIVE_READY' })
    expect(state.phase).toEqual({ status: 'listening' })
    return state
  }

  it('moves liveLoading → listening on LIVE_READY', () => {
    listening()
  })

  it('ignores LIVE_READY outside liveLoading', () => {
    const state = reducer(initialState, { type: 'LIVE_READY' })
    expect(state.phase).toEqual({ status: 'idle' })
  })

  it('streams partials as liveProcessing and commits segments back to listening', () => {
    let state = listening()
    state = reducer(state, { type: 'LIVE_PARTIAL', text: 'hello' })
    expect(state.phase).toEqual({ status: 'liveProcessing' })

    const segment: Segment = {
      id: 99,
      text: 'hello there',
      start: 0,
      end: 1.5,
      words: [{ text: 'hello', start: 0, end: 1.5 }],
      flagged: false,
      flagReasons: [],
      confidence: 1,
    }
    state = reducer(state, { type: 'LIVE_SEGMENT', segment })
    expect(state.phase).toEqual({ status: 'listening' })
    expect(state.partialText).toBe('')
    expect(state.liveSegments).toHaveLength(1)
    expect(state.liveSegments[0]?.id).toBe(0)
  })

  it('keeps accepting segments after STOP_LIVE while draining', () => {
    let state = listening()
    state = reducer(state, { type: 'STOP_LIVE' })
    expect(state.phase).toEqual({ status: 'liveStopping' })

    const segment: Segment = {
      id: 0,
      text: 'last words',
      start: 2,
      end: 3,
      words: [{ text: 'last', start: 2, end: 3 }],
      flagged: false,
      flagReasons: [],
      confidence: 1,
    }
    state = reducer(state, { type: 'LIVE_SEGMENT', segment })
    expect(state.phase).toEqual({ status: 'liveStopping' })
    expect(state.liveSegments).toHaveLength(1)
  })

  it('finishes with LIVE_TRANSCRIPT only from an active live phase', () => {
    const blocked = reducer(initialState, { type: 'LIVE_TRANSCRIPT', transcript: transcript() })
    expect(blocked.phase).toEqual({ status: 'idle' })

    let state = listening()
    state = reducer(state, { type: 'STOP_LIVE' })
    state = reducer(state, { type: 'LIVE_TRANSCRIPT', transcript: transcript() })
    expect(state.phase).toEqual({ status: 'done' })
    expect(state.transcript).not.toBeNull()
  })
})

describe('voice chat flow', () => {
  it('walks connecting → active → end and returns to the transcript', () => {
    let state: AppState = { ...ready(), phase: { status: 'done' }, transcript: transcript() }
    state = reducer(state, { type: 'VOICE_CONNECTING' })
    expect(state.phase).toEqual({ status: 'voiceConnecting' })
    expect(state.voiceStatus).toBe('connecting')

    state = reducer(state, { type: 'VOICE_ACTIVE' })
    expect(state.phase).toEqual({ status: 'voiceActive' })

    state = reducer(state, {
      type: 'VOICE_CAPTION',
      caption: { role: 'assistant', text: 'Hi', final: false },
    })
    state = reducer(state, {
      type: 'VOICE_CAPTION',
      caption: { role: 'assistant', text: 'Hi there', final: false },
    })
    expect(state.voiceCaptions).toHaveLength(1)
    expect(state.voiceCaptions[0]?.text).toBe('Hi there')

    state = reducer(state, { type: 'VOICE_END' })
    expect(state.phase).toEqual({ status: 'done' })
    expect(state.voiceStatus).toBe('ended')
  })

  it('returns to idle after voice ends without a transcript', () => {
    let state = reducer(initialState, { type: 'VOICE_CONNECTING' })
    state = reducer(state, { type: 'VOICE_ACTIVE' })
    state = reducer(state, { type: 'VOICE_END' })
    expect(state.phase).toEqual({ status: 'idle' })
  })

  it('reports voice errors', () => {
    let state = reducer(initialState, { type: 'VOICE_CONNECTING' })
    state = reducer(state, { type: 'VOICE_ERROR', message: 'denied' })
    expect(state.phase).toEqual({ status: 'error' })
    expect(state.error).toBe('denied')
  })
})

describe('reducer robustness', () => {
  it('throws on unhandled events to keep the union exhaustive', () => {
    expect(() =>
      reducer(initialState, { type: 'NOT_A_REAL_EVENT' } as unknown as AppEvent),
    ).toThrow()
  })
})
