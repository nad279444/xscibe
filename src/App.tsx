'use client'

import { useEffect, useMemo, useReducer, useRef } from 'react'
import { initialState, reducer } from './app/machine'
import type { AsrRequest, AsrResponse } from './protocol/messages'
import type { Segment } from './app/types'
import { detectDevice } from './lib/asr/device'
import { analyzeTranscript } from './lib/asr/guard'
import { decodeAudio } from './lib/audio/decode'
import { startLiveCapture, type LiveCapture } from './lib/audio/liveCapture'
import { startLiveSession, type LiveSession } from './lib/openai/live'
import Header from './components/Header'
import HomePage from './components/HomePage'
import FileDisplay from './components/FileDisplay'
import Transcribing from './components/Transcribing'
import Information from './components/Information'
import LiveView from './components/LiveView'
import VoiceView from './components/VoiceView'

function formatShortTimestamp(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(safe / 60)
  const secs = safe % 60
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState)

  const stateRef = useRef(state)
  stateRef.current = state

  const workerRef = useRef<Worker | null>(null)
  const nextRequestId = useRef(0)
  const fileRequestId = useRef(-1)
  const liveActive = useRef(false)
  const stoppingLive = useRef(false)
  const pendingUtterances = useRef(0)
  const liveSegmentsRef = useRef<Segment[]>([])
  const captureRef = useRef<LiveCapture | null>(null)
  const voiceSessionRef = useRef<LiveSession | null>(null)

  const nextId = () => ++nextRequestId.current

  function getWorker(): Worker {
    let worker = workerRef.current
    if (!worker) {
      worker = new Worker(new URL('./utils/asr.worker.ts', import.meta.url), { type: 'module' })
      worker.onmessage = (event: MessageEvent<AsrResponse>) => {
        handleWorkerMessage(event.data)
      }
      worker.onerror = (event) => {
        dispatch({
          type: 'WORKER_ERROR',
          message: event.message || 'Transcription worker crashed',
        })
      }
      workerRef.current = worker
    }
    return worker
  }

  function finalizeLive() {
    if (!stoppingLive.current || pendingUtterances.current > 0) return
    liveActive.current = false
    const segments = liveSegmentsRef.current
    if (segments.length === 0) {
      dispatch({ type: 'RESET' })
      return
    }
    const text = segments.map((segment) => segment.text).join(' ').trim()
    const duration = segments[segments.length - 1]?.end ?? 0
    const transcript = analyzeTranscript({ text, segments, duration })
    dispatch({ type: 'LIVE_TRANSCRIPT', transcript })
  }

  function handleWorkerMessage(message: AsrResponse): void {
    switch (message.kind) {
      case 'READY':
        dispatch({ type: 'WORKER_READY', device: message.device })
        break

      case 'PROGRESS':
        dispatch({ type: 'WORKER_PROGRESS', progress: message.progress })
        break

      case 'PARTIAL':
        if (message.requestId === fileRequestId.current) {
          dispatch({ type: 'WORKER_PARTIAL', text: message.text })
        } else if (liveActive.current) {
          dispatch({ type: 'LIVE_PARTIAL', text: message.text })
        }
        break

      case 'FINAL':
        if (message.requestId === fileRequestId.current) {
          dispatch({ type: 'WORKER_FINAL', transcript: analyzeTranscript(message.transcript) })
        } else if (liveActive.current) {
          pendingUtterances.current = Math.max(0, pendingUtterances.current - 1)
          for (const segment of message.transcript.segments) {
            const withId: Segment = { ...segment, id: liveSegmentsRef.current.length }
            liveSegmentsRef.current.push(withId)
            dispatch({ type: 'LIVE_SEGMENT', segment: withId })
          }
          finalizeLive()
        }
        break

      case 'ERROR': {
        if (message.code === 'CANCELLED') break
        const isFile = message.requestId === fileRequestId.current
        if (isFile) {
          dispatch({ type: 'WORKER_ERROR', message: message.message })
        } else if (message.requestId === -1 || liveActive.current) {
          if (message.requestId !== -1) {
            pendingUtterances.current = Math.max(0, pendingUtterances.current - 1)
          }
          if (message.code === 'MODEL_LOAD_FAILED' && !stoppingLive.current) {
            liveActive.current = false
            dispatch({ type: 'LIVE_ERROR', message: message.message })
          } else {
            finalizeLive()
          }
        }
        break
      }
    }
  }

  useEffect(() => {
    let cancelled = false
    void detectDevice().then((device) => {
      if (!cancelled) dispatch({ type: 'DEVICE_DETECTED', device })
    })
    return () => {
      cancelled = true
      const worker = workerRef.current
      if (worker) {
        worker.postMessage({ kind: 'DISPOSE' } satisfies AsrRequest)
        worker.terminate()
        workerRef.current = null
      }
      captureRef.current?.stop()
      captureRef.current = null
      voiceSessionRef.current?.close()
      voiceSessionRef.current = null
    }
  }, [])

  const audioUrl = useMemo(() => {
    if (!state.source) return ''
    const blob = state.source.kind === 'file' ? state.source.file : state.source.blob
    return URL.createObjectURL(blob)
  }, [state.source])

  useEffect(() => {
    if (!audioUrl) return
    return () => URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  function handleSelectFile(file: File) {
    dispatch({ type: 'SELECT_SOURCE', source: { kind: 'file', file, name: file.name } })
  }

  function handleSelectRecording(blob: Blob) {
    const stamp = formatShortTimestamp(Math.floor(Date.now() / 1000) % 3600)
    dispatch({
      type: 'SELECT_SOURCE',
      source: { kind: 'recording', blob, name: `Recording ${stamp}.webm` },
    })
  }

  async function handleSubmitFile() {
    const source = stateRef.current.source
    if (!source) return
    const requestId = nextId()
    fileRequestId.current = requestId
    dispatch({ type: 'SUBMIT_FILE' })
    const blob = source.kind === 'file' ? source.file : source.blob
    try {
      const { samples } = await decodeAudio(blob)
      if (fileRequestId.current !== requestId) return
      dispatch({ type: 'TRANSCRIBE_STARTED' })
      getWorker().postMessage({
        kind: 'TRANSCRIBE_FILE',
        audio: samples,
        requestId,
        modelKey: stateRef.current.modelKey,
        device: stateRef.current.device ?? 'wasm',
      } satisfies AsrRequest, [samples.buffer])
    } catch (error) {
      dispatch({
        type: 'WORKER_ERROR',
        message:
          error instanceof Error ? error.message : 'Could not decode this audio file',
      })
    }
  }

  function handleReset() {
    getWorker().postMessage({
      kind: 'CANCEL',
      requestId: fileRequestId.current,
    } satisfies AsrRequest)
    captureRef.current?.stop()
    captureRef.current = null
    liveActive.current = false
    stoppingLive.current = false
    pendingUtterances.current = 0
    liveSegmentsRef.current = []
    voiceSessionRef.current?.close()
    voiceSessionRef.current = null
    dispatch({ type: 'RESET' })
  }

  async function handleStartLive() {
    dispatch({ type: 'START_LIVE' })
    liveActive.current = true
    stoppingLive.current = false
    pendingUtterances.current = 0
    liveSegmentsRef.current = []

    // Preload the model through the worker queue while the VAD warms up.
    getWorker().postMessage({
      kind: 'INIT',
      modelKey: stateRef.current.modelKey,
      device: stateRef.current.device ?? 'wasm',
    } satisfies AsrRequest)

    try {
      captureRef.current = await startLiveCapture({
        onSpeechStart: () => {},
        onUtterance: (audio, startTimeSeconds) => {
          if (!liveActive.current || stoppingLive.current) return
          pendingUtterances.current += 1
          getWorker().postMessage({
            kind: 'TRANSCRIBE_UTTERANCE',
            audio,
            offsetSeconds: startTimeSeconds,
            requestId: nextId(),
            modelKey: stateRef.current.modelKey,
            device: stateRef.current.device ?? 'wasm',
          } satisfies AsrRequest)
        },
        onError: (message) => {
          liveActive.current = false
          dispatch({ type: 'LIVE_ERROR', message })
        },
      })
      if (liveActive.current) dispatch({ type: 'LIVE_READY' })
    } catch {
      // onError already dispatched LIVE_ERROR
    }
  }

  function handleStopLive() {
    dispatch({ type: 'STOP_LIVE' })
    stoppingLive.current = true
    captureRef.current?.stop()
    captureRef.current = null
    finalizeLive()
  }

  function handleStartVoice() {
    const transcript = stateRef.current.transcript
    const context = transcript
      ? transcript.segments
          .map((segment) => `[${formatShortTimestamp(segment.start)}] ${segment.text}`)
          .join('\n')
      : ''
    dispatch({ type: 'VOICE_CONNECTING' })
    void startLiveSession(context, {
      onConnected: () => dispatch({ type: 'VOICE_ACTIVE' }),
      onCaption: (caption) => dispatch({ type: 'VOICE_CAPTION', caption }),
      onEnded: () => dispatch({ type: 'VOICE_END' }),
      onError: (message) => dispatch({ type: 'VOICE_ERROR', message }),
    })
      .then((session) => {
        if (stateRef.current.phase.status !== 'voiceConnecting') {
          session.close()
          return
        }
        voiceSessionRef.current = session
      })
      .catch((error: unknown) => {
        dispatch({
          type: 'VOICE_ERROR',
          message: error instanceof Error ? error.message : String(error),
        })
      })
  }

  function handleStopVoice() {
    voiceSessionRef.current?.close()
    voiceSessionRef.current = null
  }

  function renderBody() {
    switch (state.phase.status) {
      case 'idle':
        return (
          <HomePage
            onSelectFile={handleSelectFile}
            onSelectRecording={handleSelectRecording}
            onStartLive={() => void handleStartLive()}
          />
        )

      case 'ready':
        return state.source ? (
          <FileDisplay
            source={state.source}
            modelKey={state.modelKey}
            device={state.device}
            onModelChange={(modelKey) => dispatch({ type: 'SET_MODEL', modelKey })}
            onSubmit={() => void handleSubmitFile()}
            onReset={handleReset}
          />
        ) : (
          <HomePage
            onSelectFile={handleSelectFile}
            onSelectRecording={handleSelectRecording}
            onStartLive={() => void handleStartLive()}
          />
        )

      case 'decoding':
        return <Transcribing label="Decoding audio" progress={null} />

      case 'loadingModel':
      case 'transcribing':
        return (
          <Transcribing
            label="Transcribing"
            progress={state.progress}
            partialText={state.partialText}
          />
        )

      case 'done':
        return state.transcript ? (
          <Information
            transcript={state.transcript}
            audioUrl={audioUrl}
            onVoiceStart={handleStartVoice}
          />
        ) : null

      case 'liveLoading':
        return (
          <Transcribing
            label="Starting live mode"
            progress={state.progress}
          />
        )

      case 'listening':
      case 'liveProcessing':
        return (
          <LiveView
            segments={state.liveSegments}
            partialText={state.partialText}
            processing={state.phase.status === 'liveProcessing'}
            onStop={handleStopLive}
          />
        )

      case 'liveStopping':
        return (
          <LiveView
            segments={state.liveSegments}
            partialText=""
            processing
            stopping
            onStop={handleStopLive}
          />
        )

      case 'voiceConnecting':
      case 'voiceActive':
        return (
          <VoiceView
            status={state.voiceStatus}
            captions={state.voiceCaptions}
            onStop={handleStopVoice}
          />
        )

      case 'error':
        return (
          <main className="flex-1 p-4 flex flex-col items-center justify-center gap-4 text-center pb-20">
            <h1 className="font-semibold text-4xl">
              <span className="text-rose-400 bold">Something broke</span>
            </h1>
            <p className="text-slate-500 max-w-prose">{state.error}</p>
            <button
              type="button"
              onClick={handleReset}
              className="specialBtn px-4 py-2 rounded-lg text-blue-400"
            >
              Start over
            </button>
          </main>
        )
    }
  }

  const inProgress =
    state.phase.status === 'decoding' ||
    state.phase.status === 'loadingModel' ||
    state.phase.status === 'transcribing' ||
    state.phase.status === 'liveLoading' ||
    state.phase.status === 'liveProcessing' ||
    state.phase.status === 'liveStopping' ||
    state.phase.status === 'voiceConnecting'

  return (
    <div className="min-h-screen flex flex-col bg-[#07111f] text-slate-100">
      <Header
        device={state.device}
        canStartNew={state.phase.status !== 'idle'}
        onNew={handleReset}
      />
      {renderBody()}
      <footer className="border-t border-white/[0.06] p-4 text-center text-xs text-slate-500">
        XScribe — {inProgress ? 'working…' : 'speech to text, chat and voice entirely in your browser'}
      </footer>
    </div>
  )
}
