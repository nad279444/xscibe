'use client'

import { useEffect, useMemo, useReducer, useRef } from 'react'
import { initialState, reducer } from './app/machine'
import type { Segment } from './app/types'
import { analyzeTranscript } from './lib/asr/guard'
import { startLiveCapture, type LiveCapture } from './lib/audio/liveCapture'
import { transcribeAudio } from './lib/openai/transcribe'
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

  const nextRequestId = useRef(0)
  const fileRequestId = useRef(-1)
  const fileAbortController = useRef<AbortController | null>(null)
  const liveActive = useRef(false)
  const stoppingLive = useRef(false)
  const liveGeneration = useRef(0)
  const liveSegmentsRef = useRef<Segment[]>([])
  const captureRef = useRef<LiveCapture | null>(null)
  const voiceSessionRef = useRef<LiveSession | null>(null)

  const nextId = () => ++nextRequestId.current

  function finalizeLive() {
    if (!stoppingLive.current) return
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

  useEffect(() => {
    return () => {
      fileAbortController.current?.abort()
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

  async function handleSubmitFile(durationSeconds?: number) {
    const source = stateRef.current.source
    if (!source) return
    const requestId = nextId()
    fileAbortController.current?.abort()
    const controller = new AbortController()
    fileAbortController.current = controller
    fileRequestId.current = requestId
    dispatch({ type: 'SUBMIT_FILE' })
    const blob = source.kind === 'file' ? source.file : source.blob
    try {
      if (fileRequestId.current !== requestId) return
      dispatch({ type: 'TRANSCRIBE_STARTED' })
      const transcript = await transcribeAudio(blob, source.name, controller.signal, durationSeconds)
      if (fileRequestId.current === requestId) {
        dispatch({ type: 'TRANSCRIPTION_FINAL', transcript })
      }
    } catch (error) {
      if (fileRequestId.current === requestId) {
        dispatch({
          type: 'TRANSCRIPTION_ERROR',
          message: error instanceof Error ? error.message : 'Could not transcribe this audio file',
        })
      }
    }
  }

  function handleReset() {
    liveGeneration.current += 1
    fileRequestId.current = -1
    fileAbortController.current?.abort()
    fileAbortController.current = null
    captureRef.current?.stop()
    captureRef.current = null
    liveActive.current = false
    stoppingLive.current = false
    liveSegmentsRef.current = []
    voiceSessionRef.current?.close()
    voiceSessionRef.current = null
    dispatch({ type: 'RESET' })
  }

  async function handleStartLive() {
    liveGeneration.current += 1
    const generation = liveGeneration.current
    dispatch({ type: 'START_LIVE' })
    liveActive.current = true
    stoppingLive.current = false
    liveSegmentsRef.current = []

    try {
      const capture = await startLiveCapture({
        onPartial: (text) => {
          if (generation === liveGeneration.current && liveActive.current) {
            dispatch({ type: 'LIVE_PARTIAL', text })
          }
        },
        onSegment: (segment) => {
          if (generation !== liveGeneration.current) return
          const withId: Segment = { ...segment, id: liveSegmentsRef.current.length }
          liveSegmentsRef.current.push(withId)
          dispatch({ type: 'LIVE_SEGMENT', segment: withId })
        },
        onError: (message) => {
          if (generation !== liveGeneration.current) return
          liveActive.current = false
          captureRef.current?.close()
          captureRef.current = null
          dispatch({ type: 'LIVE_ERROR', message })
        },
      })
      if (generation !== liveGeneration.current) {
        capture.stop()
        return
      }
      captureRef.current = capture
      if (liveActive.current) dispatch({ type: 'LIVE_READY' })
    } catch (error) {
      if (generation === liveGeneration.current && liveActive.current) {
        liveActive.current = false
        dispatch({
          type: 'LIVE_ERROR',
          message: error instanceof Error ? error.message : 'Could not start live audio capture',
        })
      }
    }
  }

  function handleStopLive() {
    dispatch({ type: 'STOP_LIVE' })
    stoppingLive.current = true
    void captureRef.current?.stop().finally(() => finalizeLive())
    captureRef.current = null
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
            onSubmit={(durationSeconds) => void handleSubmitFile(durationSeconds)}
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
        return <Transcribing label="Preparing audio" detail="Preparing your upload for OpenAI" />

      case 'transcribing':
        return (
          <Transcribing
            label="Transcribing with GPT-Transcribe"
            detail="Sending audio securely to OpenAI"
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
            detail="Connecting your microphone to live transcription"
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
    state.phase.status === 'transcribing' ||
    state.phase.status === 'liveLoading' ||
    state.phase.status === 'liveProcessing' ||
    state.phase.status === 'liveStopping' ||
    state.phase.status === 'voiceConnecting'

  return (
    <div className="min-h-screen flex flex-col bg-[#07111f] text-slate-100">
      <Header canStartNew={state.phase.status !== 'idle'} onNew={handleReset} />
      {renderBody()}
      <footer className="border-t border-white/[0.06] p-4 text-center text-xs text-slate-500">
        XScribe — {inProgress ? 'working…' : 'cloud transcription, transcript chat and voice'}
      </footer>
    </div>
  )
}
