import type { Segment, Word } from '@/app/types'

interface ServerEvent {
  type?: string
  delta?: string
  transcript?: string
  error?: { message?: string }
  item_id?: string
}

interface TranscriptHandlers {
  onPartial: (text: string) => void
  onSegment: (segment: Segment) => void
  onError: (message: string) => void
}

export interface RealtimeTranscription {
  stop: () => Promise<void>
  close: () => void
}

const VAD_THRESHOLD = 0.012
const SILENCE_MS = 650
const MIN_SPEECH_MS = 250
const PRE_ROLL_MS = 250

function pcmBase64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.length * 2)
  const view = new DataView(bytes.buffer)
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index] ?? 0))
    view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  let binary = ''
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, Math.min(index + 0x8000, bytes.length)))
  }
  return btoa(binary)
}

function resample(input: Float32Array, sourceRate: number): Float32Array {
  if (sourceRate === 24000) return input
  const outputLength = Math.round((input.length * 24000) / sourceRate)
  const output = new Float32Array(outputLength)
  const ratio = sourceRate / 24000
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio
    const left = Math.floor(position)
    const fraction = position - left
    const first = input[Math.min(left, input.length - 1)] ?? 0
    const second = input[Math.min(left + 1, input.length - 1)] ?? first
    output[index] = first + (second - first) * fraction
  }
  return output
}

function makeWords(text: string, start: number, end: number): Word[] {
  const tokens = text.trim().split(/\s+/).filter(Boolean)
  const span = Math.max(0, end - start)
  return tokens.map((token, index) => ({
    text: token,
    start: start + (span * index) / Math.max(tokens.length, 1),
    end: start + (span * (index + 1)) / Math.max(tokens.length, 1),
    approximate: true,
  }))
}

export async function openRealtimeTranscription(
  handlers: TranscriptHandlers,
): Promise<RealtimeTranscription> {
  const response = await fetch('/api/live/session', { method: 'POST' })
  const payload = (await response.json().catch(() => ({}))) as { token?: string; error?: string }
  if (!response.ok || !payload.token) {
    throw new Error(payload.error ?? 'Could not create a live transcription session')
  }

  const socket = new WebSocket('wss://api.openai.com/v1/realtime?intent=transcription', [
    'realtime',
    `openai-insecure-api-key.${payload.token}`,
  ])
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error('Timed out connecting to OpenAI')), 12000)
    const handleMessage = (message: MessageEvent) => {
      let event: ServerEvent
      try {
        event = JSON.parse(String(message.data)) as ServerEvent
      } catch {
        return
      }
      if (event.type === 'session.updated') {
        window.clearTimeout(timeout)
        socket.removeEventListener('message', handleMessage)
        resolve()
      } else if (event.type === 'error') {
        window.clearTimeout(timeout)
        socket.removeEventListener('message', handleMessage)
        reject(new Error(event.error?.message ?? 'Could not configure OpenAI transcription'))
      }
    }
    socket.addEventListener('open', () => {
      socket.addEventListener('message', handleMessage)
      socket.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'transcription',
          audio: {
            input: {
              format: { type: 'audio/pcm', rate: 24000 },
              transcription: { model: 'gpt-live-transcribe', delay: 'low' },
              turn_detection: null,
            },
          },
        },
      }))
    }, { once: true })
    socket.addEventListener('error', () => {
      window.clearTimeout(timeout)
      reject(new Error('Could not connect to OpenAI Realtime transcription'))
    }, { once: true })
  }).catch((error: unknown) => {
    socket.close()
    throw error
  })

  let context: AudioContext | null = null
  let stream: MediaStream | null = null
  let source: MediaStreamAudioSourceNode | null = null
  let worklet: AudioWorkletNode | null = null
  let silentGain: GainNode | null = null
  let stopped = false
  let speechStartedAt = 0
  let lastSpeechAt = 0
  let speechSamples = 0
  let partial = ''
  let segmentId = 0
  let lastSegmentEnd = 0
  let pendingFinals = 0
  const turnStarts: number[] = []
  const turnStartByItem = new Map<string, number>()
  const finalWaiters: Array<() => void> = []
  const startedAt = performance.now()
  const preRoll: Float32Array[] = []
  let preRollLimit = 1

  const send = (event: object) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(event))
  }

  socket.addEventListener('message', (message) => {
    let event: ServerEvent
    try {
      event = JSON.parse(String(message.data)) as ServerEvent
    } catch {
      return
    }
    if (event.type === 'input_audio_buffer.committed' && event.item_id) {
      turnStartByItem.set(event.item_id, turnStarts.shift() ?? lastSegmentEnd)
    } else if (event.type === 'conversation.item.input_audio_transcription.delta') {
      partial += event.delta ?? ''
      handlers.onPartial(partial)
    } else if (event.type === 'conversation.item.input_audio_transcription.completed') {
      const text = (event.transcript ?? partial).trim()
      const end = Math.max(lastSegmentEnd, (performance.now() - startedAt) / 1000)
      if (text) {
        const start = Math.min(end, Math.max(lastSegmentEnd, turnStartByItem.get(event.item_id ?? '') ?? lastSegmentEnd))
        handlers.onSegment({
          id: segmentId++, text, start, end,
          words: makeWords(text, start, end),
          flagged: false, flagReasons: [], confidence: 1,
        })
        lastSegmentEnd = end
      }
      partial = ''
      handlers.onPartial('')
      if (event.item_id) turnStartByItem.delete(event.item_id)
      pendingFinals = Math.max(0, pendingFinals - 1)
      if (pendingFinals === 0) finalWaiters.splice(0).forEach((resolve) => resolve())
    } else if (event.type === 'error') {
      handlers.onError(event.error?.message ?? 'OpenAI Realtime transcription failed')
    }
  })

  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
    context = new AudioContext()
    preRollLimit = Math.ceil((PRE_ROLL_MS / 1000) * context.sampleRate / 2048) + 1
    await context.audioWorklet.addModule('/live-audio-processor.js')
    source = context.createMediaStreamSource(stream)
    worklet = new AudioWorkletNode(context, 'live-audio-processor')
    silentGain = context.createGain()
    silentGain.gain.value = 0
    source.connect(worklet)
    worklet.connect(silentGain)
    silentGain.connect(context.destination)
    worklet.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
      if (stopped || socket.readyState !== WebSocket.OPEN) return
      const samples = resample(data, context?.sampleRate ?? 48000)
      let energy = 0
      for (const value of samples) energy += value * value
      const rms = Math.sqrt(energy / Math.max(samples.length, 1))
      const now = performance.now()
      const elapsed = (now - startedAt) / 1000

      if (rms >= VAD_THRESHOLD) {
        if (speechStartedAt === 0) {
          const preRollDuration = preRoll.reduce((duration, frame) => duration + frame.length / 24000, 0)
          speechStartedAt = Math.max(0, elapsed - preRollDuration)
          speechSamples = 0
          for (const prior of preRoll) {
            send({ type: 'input_audio_buffer.append', audio: pcmBase64(prior) })
            speechSamples += prior.length
          }
        }
        lastSpeechAt = now
        send({ type: 'input_audio_buffer.append', audio: pcmBase64(samples) })
        speechSamples += samples.length
      } else if (speechStartedAt !== 0) {
        send({ type: 'input_audio_buffer.append', audio: pcmBase64(samples) })
        speechSamples += samples.length
        if (now - lastSpeechAt >= SILENCE_MS) {
          if (speechSamples >= (24000 * MIN_SPEECH_MS) / 1000) {
            turnStarts.push(speechStartedAt)
            pendingFinals += 1
            send({ type: 'input_audio_buffer.commit' })
            partial = ''
          } else {
            send({ type: 'input_audio_buffer.clear' })
          }
          speechStartedAt = 0
          speechSamples = 0
          preRoll.length = 0
          handlers.onPartial('')
        }
      } else {
        preRoll.push(samples)
        while (preRoll.length > preRollLimit) preRoll.shift()
      }
    }
  } catch (error) {
    socket.close()
    stream?.getTracks().forEach((track) => track.stop())
    await context?.close().catch(() => {})
    throw error instanceof Error ? error : new Error(String(error))
  }

  const cleanup = () => {
    if (stopped) return
    stopped = true
    worklet?.port.close()
    source?.disconnect()
    worklet?.disconnect()
    silentGain?.disconnect()
    stream?.getTracks().forEach((track) => track.stop())
    void context?.close()
  }

  return {
    stop: async () => {
      if (stopped) return
      cleanup()
      if (speechStartedAt !== 0 && speechSamples >= (24000 * MIN_SPEECH_MS) / 1000) {
        turnStarts.push(speechStartedAt)
        pendingFinals += 1
        send({ type: 'input_audio_buffer.commit' })
      }
      if (pendingFinals > 0) {
        await Promise.race([
          new Promise<void>((resolve) => finalWaiters.push(resolve)),
          new Promise<void>((resolve) => window.setTimeout(resolve, 2500)),
        ])
      }
      socket.close()
    },
    close: () => {
      cleanup()
      socket.close()
    },
  }
}
