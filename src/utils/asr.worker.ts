import {
  env,
  pipeline,
  WhisperTextStreamer,
  InterruptableStoppingCriteria,
} from '@huggingface/transformers'
import type { AsrRequest, AsrResponse } from '../protocol/messages'
import type { Device, ModelKey, ModelProgress, Word } from '../app/types'
import { MODELS, MODEL_DTYPE } from '../lib/asr/registry'
import { groupWordsIntoSegments } from '../lib/audio/decode'

env.allowLocalModels = false

interface RawWordChunk {
  text: string
  timestamp: [number | null, number | null]
}

interface RawAsrOutput {
  text?: string
  chunks?: RawWordChunk[]
}

interface AsrPipeline {
  tokenizer: unknown
  (audio: Float32Array, options: Record<string, unknown>): Promise<RawAsrOutput>
}

const scope = self as unknown as {
  postMessage(message: AsrResponse): void
  addEventListener(
    type: 'message',
    listener: (event: MessageEvent<AsrRequest>) => void,
  ): void
}

let transcriber: AsrPipeline | null = null
let loadedKey: ModelKey | null = null
let loadedDevice: Device | null = null
let queue: Promise<void> = Promise.resolve()
const jobs = new Map<number, InterruptableStoppingCriteria>()

function post(message: AsrResponse): void {
  scope.postMessage(message)
}

interface ProgressEvent {
  status?: string
  file?: string
  progress?: number
  loaded?: number
  total?: number
}

async function ensurePipeline(
  modelKey: ModelKey,
  device: Device,
  requestId: number,
): Promise<{ instance: AsrPipeline; device: Device }> {
  if (transcriber && loadedKey === modelKey && loadedDevice === device) {
    return { instance: transcriber, device }
  }

  const info = MODELS[modelKey]
  const onProgress = (data: ProgressEvent) => {
    if (data.status === 'progress') {
      const progress: ModelProgress = {
        phase: 'download',
        ...(data.file !== undefined && { file: data.file }),
        ...(data.progress !== undefined && { progress: data.progress }),
        ...(data.loaded !== undefined && { loaded: data.loaded }),
        ...(data.total !== undefined && { total: data.total }),
      }
      post({ kind: 'PROGRESS', requestId, progress })
    } else if (data.status === 'ready' || data.status === 'done') {
      post({ kind: 'PROGRESS', requestId, progress: { phase: 'load' } })
    }
  }

  post({ kind: 'PROGRESS', requestId, progress: { phase: 'load' } })

  const options = {
    dtype: MODEL_DTYPE,
    progress_callback: onProgress,
  }

  let actualDevice: Device = device
  try {
    const instance = (await pipeline('automatic-speech-recognition', info.repo, {
      ...options,
      device,
    })) as unknown as AsrPipeline
    transcriber = instance
    loadedKey = modelKey
    loadedDevice = device
    return { instance, device }
  } catch (primaryError) {
    if (device !== 'webgpu') throw primaryError
    actualDevice = 'wasm'
    const instance = (await pipeline('automatic-speech-recognition', info.repo, {
      ...options,
      device: 'wasm',
    })) as unknown as AsrPipeline
    transcriber = instance
    loadedKey = modelKey
    loadedDevice = 'wasm'
    return { instance, device: actualDevice }
  }
}

function buildWords(chunks: RawWordChunk[], offsetSeconds: number): Word[] {
  const words: Word[] = []
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]
    if (!chunk) continue
    const rawStart = chunk.timestamp[0]
    const rawEnd = chunk.timestamp[1]
    const prevEnd = words.length > 0 ? words[words.length - 1]!.end : null
    const start = rawStart ?? prevEnd ?? 0
    const nextStart =
      i + 1 < chunks.length ? chunks[i + 1]?.timestamp[0] ?? null : null
    const end = rawEnd ?? nextStart ?? start + Math.max(0.2, chunk.text.trim().length * 0.06)
    words.push({
      text: chunk.text,
      start: Math.max(0, start + offsetSeconds),
      end: Math.max(0, end + offsetSeconds),
    })
  }
  return words
}

async function runTranscription(
  audio: Float32Array,
  requestId: number,
  offsetSeconds: number,
  modelKey: ModelKey,
  requestedDevice: Device,
): Promise<void> {
  const job = new InterruptableStoppingCriteria()
  jobs.set(requestId, job)

  try {
    const { instance, device } = await ensurePipeline(modelKey, requestedDevice, requestId)
    if (job.interrupted) {
      post({ kind: 'ERROR', requestId, code: 'CANCELLED', message: 'Cancelled' })
      return
    }

    post({ kind: 'READY', device, modelKey })

    const completedChunks: string[] = []
    let currentChunk = ''
    const emitPartial = () => {
      const text = [...completedChunks, currentChunk].filter(Boolean).join(' ')
      if (text.trim()) post({ kind: 'PARTIAL', requestId, text: text.trim() })
    }

    const streamer = new WhisperTextStreamer(instance.tokenizer as never, {
      skip_special_tokens: true,
      callback_function: (text: string) => {
        currentChunk = text
        emitPartial()
      },
      on_finalize: () => {
        if (currentChunk.trim()) completedChunks.push(currentChunk)
        currentChunk = ''
      },
    })

    const output = await instance(audio, {
      top_k: 0,
      do_sample: false,
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: 'word',
      streamer,
      stopping_criteria: [job],
    })

    jobs.delete(requestId)

    if (job.interrupted) {
      post({ kind: 'ERROR', requestId, code: 'CANCELLED', message: 'Cancelled' })
      return
    }

    const words = buildWords(output.chunks ?? [], offsetSeconds)
    const grouped = groupWordsIntoSegments(words)
    let id = 0
    const segments = grouped.map((group) => ({
      id: id++,
      text: group.text,
      start: group.start,
      end: group.end,
      words: group.words,
      flagged: false,
      flagReasons: [],
      confidence: 1,
    }))
    const text =
      output.text?.trim() || segments.map((segment) => segment.text).join(' ')
    const duration =
      offsetSeconds + audio.length / 16000

    post({
      kind: 'FINAL',
      requestId,
      transcript: { text, segments, duration },
    })
  } catch (error) {
    jobs.delete(requestId)
    const message = error instanceof Error ? error.message : String(error)
    post({
      kind: 'ERROR',
      requestId,
      code: 'INFERENCE_FAILED',
      message,
    })
  }
}

scope.addEventListener('message', (event) => {
  const message = event.data
  switch (message.kind) {
    case 'INIT': {
      const requestId = -1
      queue = queue.then(() =>
        ensurePipeline(message.modelKey, message.device, requestId)
          .then(({ device }) => {
            post({ kind: 'READY', device, modelKey: message.modelKey })
          })
          .catch((error: unknown) => {
            post({
              kind: 'ERROR',
              requestId,
              code: 'MODEL_LOAD_FAILED',
              message: error instanceof Error ? error.message : String(error),
            })
          }),
      )
      break
    }
    case 'TRANSCRIBE_FILE': {
      const { audio, requestId, modelKey, device } = message
      queue = queue.then(() =>
        runTranscription(audio, requestId, 0, modelKey, device),
      )
      break
    }
    case 'TRANSCRIBE_UTTERANCE': {
      const { audio, offsetSeconds, requestId, modelKey, device } = message
      queue = queue.then(() =>
        runTranscription(audio, requestId, offsetSeconds, modelKey, device),
      )
      break
    }
    case 'CANCEL': {
      jobs.get(message.requestId)?.interrupt()
      break
    }
    case 'DISPOSE': {
      transcriber = null
      loadedKey = null
      loadedDevice = null
      break
    }
  }
})
