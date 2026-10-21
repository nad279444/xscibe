import type { Device, ModelKey, ModelProgress, Transcript } from '../app/types'

export type AsrRequest =
  | { kind: 'INIT'; modelKey: ModelKey; device: Device }
  | {
      kind: 'TRANSCRIBE_FILE'
      audio: Float32Array
      requestId: number
      modelKey: ModelKey
      device: Device
    }
  | {
      kind: 'TRANSCRIBE_UTTERANCE'
      audio: Float32Array
      offsetSeconds: number
      requestId: number
      modelKey: ModelKey
      device: Device
    }
  | { kind: 'CANCEL'; requestId: number }
  | { kind: 'DISPOSE' }

export type AsrResponse =
  | { kind: 'READY'; device: Device; modelKey: ModelKey; fallbackReason?: string }
  | { kind: 'PROGRESS'; requestId: number; progress: ModelProgress }
  | { kind: 'PARTIAL'; requestId: number; text: string }
  | { kind: 'FINAL'; requestId: number; transcript: Transcript }
  | { kind: 'ERROR'; requestId: number; code: AsrErrorCode; message: string }

export type AsrErrorCode =
  | 'MODEL_LOAD_FAILED'
  | 'INFERENCE_FAILED'
  | 'CANCELLED'
  | 'UNSUPPORTED_DEVICE'
