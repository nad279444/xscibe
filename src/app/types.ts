export interface Word {
  text: string
  start: number
  end: number
  confidence?: number
}

export interface Segment {
  id: number
  text: string
  start: number
  end: number
  words: Word[]
  flagged: boolean
  flagReasons: GuardFlag[]
  confidence: number
}

export interface Transcript {
  text: string
  segments: Segment[]
  duration: number
}

export type GuardFlag =
  | 'repetition'
  | 'duration-outlier'
  | 'known-hallucination'
  | 'empty'
  | 'low-confidence'

export type Device = 'webgpu' | 'wasm'

export type ModelKey = 'tiny' | 'base' | 'small'

export interface ModelInfo {
  key: ModelKey
  label: string
  repo: string
  params: string
  realtimeCapable: boolean
}

export interface ModelProgress {
  phase: 'download' | 'load' | 'warmup'
  file?: string
  progress?: number
  loaded?: number
  total?: number
}

export type AudioSource =
  | { kind: 'file'; file: File; name: string }
  | { kind: 'recording'; blob: Blob; name: string }

export interface VoiceCaption {
  role: 'user' | 'assistant'
  text: string
  final: boolean
}

export type VoiceStatus = 'idle' | 'connecting' | 'active' | 'ended' | 'error'

export interface ChatTurn {
  id: number
  role: 'user' | 'assistant'
  text: string
}
