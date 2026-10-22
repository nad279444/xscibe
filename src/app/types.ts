export interface Word {
  text: string
  start: number
  end: number
  confidence?: number
  approximate?: boolean
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
