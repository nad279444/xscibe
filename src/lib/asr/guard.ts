import type { GuardFlag, Segment, Transcript, Word } from '../../app/types'

const KNOWN_HALLUCINATIONS: RegExp[] = [
  /thank you for watching/i,
  /please subscribe/i,
  /like and subscribe/i,
  /don't forget to/i,
  /\bvisit (our|us)\b.*\bwebsite\b/i,
  /for more (videos|information)/i,
  /^\s*you\s*[.!?]*\s*$/i,
  /^\s*so\s*[.!?]*\s*$/i,
  /^\s*bye\s*[.!?]*\s*$/i,
  /♪/,
  /\bnull\b/i,
  /\ballah\b.*\ballah\b/i,
]

const MAX_WORDS_PER_SECOND = 4.5
const MIN_WORDS_PER_SECOND = 0.4
const SILENT_LONG_SEGMENT_SECONDS = 8

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, '').replace(/\s+/g, ' ').trim()
}

function repeatedNgramRatio(words: Word[]): number {
  const tokens = words.map((w) => normalize(w.text)).filter(Boolean)
  if (tokens.length < 6) return 0
  const seen = new Map<string, number>()
  let repeated = 0
  const n = 3
  for (let i = 0; i + n <= tokens.length; i++) {
    const gram = tokens.slice(i, i + n).join(' ')
    const count = seen.get(gram) ?? 0
    if (count > 0) repeated += n
    seen.set(gram, count + 1)
  }
  return tokens.length > 0 ? Math.min(1, repeated / tokens.length) : 0
}

function durationOutlier(segment: Segment): boolean {
  const duration = segment.end - segment.start
  if (duration <= 0) return false
  const wordCount = segment.words.filter((w) => w.text.trim()).length
  const wordsPerSecond = wordCount / duration
  if (wordCount > 3 && wordsPerSecond > MAX_WORDS_PER_SECOND) return true
  if (duration >= SILENT_LONG_SEGMENT_SECONDS && wordCount <= 2) return true
  if (wordCount > 3 && wordsPerSecond < MIN_WORDS_PER_SECOND) return true
  return false
}

export function analyzeSegment(segment: Segment, previous: Segment | null): Segment {
  const reasons = new Set<GuardFlag>()
  let confidence = 1

  const text = segment.text.trim()
  if (!text || segment.words.length === 0) {
    reasons.add('empty')
    confidence = 0
  }

  if (KNOWN_HALLUCINATIONS.some((pattern) => pattern.test(text))) {
    reasons.add('known-hallucination')
    confidence *= 0.15
  }

  const repetition = repeatedNgramRatio(segment.words)
  if (repetition > 0.4) {
    reasons.add('repetition')
    confidence *= 1 - repetition
  }

  if (previous && normalize(previous.text) === normalize(text)) {
    reasons.add('repetition')
    confidence = Math.min(confidence, 0.2)
  }

  if (durationOutlier(segment)) {
    reasons.add('duration-outlier')
    confidence *= 0.5
  }

  if (confidence < 0.5 && !reasons.has('empty')) {
    reasons.add('low-confidence')
  }

  return {
    ...segment,
    flagged: reasons.size > 0,
    flagReasons: [...reasons],
    confidence: Math.round(Math.max(0, Math.min(1, confidence)) * 100) / 100,
  }
}

export function analyzeTranscript(transcript: Transcript): Transcript {
  let previous: Segment | null = null
  const segments = transcript.segments.map((segment) => {
    const analyzed = analyzeSegment(segment, previous)
    previous = analyzed
    return analyzed
  })
  return { ...transcript, segments }
}

export function stripFlagged(transcript: Transcript): Transcript {
  const segments = transcript.segments.filter((segment) => !segment.flagged)
  return {
    ...transcript,
    segments,
    text: segments.map((segment) => segment.text).join(' '),
  }
}
