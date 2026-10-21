import { describe, expect, it } from 'vitest'
import { analyzeSegment, analyzeTranscript, stripFlagged } from './guard'
import type { Segment, Transcript, Word } from '../../app/types'

function word(text: string, start: number, end: number): Word {
  return { text, start, end }
}

function segment(
  id: number,
  text: string,
  start: number,
  end: number,
  words?: Word[],
): Segment {
  return {
    id,
    text,
    start,
    end,
    words: words ?? [word(text, start, end)],
    flagged: false,
    flagReasons: [],
    confidence: 1,
  }
}

describe('analyzeSegment', () => {
  it('passes a clean segment with full confidence', () => {
    const result = analyzeSegment(segment(0, 'The meeting starts at noon.', 0, 3), null)
    expect(result.flagged).toBe(false)
    expect(result.flagReasons).toEqual([])
    expect(result.confidence).toBe(1)
  })

  it('flags known hallucinations', () => {
    const result = analyzeSegment(segment(0, 'Thank you for watching!', 0, 2), null)
    expect(result.flagged).toBe(true)
    expect(result.flagReasons).toContain('known-hallucination')
    expect(result.flagReasons).toContain('low-confidence')
    expect(result.confidence).toBeLessThan(0.5)
  })

  it('flags n-gram repetition', () => {
    const tokens = [
      'the', 'cat', 'sat', 'the', 'cat', 'sat', 'the', 'cat', 'sat',
    ]
    const words = tokens.map((token, index) => word(token, index, index + 1))
    const result = analyzeSegment(
      segment(0, tokens.join(' '), 0, tokens.length, words),
      null,
    )
    expect(result.flagReasons).toContain('repetition')
  })

  it('flags implausible speech rate', () => {
    const words = ['one', 'two', 'three', 'four', 'five'].map((token, index) =>
      word(token, index * 0.1, index * 0.1 + 0.1),
    )
    const result = analyzeSegment(
      segment(0, 'one two three four five', 0, 0.5, words),
      null,
    )
    expect(result.flagReasons).toContain('duration-outlier')
  })

  it('flags long nearly-silent segments', () => {
    const result = analyzeSegment(segment(0, 'um yeah', 0, 12), null)
    expect(result.flagReasons).toContain('duration-outlier')
  })

  it('flags empty segments', () => {
    const result = analyzeSegment(segment(0, '', 0, 0, []), null)
    expect(result.flagReasons).toContain('empty')
    expect(result.confidence).toBe(0)
  })

  it('flags a segment that repeats the previous one verbatim', () => {
    const first = segment(0, 'we need to ship tomorrow', 0, 3)
    const second = segment(1, 'We Need To Ship Tomorrow!', 3, 6)
    const result = analyzeSegment(second, first)
    expect(result.flagReasons).toContain('repetition')
    expect(result.confidence).toBeLessThanOrEqual(0.2)
  })
})

describe('analyzeTranscript', () => {
  it('analyzes every segment and keeps transcript metadata', () => {
    const transcript: Transcript = {
      text: 'Hello there. Thank you for watching!',
      duration: 6,
      segments: [
        segment(0, 'Hello there.', 0, 3),
        segment(1, 'Thank you for watching!', 3, 6),
      ],
    }
    const result = analyzeTranscript(transcript)
    expect(result.duration).toBe(6)
    expect(result.segments[0]?.flagged).toBe(false)
    expect(result.segments[1]?.flagged).toBe(true)
  })
})

describe('stripFlagged', () => {
  it('removes flagged segments and rebuilds the text', () => {
    const transcript: Transcript = {
      text: 'Hello there. Thank you for watching!',
      duration: 6,
      segments: [
        segment(0, 'Hello there.', 0, 3),
        segment(1, 'Thank you for watching!', 3, 6),
      ],
    }
    const clean = stripFlagged(analyzeTranscript(transcript))
    expect(clean.segments).toHaveLength(1)
    expect(clean.text).toBe('Hello there.')
  })
})
