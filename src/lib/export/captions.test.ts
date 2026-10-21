import { describe, expect, it } from 'vitest'
import {
  EXPORT_FORMATS,
  formatTimestamp,
  renderExport,
  toJson,
  toSrt,
  toTxt,
  toVtt,
} from './captions'
import type { Transcript } from '../../app/types'

function transcript(): Transcript {
  return {
    text: 'Hello world. Goodbye moon.',
    duration: 7.5,
    segments: [
      {
        id: 0,
        text: 'Hello world.',
        start: 0,
        end: 2.5,
        words: [
          { text: 'Hello', start: 0, end: 1.2 },
          { text: 'world.', start: 1.2, end: 2.5 },
        ],
        flagged: false,
        flagReasons: [],
        confidence: 1,
      },
      {
        id: 1,
        text: 'Goodbye moon.',
        start: 3,
        end: 7.5,
        words: [
          { text: 'Goodbye', start: 3, end: 5 },
          { text: 'moon.', start: 5, end: 7.5 },
        ],
        flagged: false,
        flagReasons: [],
        confidence: 0.9,
      },
    ],
  }
}

describe('formatTimestamp', () => {
  it('formats hours, minutes, seconds and milliseconds', () => {
    expect(formatTimestamp(3661.5)).toBe('01:01:01.500')
  })

  it('supports comma separator for SRT', () => {
    expect(formatTimestamp(2.5, ',')).toBe('00:00:02,500')
  })

  it('clamps negative values to zero', () => {
    expect(formatTimestamp(-3)).toBe('00:00:00.000')
  })
})

describe('toSrt', () => {
  it('numbers blocks and uses comma milliseconds', () => {
    const srt = toSrt(transcript())
    expect(srt).toContain('1\n00:00:00,000 --> 00:00:02,500\nHello world.')
    expect(srt).toContain('2\n00:00:03,000 --> 00:00:07,500\nGoodbye moon.')
  })
})

describe('toVtt', () => {
  it('starts with WEBVTT and uses dot milliseconds', () => {
    const vtt = toVtt(transcript())
    expect(vtt.startsWith('WEBVTT')).toBe(true)
    expect(vtt).toContain('00:00:00.000 --> 00:00:02.500')
  })
})

describe('toJson', () => {
  it('round-trips through JSON.parse with word timestamps', () => {
    const parsed = JSON.parse(toJson(transcript())) as {
      duration: number
      segments: { words: { text: string }[] }[]
    }
    expect(parsed.duration).toBe(7.5)
    expect(parsed.segments).toHaveLength(2)
    expect(parsed.segments[0]?.words[0]?.text).toBe('Hello')
  })
})

describe('toTxt', () => {
  it('joins segment texts into lines', () => {
    expect(toTxt(transcript())).toBe('Hello world.\nGoodbye moon.\n')
  })
})

describe('renderExport', () => {
  it('dispatches every registered format without throwing', () => {
    for (const { format } of EXPORT_FORMATS) {
      expect(renderExport(transcript(), format)).toBeTruthy()
    }
  })
})
