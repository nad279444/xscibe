import type { Transcript } from '../../app/types'

function pad(value: number, length = 2): string {
  return String(Math.floor(value)).padStart(length, '0')
}

export function formatTimestamp(seconds: number, decimalSeparator: '.' | ',' = '.'): string {
  const safe = Math.max(0, seconds)
  const hours = Math.floor(safe / 3600)
  const minutes = Math.floor((safe % 3600) / 60)
  const secs = Math.floor(safe % 60)
  const millis = Math.round((safe - Math.floor(safe)) * 1000)
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)}${decimalSeparator}${pad(millis, 3)}`
}

export function toSrt(transcript: Transcript): string {
  return transcript.segments
    .map(
      (segment, index) =>
        `${index + 1}\n${formatTimestamp(segment.start, ',')} --> ${formatTimestamp(segment.end, ',')}\n${segment.text}`,
    )
    .join('\n\n')
    .concat('\n')
}

export function toVtt(transcript: Transcript): string {
  const body = transcript.segments
    .map(
      (segment) =>
        `${formatTimestamp(segment.start, '.')} --> ${formatTimestamp(segment.end, '.')}\n${segment.text}`,
    )
    .join('\n\n')
  return `WEBVTT\n\n${body}\n`
}

export function toJson(transcript: Transcript): string {
  return JSON.stringify(
    {
      text: transcript.text,
      duration: transcript.duration,
      segments: transcript.segments.map((segment) => ({
        start: segment.start,
        end: segment.end,
        text: segment.text,
        confidence: segment.confidence,
        flagged: segment.flagged,
        words: segment.words.map((word) => ({
          text: word.text,
          start: word.start,
          end: word.end,
        })),
      })),
    },
    null,
    2,
  )
}

export function toTxt(transcript: Transcript): string {
  return transcript.segments.map((segment) => segment.text).join('\n') + '\n'
}

export type ExportFormat = 'txt' | 'srt' | 'vtt' | 'json'

export const EXPORT_FORMATS: { format: ExportFormat; label: string; mime: string }[] = [
  { format: 'txt', label: 'TXT', mime: 'text/plain' },
  { format: 'srt', label: 'SRT', mime: 'application/x-subrip' },
  { format: 'vtt', label: 'VTT', mime: 'text/vtt' },
  { format: 'json', label: 'JSON', mime: 'application/json' },
]

export function renderExport(transcript: Transcript, format: ExportFormat): string {
  switch (format) {
    case 'srt':
      return toSrt(transcript)
    case 'vtt':
      return toVtt(transcript)
    case 'json':
      return toJson(transcript)
    case 'txt':
      return toTxt(transcript)
  }
}
