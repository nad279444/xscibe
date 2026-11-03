import type { Transcript } from '@/app/types'
import { validateAudioFile } from '@/lib/audio/uploadLimits'

interface TranscriptionStreamEvent {
  type?: string
  text?: string
  message?: string
  transcript?: Transcript
}

export async function transcribeAudio(
  blob: Blob,
  filename: string,
  signal?: AbortSignal,
  durationSeconds?: number,
  onPartial?: (text: string) => void,
): Promise<Transcript> {
  const problem = validateAudioFile({ name: filename, size: blob.size, type: blob.type })
  if (problem) throw new Error(problem)

  const safeName = filename.replace(/[^\w.-]+/g, '_').slice(-120) || 'audio.webm'
  const contentType = blob.type || 'application/octet-stream'
  const ticketResponse = await fetch('/api/transcribe/upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename: safeName, contentType, size: blob.size }),
    signal,
  })
  const ticket = (await ticketResponse.json().catch(() => ({}))) as {
    error?: string
    uploadUrl?: string
    objectName?: string
  }
  if (!ticketResponse.ok || !ticket.uploadUrl || !ticket.objectName) {
    throw new Error(ticket.error ?? `Could not prepare upload (${ticketResponse.status})`)
  }

  const uploadResponse = await fetch(ticket.uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': contentType },
    body: blob,
    signal,
  })
  if (!uploadResponse.ok) throw new Error(`Audio upload failed (${uploadResponse.status})`)

  const response = await fetch('/api/transcribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ objectName: ticket.objectName, filename: safeName, durationSeconds }),
    signal,
  })

  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { error?: string }
    throw new Error(payload.error ?? `Transcription failed (${response.status})`)
  }

  if (!(response.headers.get('content-type') ?? '').includes('text/event-stream')) {
    const payload = (await response.json().catch(() => ({}))) as {
      error?: string
      transcript?: Transcript
    }
    if (!payload.transcript) throw new Error(payload.error ?? 'Transcription failed')
    return payload.transcript
  }

  if (!response.body) throw new Error('Transcription response has no body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let transcript: Transcript | null = null

  const handleEvent = (event: TranscriptionStreamEvent) => {
    if (event.type === 'partial' && typeof event.text === 'string') {
      onPartial?.(event.text)
    } else if (event.type === 'final' && event.transcript) {
      transcript = event.transcript
    } else if (event.type === 'error') {
      throw new Error(event.message ?? 'Transcription failed')
    }
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let boundary = /\r?\n\r?\n/.exec(buffer)
      while (boundary) {
        const raw = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        const data = raw
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trim())
          .join('\n')
        if (data && data !== '[DONE]') {
          handleEvent(JSON.parse(data) as TranscriptionStreamEvent)
        }
        boundary = /\r?\n\r?\n/.exec(buffer)
      }
    }
  } finally {
    reader.releaseLock()
  }

  if (!transcript) throw new Error('Transcription stream ended without a result')
  return transcript
}

export function encodeWav(samples: Float32Array, sampleRate = 16_000): Blob {
  const bytesPerSample = 2
  const dataSize = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)
  const write = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index))
    }
  }

  write(0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true)
  write(8, 'WAVE')
  write(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * bytesPerSample, true)
  view.setUint16(32, bytesPerSample, true)
  view.setUint16(34, 16, true)
  write(36, 'data')
  view.setUint32(40, dataSize, true)

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index] ?? 0))
    view.setInt16(44 + index * bytesPerSample, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}
