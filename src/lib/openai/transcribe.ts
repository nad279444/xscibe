import type { Transcript } from '@/app/types'
import { MAX_TRANSCRIPTION_UPLOAD_BYTES } from '@/lib/audio/uploadLimits'

export async function transcribeAudio(
  blob: Blob,
  filename: string,
  signal?: AbortSignal,
  durationSeconds?: number,
): Promise<Transcript> {
  if (blob.size > MAX_TRANSCRIPTION_UPLOAD_BYTES) {
    throw new Error('Audio must be 20 MB or smaller')
  }

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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ objectName: ticket.objectName, filename: safeName, durationSeconds }),
    signal,
  })
  const payload = (await response.json().catch(() => ({}))) as {
    error?: string
    transcript?: Transcript
  }

  if (!response.ok || !payload.transcript) {
    throw new Error(payload.error ?? `Transcription failed (${response.status})`)
  }
  return payload.transcript
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
