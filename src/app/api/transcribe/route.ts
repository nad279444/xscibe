import type { Segment, Transcript, Word } from '@/app/types'
import { analyzeTranscript } from '@/lib/asr/guard'
import { audioBucket } from '@/lib/server/gcs'
import { clientIp, jsonError, rateLimit } from '@/lib/server/rateLimit'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'
export const maxDuration = 300

const MAX_UPLOAD_BYTES = 4 * 1024 * 1024

interface OpenAIWord {
  word?: unknown
  start?: unknown
  end?: unknown
}

interface OpenAISegment {
  text?: unknown
  start?: unknown
  end?: unknown
}

interface OpenAITranscript {
  text?: unknown
  duration?: unknown
  usage?: { seconds?: unknown }
  words?: OpenAIWord[]
  segments?: OpenAISegment[]
}

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function normalizeTranscript(payload: OpenAITranscript, fallbackDuration: number): Transcript {
  const text = typeof payload.text === 'string' ? payload.text.trim() : ''
  const duration = toNumber(payload.usage?.seconds, toNumber(payload.duration, fallbackDuration))
  const words: Word[] = Array.isArray(payload.words)
    ? payload.words.flatMap((item) => {
        if (typeof item.word !== 'string') return []
        return [{ text: item.word, start: toNumber(item.start), end: toNumber(item.end) }]
      })
    : []

  const rawSegments = Array.isArray(payload.segments) ? payload.segments : []
  let segments: Segment[] = rawSegments.map((item, index) => {
    const start = toNumber(item.start)
    const end = Math.max(start, toNumber(item.end, start))
    const segmentText = typeof item.text === 'string' ? item.text.trim() : ''
    let segmentWords = words.filter((word) => word.start >= start && word.start <= end)
    if (segmentWords.length === 0 && segmentText) {
      const tokens = segmentText.trim().split(/\s+/)
      const span = Math.max(0, end - start)
      segmentWords = tokens.map((token, tokenIndex) => ({
        text: token,
        start: start + (span * tokenIndex) / tokens.length,
        end: start + (span * (tokenIndex + 1)) / tokens.length,
        approximate: true,
      }))
    }
    return {
      id: index,
      text: segmentText,
      start,
      end,
      words: segmentWords,
      flagged: false,
      flagReasons: [],
      confidence: 1,
    }
  })

  if (segments.length === 0 && (text || words.length > 0)) {
    const start = words[0]?.start ?? 0
    const end = words[words.length - 1]?.end ?? toNumber(payload.duration, duration)
    const fallbackWords = words.length ? words : text.split(/\s+/).filter(Boolean).map((token, index, all) => ({
      text: token,
      start: start + ((end - start) * index) / all.length,
      end: start + ((end - start) * (index + 1)) / all.length,
      approximate: true,
    }))
    segments = [{
      id: 0,
      text,
      start,
      end,
      words: fallbackWords,
      flagged: false,
      flagReasons: [],
      confidence: 1,
    }]
  }

  const transcriptDuration = Math.max(duration, ...segments.map((segment) => segment.end), 0)
  return analyzeTranscript({ text, segments, duration: transcriptDuration })
}

export async function POST(request: Request): Promise<Response> {
  if (!rateLimit(clientIp(request), 30)) {
    return jsonError(429, 'Too many transcription requests, slow down')
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) return jsonError(503, 'Server is missing OPENAI_API_KEY')

  let file: File
  let formDuration: FormDataEntryValue | undefined
  let isGcsUpload = false
  const contentType = request.headers.get('content-type') ?? ''
  if (contentType.includes('application/json')) {
    isGcsUpload = true
    let body: { objectName?: unknown; filename?: unknown; durationSeconds?: unknown }
    try {
      body = (await request.json()) as typeof body
    } catch {
      return jsonError(400, 'Invalid transcription request')
    }
    if (typeof body.objectName !== 'string' || !/^audio-uploads\/[0-9a-f-]{36}\/[\w.-]{1,120}$/i.test(body.objectName)) {
      return jsonError(400, 'Invalid audio object')
    }
    try {
      const object = audioBucket().file(body.objectName)
      const [metadata] = await object.getMetadata()
      const size = Number(metadata.size)
      if (!Number.isFinite(size) || size < 1) return jsonError(400, 'Audio file is empty')
      if (size > 20 * 1024 * 1024) {
        await object.delete({ ignoreNotFound: true }).catch(() => undefined)
        return jsonError(413, 'Audio must be 20 MB or smaller')
      }
      const [audio] = await object.download()
      const audioBytes = new Uint8Array(audio.byteLength)
      audioBytes.set(audio)
      file = new File([audioBytes.buffer], typeof body.filename === 'string' ? body.filename : 'audio', {
        type: metadata.contentType || 'application/octet-stream',
      })
      await object.delete({ ignoreNotFound: true }).catch(() => undefined)
    } catch {
      return jsonError(400, 'Could not retrieve the uploaded audio from GCS')
    }
    formDuration = typeof body.durationSeconds === 'number' ? String(body.durationSeconds) : undefined
  } else {
    let form: FormData
    try {
      form = await request.formData()
    } catch {
      return jsonError(400, 'Expected multipart audio upload')
    }
    const uploadedFile = form.get('file')
    if (!(uploadedFile instanceof File)) return jsonError(400, 'Audio file is required')
    file = uploadedFile
    formDuration = form.get('durationSeconds') ?? undefined
  }
  if (file.size === 0) return jsonError(400, 'Audio file is empty')
  if (file.size > (isGcsUpload ? 20 * 1024 * 1024 : MAX_UPLOAD_BYTES)) {
    return jsonError(413, isGcsUpload ? 'Audio must be 20 MB or smaller' : 'Audio must be 4 MB or smaller when sent directly to the transcription route')
  }
  const fallbackDuration = typeof formDuration === 'string' ? toNumber(Number(formDuration)) : 0

  const upstreamForm = new FormData()
  upstreamForm.append('file', file, file.name || 'audio.wav')
  upstreamForm.append('model', 'gpt-transcribe')
  upstreamForm.append('response_format', 'json')

  try {
    const upstream = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: upstreamForm,
      signal: request.signal,
    })

    if (!upstream.ok) {
      const detail = await upstream.text().catch(() => '')
      return jsonError(502, `OpenAI transcription failed (${upstream.status}): ${detail.slice(0, 300)}`)
    }

    const payload = (await upstream.json()) as OpenAITranscript
    return Response.json({ transcript: normalizeTranscript(payload, fallbackDuration) })
  } catch (error) {
    if (request.signal.aborted) return jsonError(408, 'Transcription request was cancelled')
    const message = error instanceof Error ? error.message : 'Unknown upstream error'
    return jsonError(502, `Could not reach OpenAI transcription: ${message}`)
  }
}

export async function GET(): Promise<Response> {
  return jsonError(405, 'Method not allowed')
}

