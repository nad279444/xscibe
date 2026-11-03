import type { Segment, Transcript, Word } from '@/app/types'
import { analyzeTranscript } from '@/lib/asr/guard'
import {
  isSupportedAudioType,
  MAX_TRANSCRIPTION_UPLOAD_BYTES,
  MAX_TRANSCRIPTION_UPLOAD_MB,
  SUPPORTED_AUDIO_DESCRIPTION,
} from '@/lib/audio/uploadLimits'
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

interface OpenAIStreamEvent {
  type?: unknown
  delta?: unknown
  text?: unknown
  start?: unknown
  end?: unknown
  duration?: unknown
  message?: unknown
  error?: { message?: unknown }
  segment?: { text?: unknown; start?: unknown; end?: unknown }
  usage?: { seconds?: unknown }
}

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function estimateSegments(text: string, duration: number): OpenAISegment[] {
  const trimmed = text.trim()
  if (!trimmed) return []
  const sentences = trimmed.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [trimmed]
  const groups: string[] = []
  let current = ''
  for (const sentence of sentences) {
    const piece = sentence.trim()
    if (!piece) continue
    if (current && current.length + piece.length > 180) {
      groups.push(current)
      current = piece
    } else {
      current = current ? `${current} ${piece}` : piece
    }
  }
  if (current) groups.push(current)
  const totalChars = groups.reduce((sum, group) => sum + group.length, 0) || 1
  let cursor = 0
  return groups.map((group) => {
    const span = duration * (group.length / totalChars)
    const start = cursor
    cursor += span
    return { text: group, start, end: cursor }
  })
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

function buildUpstreamForm(file: File, stream: boolean): FormData {
  const form = new FormData()
  form.append('file', file, file.name || 'audio.wav')
  form.append('model', process.env.TRANSCRIBE_MODEL ?? 'gpt-transcribe')
  form.append('response_format', 'json')
  if (stream) form.append('stream', 'true')
  return form
}

async function streamTranscription(
  file: File,
  apiKey: string,
  signal: AbortSignal,
  fallbackDuration: number,
): Promise<Response> {
  const endpoint = 'https://api.openai.com/v1/audio/transcriptions'
  const requestUpstream = (stream: boolean) =>
    fetch(endpoint, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: buildUpstreamForm(file, stream),
      signal,
    })

  let upstream = await requestUpstream(true)
  if (!upstream.ok) {
    let detail = await upstream.text().catch(() => '')
    if (upstream.status === 400 && /stream|response_format/i.test(detail)) {
      upstream = await requestUpstream(false)
      detail = upstream.ok ? '' : await upstream.text().catch(() => '')
    }
    if (!upstream.ok) {
      return jsonError(502, `OpenAI transcription failed (${upstream.status}): ${detail.slice(0, 300)}`)
    }
  }

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder()
      const send = (event: object) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }

      if (!(upstream.headers.get('content-type') ?? '').includes('text/event-stream')) {
        try {
          const payload = (await upstream.json()) as OpenAITranscript
          send({ type: 'final', transcript: normalizeTranscript(payload, fallbackDuration) })
        } catch (error) {
          send({
            type: 'error',
            message: error instanceof Error ? error.message : 'Could not read the transcription response',
          })
        }
        controller.close()
        return
      }

      const reader = upstream.body?.getReader()
      if (!reader) {
        send({ type: 'error', message: 'Transcription response has no body' })
        controller.close()
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''
      let text = ''
      let doneText: string | null = null
      let duration = fallbackDuration
      const segments: OpenAISegment[] = []

      const handleEvent = (event: OpenAIStreamEvent) => {
        const type = typeof event.type === 'string' ? event.type : ''
        if (type === 'error') {
          const message =
            (typeof event.error?.message === 'string' && event.error.message) ||
            (typeof event.message === 'string' && event.message) ||
            'OpenAI transcription failed'
          throw new Error(message)
        }

        const eventDuration = toNumber(event.duration, toNumber(event.usage?.seconds, 0))
        if (eventDuration > 0) duration = eventDuration

        const segment = event.segment
        if (segment && typeof segment.text === 'string') {
          segments.push({ text: segment.text, start: segment.start, end: segment.end })
        } else if (type.includes('segment') && typeof event.text === 'string') {
          segments.push({ text: event.text, start: event.start, end: event.end })
        }

        if (typeof event.delta === 'string' && event.delta) {
          text += event.delta
          send({ type: 'partial', text })
        } else if (type === 'transcript.text.done') {
          if (typeof event.text === 'string') doneText = event.text
        }
      }

      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let newline = buffer.indexOf('\n')
          while (newline >= 0) {
            const line = buffer.slice(0, newline).replace(/\r$/, '').trim()
            buffer = buffer.slice(newline + 1)
            if (line.startsWith('data:')) {
              const data = line.slice(5).trim()
              if (data && data !== '[DONE]') {
                try {
                  handleEvent(JSON.parse(data) as OpenAIStreamEvent)
                } catch (error) {
                  if (error instanceof SyntaxError) continue
                  throw error
                }
              }
            }
            newline = buffer.indexOf('\n')
          }
        }

        const finalText = (doneText ?? text).trim()
        const finalSegments = segments.length > 0 ? segments : estimateSegments(finalText, duration)
        send({
          type: 'final',
          transcript: normalizeTranscript(
            { text: finalText, duration, segments: finalSegments },
            fallbackDuration,
          ),
        })
      } catch (error) {
        if (!signal.aborted) {
          send({
            type: 'error',
            message: error instanceof Error ? error.message : 'Transcription streaming failed',
          })
        }
      } finally {
        reader.releaseLock()
        try {
          controller.close()
        } catch {
          // The client may have disconnected and already closed the stream.
        }
      }
    },
    cancel() {
      void upstream.body?.cancel()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
      Connection: 'keep-alive',
    },
  })
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
      if (size > MAX_TRANSCRIPTION_UPLOAD_BYTES) {
        await object.delete({ ignoreNotFound: true }).catch(() => undefined)
        return jsonError(413, `Audio must be ${MAX_TRANSCRIPTION_UPLOAD_MB} MB or smaller`)
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
  if (!isSupportedAudioType(file.type, file.name)) {
    return jsonError(400, `Unsupported audio type. Use ${SUPPORTED_AUDIO_DESCRIPTION}.`)
  }
  if (file.size > (isGcsUpload ? MAX_TRANSCRIPTION_UPLOAD_BYTES : MAX_UPLOAD_BYTES)) {
    return jsonError(413, isGcsUpload ? `Audio must be ${MAX_TRANSCRIPTION_UPLOAD_MB} MB or smaller` : 'Audio must be 4 MB or smaller when sent directly to the transcription route')
  }
  const fallbackDuration = typeof formDuration === 'string' ? toNumber(Number(formDuration)) : 0

  if (request.headers.get('accept')?.includes('text/event-stream')) {
    try {
      return await streamTranscription(file, apiKey, request.signal, fallbackDuration)
    } catch (error) {
      if (request.signal.aborted) return jsonError(408, 'Transcription request was cancelled')
      const message = error instanceof Error ? error.message : 'Unknown upstream error'
      return jsonError(502, `Could not reach OpenAI transcription: ${message}`)
    }
  }

  try {
    const upstream = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}` },
      body: buildUpstreamForm(file, false),
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
