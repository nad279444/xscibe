import { clientIp, jsonError, rateLimit } from '@/lib/server/rateLimit'

export const dynamic = 'force-dynamic'

const MAX_TRANSCRIPT_CHARS = 150_000
const MAX_QUESTION_CHARS = 4_000
const MAX_HISTORY_TURNS = 6

interface HistoryMessage {
  role?: unknown
  text?: unknown
}

interface ChatBody {
  transcript?: unknown
  question?: unknown
  history?: unknown
}

function asHistory(value: unknown): { role: 'user' | 'assistant'; content: string }[] {
  if (!Array.isArray(value)) return []
  const turns: { role: 'user' | 'assistant'; content: string }[] = []
  for (const raw of value.slice(-MAX_HISTORY_TURNS)) {
    const message = raw as HistoryMessage
    if (message.role !== 'user' && message.role !== 'assistant') continue
    if (typeof message.text !== 'string') continue
    turns.push({ role: message.role, content: message.text.slice(0, 2_000) })
  }
  return turns
}

export async function POST(request: Request): Promise<Response> {
  if (!rateLimit(clientIp(request), 8)) {
    return jsonError(429, 'Too many requests, slow down')
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    return jsonError(503, 'Server is missing OPENAI_API_KEY')
  }

  let body: ChatBody
  try {
    body = (await request.json()) as ChatBody
  } catch {
    return jsonError(400, 'Invalid JSON body')
  }

  const transcript =
    typeof body.transcript === 'string' ? body.transcript.slice(0, MAX_TRANSCRIPT_CHARS) : ''
  const question =
    typeof body.question === 'string' ? body.question.trim().slice(0, MAX_QUESTION_CHARS) : ''

  if (!question) return jsonError(400, 'question is required')
  if (!transcript) return jsonError(400, 'transcript is required')

  const instructions = [
    'You are XScribe, an assistant that answers questions about a transcript of a recorded conversation.',
    'Answer using only the transcript. If the answer is not in the transcript, say so plainly.',
    'Reference specific moments with timestamps like [12:34] when relevant.',
    'Be concise and conversational.',
    '',
    'Transcript:',
    transcript,
  ].join('\n')

  const history = asHistory(body.history)

  const upstream = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: process.env.CHAT_MODEL ?? 'gpt-6-luna',
      instructions,
      input: [
        ...history,
        { role: 'user', content: question },
      ],
      stream: true,
      reasoning: { effort: process.env.CHAT_REASONING ?? 'low' },
    }),
  })

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => '')
    return jsonError(502, `Upstream error ${upstream.status}: ${detail.slice(0, 300)}`)
  }

  const upstreamReader = upstream.body.getReader()
  const encoder = new TextEncoder()
  const decoder = new TextDecoder()

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      let buffer = ''
      for (;;) {
        const { done, value } = await upstreamReader.read()
        if (done) {
          controller.close()
          return
        }
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (!payload || payload === '[DONE]') continue
          try {
            const event = JSON.parse(payload) as { type?: string; delta?: unknown }
            if (
              event.type === 'response.output_text.delta' &&
              typeof event.delta === 'string'
            ) {
              controller.enqueue(encoder.encode(event.delta))
            }
          } catch {
            // partial JSON across chunks is buffered by the outer loop
          }
        }
      }
    },
    cancel() {
      void upstreamReader.cancel()
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  })
}

export async function GET(): Promise<Response> {
  return jsonError(405, 'Method not allowed')
}
