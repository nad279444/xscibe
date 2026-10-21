import type { Transcript } from '../../app/types'

export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
}

export interface StreamChatOptions {
  transcript: Transcript
  question: string
  history?: ChatMessage[]
}

export async function* streamChat(
  options: StreamChatOptions,
): AsyncGenerator<string, void, unknown> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transcript: options.transcript.text,
      question: options.question,
      history: options.history ?? [],
    }),
  })

  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`
    try {
      const data = (await response.json()) as { error?: string }
      if (data.error) detail = data.error
    } catch {
      // non-JSON error body
    }
    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error('Response has no body')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) return
      yield decoder.decode(value, { stream: true })
    }
  } finally {
    reader.releaseLock()
  }
}
