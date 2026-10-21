import { clientIp, jsonError, rateLimit } from './_shared'

export const config = { runtime: 'edge' }

const MAX_CONTEXT_CHARS = 24_000
const MAX_INSTRUCTIONS_CHARS = 8_000

interface LiveBody {
  sdp?: unknown
  context?: unknown
}

export default async function handler(request: Request): Promise<Response> {
  if (request.method !== 'POST') {
    return jsonError(405, 'Method not allowed')
  }
  if (!rateLimit(clientIp(request), 6)) {
    return jsonError(429, 'Too many voice sessions, slow down')
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) {
    return jsonError(503, 'Server is missing OPENAI_API_KEY')
  }

  let body: LiveBody
  try {
    body = (await request.json()) as LiveBody
  } catch {
    return jsonError(400, 'Invalid JSON body')
  }

  if (typeof body.sdp !== 'string' || !body.sdp.includes('v=0')) {
    return jsonError(400, 'sdp is required')
  }

  const context =
    typeof body.context === 'string' ? body.context.slice(0, MAX_CONTEXT_CHARS) : ''

  const instructions = context
    ? [
        'You are XScribe, a voice assistant speaking with the person whose recording transcript is below.',
        'Answer questions about the recording conversationally and briefly.',
        'Reference moments with timestamps like [12:34] when useful.',
        'If something is not in the transcript, say so.',
        '',
        'Transcript:',
        context,
      ].join('\n').slice(0, MAX_INSTRUCTIONS_CHARS)
    : [
        'You are XScribe, a concise and friendly voice assistant.',
        'The user has not provided a transcript yet; offer to help them transcribe a recording.',
      ].join('\n')

  const upstream = await fetch('https://api.openai.com/v1/live/sessions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session: {
        model: process.env.LIVE_MODEL ?? 'gpt-live-1',
        instructions,
        audio: {
          output: { voice: process.env.LIVE_VOICE ?? 'marin' },
        },
        delegation: {
          type: 'responses',
          responses: {
            model: process.env.CHAT_MODEL ?? 'gpt-6-luna',
          },
        },
      },
      transport: {
        type: 'webrtc',
        sdp: body.sdp,
      },
    }),
  })

  const payload = await upstream.text()
  return new Response(payload, {
    status: upstream.status,
    headers: { 'Content-Type': 'application/json' },
  })
}
