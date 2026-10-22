import { clientIp, jsonError, rateLimit } from '@/lib/server/rateLimit'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function POST(request: Request): Promise<Response> {
  if (!rateLimit(clientIp(request), 20)) {
    return jsonError(429, 'Too many live transcription sessions, slow down')
  }

  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) return jsonError(503, 'Server is missing OPENAI_API_KEY')

  const upstream = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session: {
        type: 'transcription',
        audio: {
          input: {
            format: { type: 'audio/pcm', rate: 24000 },
            transcription: { model: 'gpt-live-transcribe', delay: 'low' },
            turn_detection: null,
          },
        },
      },
    }),
    signal: request.signal,
  })

  const payload = (await upstream.json().catch(() => null)) as
    | { value?: string; client_secret?: { value?: string }; error?: { message?: string } }
    | null
  if (!upstream.ok) {
    return jsonError(502, payload?.error?.message ?? 'Could not create a live transcription session')
  }

  const token = payload?.value ?? payload?.client_secret?.value
  if (!token) return jsonError(502, 'OpenAI did not return a temporary session token')
  return Response.json({ token })
}

export async function GET(): Promise<Response> {
  return jsonError(405, 'Method not allowed')
}
