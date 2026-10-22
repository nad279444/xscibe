import { audioBucket } from '@/lib/server/gcs'
import { clientIp, jsonError, rateLimit } from '@/lib/server/rateLimit'
import { MAX_TRANSCRIPTION_UPLOAD_BYTES } from '@/lib/audio/uploadLimits'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const AUDIO_TYPES = new Set([
  'audio/mpeg',
  'audio/mp4',
  'audio/wav',
  'audio/x-wav',
  'audio/webm',
  'audio/ogg',
  'audio/flac',
  'video/mp4',
  'application/octet-stream',
])

export async function POST(request: Request): Promise<Response> {
  if (!rateLimit(`gcs-upload:${clientIp(request)}`, 8, 60 * 60 * 1000)) {
    return jsonError(429, 'Upload limit reached. Try again later.')
  }
  let body: { filename?: unknown; contentType?: unknown; size?: unknown }
  try {
    body = (await request.json()) as typeof body
  } catch {
    return jsonError(400, 'Invalid upload request')
  }

  if (typeof body.filename !== 'string' || typeof body.contentType !== 'string') {
    return jsonError(400, 'Filename and content type are required')
  }
  if (!Number.isSafeInteger(body.size) || (body.size as number) < 1 || (body.size as number) > MAX_TRANSCRIPTION_UPLOAD_BYTES) {
    return jsonError(413, 'Audio must be 20 MB or smaller')
  }
  if (!AUDIO_TYPES.has(body.contentType)) return jsonError(400, 'Unsupported audio type')

  try {
    const filename = body.filename.replace(/[^\w.-]+/g, '_').slice(-120) || 'audio.webm'
    const objectName = `audio-uploads/${crypto.randomUUID()}/${filename}`
    const [uploadUrl] = await audioBucket().file(objectName).getSignedUrl({
      version: 'v4',
      action: 'write',
      expires: Date.now() + 10 * 60 * 1000,
      contentType: body.contentType,
    })
    return Response.json({ uploadUrl, objectName })
  } catch (error) {
    console.error('[gcs] Could not create signed upload URL', {
      name: error instanceof Error ? error.name : 'UnknownError',
      message: error instanceof Error ? error.message : String(error),
    })
    return jsonError(503, 'Could not create a GCS upload URL. Check server credentials and bucket permissions.')
  }
}

export async function GET(): Promise<Response> {
  return jsonError(405, 'Method not allowed')
}
