export const MAX_TRANSCRIPTION_UPLOAD_BYTES = 20 * 1024 * 1024
export const MAX_TRANSCRIPTION_UPLOAD_MB = MAX_TRANSCRIPTION_UPLOAD_BYTES / (1024 * 1024)

export const SUPPORTED_AUDIO_MIME_TYPES = [
  'audio/aac',
  'audio/x-aac',
  'audio/flac',
  'audio/x-flac',
  'audio/m4a',
  'audio/x-m4a',
  'audio/mp3',
  'audio/mpeg',
  'audio/mpga',
  'audio/mp4',
  'audio/ogg',
  'audio/oga',
  'audio/opus',
  'audio/wav',
  'audio/wave',
  'audio/x-wav',
  'audio/vnd.wave',
  'audio/webm',
  'audio/3gpp',
  'audio/3gpp2',
  'video/mp4',
  'video/webm',
  'video/mpeg',
  'video/3gpp',
  'application/ogg',
  'application/octet-stream',
] as const

export const SUPPORTED_AUDIO_EXTENSIONS = [
  'mp3',
  'mp4',
  'mpeg',
  'mpga',
  'm4a',
  'wav',
  'webm',
  'ogg',
  'oga',
  'opus',
  'flac',
  'aac',
  '3gp',
] as const

export const SUPPORTED_AUDIO_DESCRIPTION = 'MP3, M4A/MP4, WAV, WebM, OGG/Opus, FLAC, AAC, or MPEG'

const MIME_TYPES = new Set<string>(SUPPORTED_AUDIO_MIME_TYPES)
const EXTENSIONS = new Set<string>(SUPPORTED_AUDIO_EXTENSIONS)

export function isSupportedAudioType(type: string, filename = ''): boolean {
  const normalized = type.toLowerCase().split(';')[0]?.trim() ?? ''
  if (normalized === '' || normalized === 'application/octet-stream') return true
  if (normalized.startsWith('audio/')) return true
  if (MIME_TYPES.has(normalized)) return true
  const extension = filename.toLowerCase().split('.').pop()
  return extension ? EXTENSIONS.has(extension) : false
}

export function formatMegabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 10 * 1024 * 1024 ? 0 : 1)} MB`
}

export function validateAudioFile(file: { name: string; size: number; type: string }): string | null {
  if (!isSupportedAudioType(file.type, file.name)) {
    return `Unsupported audio type${file.type ? ` (${file.type})` : ''}. Use ${SUPPORTED_AUDIO_DESCRIPTION}.`
  }
  if (file.size < 1) return 'That file is empty. Select another audio file.'
  if (file.size > MAX_TRANSCRIPTION_UPLOAD_BYTES) {
    return `That file is ${formatMegabytes(file.size)}. Select an audio file that is ${MAX_TRANSCRIPTION_UPLOAD_MB} MB or smaller.`
  }
  return null
}
