const buckets = new Map<string, { tokens: number; resetAt: number }>()

const MAX_BUCKETS = 5000

export function clientIp(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for')
  if (forwarded) {
    const first = forwarded.split(',')[0]
    if (first) return first.trim()
  }
  return request.headers.get('x-real-ip') ?? 'unknown'
}

export function rateLimit(ip: string, limit = 8, windowMs = 60_000): boolean {
  const now = Date.now()
  if (buckets.size > MAX_BUCKETS) {
    for (const [key, bucket] of buckets) {
      if (now > bucket.resetAt) buckets.delete(key)
    }
  }
  const bucket = buckets.get(ip)
  if (!bucket || now > bucket.resetAt) {
    buckets.set(ip, { tokens: limit - 1, resetAt: now + windowMs })
    return true
  }
  if (bucket.tokens <= 0) return false
  bucket.tokens -= 1
  return true
}

export function jsonError(status: number, message: string): Response {
  return Response.json({ error: message }, { status })
}
