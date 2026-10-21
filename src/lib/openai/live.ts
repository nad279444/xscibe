import type { VoiceCaption } from '../../app/types'

export interface LiveHandlers {
  onConnected: () => void
  onCaption: (caption: VoiceCaption) => void
  onEnded: () => void
  onError: (message: string) => void
}

export interface LiveSession {
  close: () => void
}

interface UpstreamEvent {
  type?: string
  delta?: string
  error?: unknown
}

function errorMessageFrom(value: unknown): string {
  if (value && typeof value === 'object' && 'message' in value) {
    const message = (value as { message?: unknown }).message
    if (typeof message === 'string') return message
  }
  return 'Voice session error'
}

function waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise((resolve) => {
    const timeout = setTimeout(resolve, 4000)
    const onStateChange = () => {
      if (pc.iceGatheringState === 'complete') {
        clearTimeout(timeout)
        pc.removeEventListener('icegatheringstatechange', onStateChange)
        resolve()
      }
    }
    pc.addEventListener('icegatheringstatechange', onStateChange)
  })
}

export async function startLiveSession(
  context: string,
  handlers: LiveHandlers,
): Promise<LiveSession> {
  const mic = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  })

  const pc = new RTCPeerConnection()
  const speaker = new Audio()
  speaker.autoplay = true

  let closed = false
  let userText = ''
  let assistantText = ''

  const cleanup = () => {
    try {
      pc.close()
    } catch {
      // already closed
    }
    for (const track of mic.getTracks()) track.stop()
    speaker.srcObject = null
  }

  const finish = () => {
    if (closed) return
    closed = true
    cleanup()
    handlers.onEnded()
  }

  const fail = (message: string) => {
    if (closed) return
    closed = true
    cleanup()
    handlers.onError(message)
  }

  pc.ontrack = (event) => {
    const stream = event.streams[0] ?? new MediaStream([event.track])
    speaker.srcObject = stream
    void speaker.play().catch(() => {})
  }

  pc.addEventListener('connectionstatechange', () => {
    if (pc.connectionState === 'failed') {
      fail('Voice connection failed')
    }
  })

  const dc = pc.createDataChannel('oai-events')

  dc.addEventListener('message', (event) => {
    let message: UpstreamEvent
    try {
      message = JSON.parse(String(event.data)) as UpstreamEvent
    } catch {
      return
    }
    const type = message.type ?? ''
    const delta = typeof message.delta === 'string' ? message.delta : ''

    if (type === 'session.started') {
      handlers.onConnected()
      return
    }
    if (type === 'session.closed') {
      finish()
      return
    }
    if (type === 'error') {
      fail(errorMessageFrom(message.error))
      return
    }
    if (type.includes('input_transcript')) {
      if (type.endsWith('.delta')) {
        userText += delta
        handlers.onCaption({ role: 'user', text: userText, final: false })
      } else if (type.includes('done') || type.includes('complete') || type.includes('final')) {
        handlers.onCaption({ role: 'user', text: userText, final: true })
      }
      return
    }
    if (type.includes('output_transcript')) {
      if (type.endsWith('.delta')) {
        assistantText += delta
        handlers.onCaption({ role: 'assistant', text: assistantText, final: false })
      } else if (type.includes('done') || type.includes('complete') || type.includes('final')) {
        handlers.onCaption({ role: 'assistant', text: assistantText, final: true })
      }
    }
  })

  try {
    for (const track of mic.getTracks()) {
      pc.addTrack(track, mic)
    }

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await waitForIceGathering(pc)

    const response = await fetch('/api/live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sdp: pc.localDescription?.sdp ?? offer.sdp,
        context,
      }),
    })

    if (!response.ok) {
      let detail = `HTTP ${response.status}`
      try {
        const data = (await response.json()) as { error?: string }
        if (data.error) detail = data.error
      } catch {
        // non-JSON error body
      }
      throw new Error(detail)
    }

    const data = (await response.json()) as { transport?: { sdp?: string } }
    const answerSdp = data.transport?.sdp
    if (!answerSdp) {
      throw new Error('Voice session did not return an SDP answer')
    }
    await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
  } catch (error) {
    cleanup()
    throw error instanceof Error ? error : new Error(String(error))
  }

  return {
    close: () => {
      if (closed) return
      try {
        if (dc.readyState === 'open') {
          dc.send(JSON.stringify({ type: 'session.close' }))
        }
      } catch {
        // channel already closing
      }
      // Give the remote session a moment to emit session.closed (final usage),
      // then force local cleanup either way.
      const fallback = setTimeout(finish, 1500)
      const onClosed = () => {
        clearTimeout(fallback)
        finish()
      }
      if (dc.readyState === 'closed') {
        onClosed()
      } else {
        dc.addEventListener('close', onClosed, { once: true })
      }
    },
  }
}
