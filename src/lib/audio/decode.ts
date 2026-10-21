import type { Word } from '../../app/types'

export interface DecodedAudio {
  samples: Float32Array
  duration: number
}

export const TARGET_SAMPLE_RATE = 16000

export async function decodeAudio(source: Blob): Promise<DecodedAudio> {
  const arrayBuffer = await source.arrayBuffer()
  const probe = new AudioContext()
  let buffer: AudioBuffer
  try {
    buffer = await probe.decodeAudioData(arrayBuffer)
  } finally {
    void probe.close()
  }

  const channel = buffer.getChannelData(0)
  if (buffer.sampleRate === TARGET_SAMPLE_RATE) {
    return { samples: channel, duration: buffer.length / buffer.sampleRate }
  }

  const targetLength = Math.ceil((buffer.length * TARGET_SAMPLE_RATE) / buffer.sampleRate)
  const resampleContext = new OfflineAudioContext(1, targetLength, TARGET_SAMPLE_RATE)
  const sourceNode = resampleContext.createBufferSource()
  sourceNode.buffer = buffer
  sourceNode.connect(resampleContext.destination)
  sourceNode.start(0)
  const rendered = await resampleContext.startRendering()
  return {
    samples: rendered.getChannelData(0),
    duration: buffer.duration,
  }
}

export function groupWordsIntoSegments(words: Word[], maxWords = 14): { text: string; start: number; end: number; words: Word[] }[] {
  const segments: { text: string; start: number; end: number; words: Word[] }[] = []
  let current: Word[] = []

  const flush = () => {
    if (current.length === 0) return
    const text = current.map((w) => w.text).join('').trim()
    segments.push({
      text,
      start: current[0]!.start,
      end: current[current.length - 1]!.end,
      words: current,
    })
    current = []
  }

  for (const word of words) {
    current.push(word)
    const endsSentence = /[.!?…]["')\]]?$/.test(word.text.trim())
    if (endsSentence || current.length >= maxWords) flush()
  }
  flush()
  return segments
}
