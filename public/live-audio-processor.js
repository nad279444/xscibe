class LiveAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.frames = new Float32Array(2048)
    this.offset = 0
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    if (!channel) return true
    let inputOffset = 0
    while (inputOffset < channel.length) {
      const count = Math.min(this.frames.length - this.offset, channel.length - inputOffset)
      this.frames.set(channel.subarray(inputOffset, inputOffset + count), this.offset)
      this.offset += count
      inputOffset += count
      if (this.offset === this.frames.length) {
        const frame = this.frames
        this.frames = new Float32Array(2048)
        this.offset = 0
        this.port.postMessage(frame, [frame.buffer])
      }
    }
    return true
  }
}

registerProcessor('live-audio-processor', LiveAudioProcessor)
