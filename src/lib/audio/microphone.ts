export const NOISE_CANCELLING_AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  channelCount: 1,
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
}

export function getMicrophoneStream(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: NOISE_CANCELLING_AUDIO_CONSTRAINTS,
    video: false,
  })
}
