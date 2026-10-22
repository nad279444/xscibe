import { openRealtimeTranscription, type RealtimeTranscription } from '../openai/realtimeTranscription'

export type LiveCapture = RealtimeTranscription

export function startLiveCapture(handlers: Parameters<typeof openRealtimeTranscription>[0]): Promise<LiveCapture> {
  return openRealtimeTranscription(handlers)
}
