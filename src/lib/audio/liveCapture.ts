import { MicVAD } from '@ricky0123/vad-web'

export interface UtteranceHandlers {
  onSpeechStart: (startTimeSeconds: number) => void
  onUtterance: (audio: Float32Array, startTimeSeconds: number) => void
  onError: (message: string) => void
}

export interface LiveCapture {
  stop: () => void
}

export async function startLiveCapture(handlers: UtteranceHandlers): Promise<LiveCapture> {
  const t0 = performance.now()
  let utteranceStartSeconds = 0
  let stopped = false

  const elapsed = () => (performance.now() - t0) / 1000

  let vad: MicVAD
  try {
    vad = await MicVAD.new({
      model: 'legacy',
      // Self-hosted: document.currentScript is null for module scripts, so the
      // library default ("./") can resolve against deep routes. Serve from root
      // instead (worklet + silero_vad_legacy.onnx live in public/).
      baseAssetPath: '/',
      ortConfig: (ort) => {
        const runtimeVersion = ort.env.versions.common
        const wasmAssetBase = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${runtimeVersion}/dist/`
        ort.env.wasm.wasmPaths = {
          mjs: `${wasmAssetBase}ort-wasm-simd-threaded.mjs`,
          wasm: `${wasmAssetBase}ort-wasm-simd-threaded.wasm`,
        }
      },
      positiveSpeechThreshold: 0.6,
      negativeSpeechThreshold: 0.35,
      minSpeechMs: 150,
      redemptionMs: 300,
      onSpeechStart: () => {
        utteranceStartSeconds = elapsed()
        handlers.onSpeechStart(utteranceStartSeconds)
      },
      onSpeechEnd: (audio) => {
        if (!stopped) handlers.onUtterance(audio, utteranceStartSeconds)
      },
      onVADMisfire: () => {
        // ignore: shorter than minSpeechFrames
      },
    })
    await vad.start()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    handlers.onError(message)
    throw error instanceof Error ? error : new Error(message)
  }

  return {
    stop: () => {
      if (stopped) return
      stopped = true
      void vad.destroy().catch(() => {})
    },
  }
}
