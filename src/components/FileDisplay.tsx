import { useEffect, useRef } from 'react'
import type { AudioSource, Device, ModelKey } from '../app/types'
import { MODELS, MODEL_KEYS } from '../lib/asr/registry'

interface FileDisplayProps {
  source: AudioSource
  modelKey: ModelKey
  device: Device | null
  onModelChange: (modelKey: ModelKey) => void
  onSubmit: () => void
  onReset: () => void
}

export default function FileDisplay({
  source,
  modelKey,
  device,
  onModelChange,
  onSubmit,
  onReset,
}: FileDisplayProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    const element = audioRef.current
    if (!element) return
    const blob = source.kind === 'file' ? source.file : source.blob
    const url = URL.createObjectURL(blob)
    element.src = url
    return () => {
      URL.revokeObjectURL(url)
      element.removeAttribute('src')
    }
  }, [source])

  return (
    <main className="flex-1 p-4 flex flex-col gap-3 text-center sm:gap-4 justify-center pb-20 w-full max-w-prose mx-auto">
      <h1 className="font-semibold text-4xl sm:text-5xl md:text-6xl">
        Your <span className="text-blue-400 bold">File</span>
      </h1>
      <div className="flex flex-col text-left my-4">
        <h3 className="font-semibold">Name</h3>
        <p className="truncate">{source.name}</p>
      </div>
      <div className="flex flex-col mb-2">
        <audio ref={audioRef} className="w-full" controls>
          Your browser does not support the audio element.
        </audio>
      </div>
      <div className="flex flex-col text-left gap-1 mb-2">
        <label htmlFor="model-select" className="text-xs sm:text-sm font-medium text-slate-500">
          Model
        </label>
        <div className="flex items-center gap-3">
          <select
            id="model-select"
            value={modelKey}
            onChange={(event) => onModelChange(event.target.value as ModelKey)}
            className="flex-1 bg-white duration-200 p-2 rounded outline-none focus:outline-none"
          >
            {MODEL_KEYS.map((key) => (
              <option key={key} value={key}>
                {MODELS[key].label}
                {key !== 'small' ? ' — faster' : ''}
              </option>
            ))}
          </select>
          {device && (
            <span className="text-xs text-slate-400 whitespace-nowrap">
              {device === 'webgpu' ? 'GPU acceleration on' : 'WASM fallback'}
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={onReset}
          className="text-slate-400 hover:text-blue-600 duration-200"
        >
          Reset
        </button>
        <button
          type="button"
          onClick={onSubmit}
          className="specialBtn px-3 p-2 rounded-lg text-blue-400 flex items-center gap-2 font-medium"
        >
          <p>Transcribe</p>
          <i className="fa-solid fa-pen-nib"></i>
        </button>
      </div>
    </main>
  )
}
