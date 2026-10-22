import { useEffect, useRef } from 'react'
import type { AudioSource } from '../app/types'

interface FileDisplayProps {
  source: AudioSource
  onSubmit: (durationSeconds?: number) => void
  onReset: () => void
}

export default function FileDisplay({
  source,
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
      <p className="rounded-lg border border-cyan-300/15 bg-cyan-300/[0.04] p-3 text-left text-xs leading-relaxed text-slate-400">
        Audio is sent to our transcription service. Files must be 4 MB or smaller; word seek positions are estimated.
      </p>
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
          onClick={() => onSubmit(audioRef.current?.duration)}
          className="specialBtn px-3 p-2 rounded-lg text-blue-400 flex items-center gap-2 font-medium"
        >
          <p>Transcribe</p>
          <i className="fa-solid fa-pen-nib"></i>
        </button>
      </div>
    </main>
  )
}
