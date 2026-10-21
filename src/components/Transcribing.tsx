import type { ModelProgress } from '../app/types'

interface TranscribingProps {
  label: string
  progress: ModelProgress | null
  partialText?: string
}

function statusLine(progress: ModelProgress | null): string {
  if (!progress) return 'Warming up cylinders'
  if (progress.phase === 'download') {
    const name = progress.file?.split('/').pop()
    return progress.progress !== undefined
      ? `Downloading ${name ?? 'model'} — ${Math.round(progress.progress)}%`
      : `Downloading ${name ?? 'model'}…`
  }
  if (progress.phase === 'load') return 'Loading model into memory'
  return 'Running inference'
}

export default function Transcribing({ label, progress, partialText }: TranscribingProps) {
  const percent = progress?.progress ?? null
  return (
    <div className="flex items-center flex-1 flex-col justify-center gap-10 md:gap-14 text-center pb-24 p-4">
      <div className="flex flex-col gap-2 sm:gap-4">
        <h1 className="font-semibold text-4xl sm:text-5xl md:text-6xl">
          <span className="text-blue-400 bold">{label}</span>
        </h1>
        <p className="text-slate-500">{statusLine(progress)}</p>
      </div>
      {percent !== null && (
        <div className="w-full max-w-[400px] mx-auto">
          <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
            <div
              className="h-full bg-blue-400 duration-200"
              style={{ width: `${Math.max(4, Math.min(100, percent))}%` }}
            ></div>
          </div>
        </div>
      )}
      <div className="flex flex-col gap-2 sm:gap-3 max-w-[400px] mx-auto w-full">
        {[0, 1, 2].map((value) => (
          <div
            key={value}
            className={'rounded-full h-2 sm:h-3 bg-slate-400 loading ' + `loading${value}`}
          ></div>
        ))}
      </div>
      {partialText && (
        <p className="max-w-prose text-slate-500 italic text-left">{partialText}</p>
      )}
    </div>
  )
}
