import { useEffect, useRef } from 'react'

interface TranscribingProps {
  label: string
  detail: string
  preview?: string
}

export default function Transcribing({ label, detail, preview = '' }: TranscribingProps) {
  const previewRef = useRef<HTMLParagraphElement | null>(null)

  useEffect(() => {
    previewRef.current?.scrollTo({ top: previewRef.current.scrollHeight, behavior: 'auto' })
  }, [preview])

  return (
    <div className="flex items-center flex-1 flex-col justify-center gap-10 md:gap-14 text-center pb-24 p-4">
      <div className="flex flex-col gap-2 sm:gap-4">
        <h1 className="font-semibold text-4xl sm:text-5xl md:text-6xl">
          <span className="text-blue-400 bold">{label}</span>
        </h1>
        <p className="text-slate-500">{detail}</p>
      </div>
      {preview ? (
        <div className="w-full max-w-2xl rounded-2xl border border-sky-200 bg-white/85 p-5 text-left shadow-sm">
          <p className="text-xs uppercase tracking-[0.18em] text-blue-600">Live transcript</p>
          <p
            ref={previewRef}
            aria-live="polite"
            className="mt-3 max-h-[40vh] overflow-y-auto whitespace-pre-wrap break-words text-base leading-7 text-slate-900"
          >
            {preview}
            <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-blue-500 align-middle" />
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2 sm:gap-3 max-w-[400px] mx-auto w-full">
          {[0, 1, 2].map((value) => (
            <div
              key={value}
              className={'rounded-full h-2 sm:h-3 bg-slate-400 loading ' + `loading${value}`}
            ></div>
          ))}
        </div>
      )}
    </div>
  )
}
