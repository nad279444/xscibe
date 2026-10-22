import { useEffect, useRef } from 'react'
import type { Segment } from '../app/types'
import { formatTimestamp } from '../lib/export/captions'

interface LiveViewProps {
  segments: Segment[]
  partialText: string
  processing: boolean
  stopping?: boolean
  onStop: () => void
}

export default function LiveView({
  segments,
  partialText,
  processing,
  stopping = false,
  onStop,
}: LiveViewProps) {
  const transcriptRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: 'auto' })
  }, [segments, partialText])

  return (
    <main className="flex-1 min-h-0 p-4 flex flex-col gap-4 text-center pb-20 w-full max-w-2xl mx-auto">
      <div className="flex items-center justify-between gap-4">
        <h1 className="font-semibold text-4xl sm:text-5xl">
          <span className="text-blue-700 bold">Listening…</span>
        </h1>
        <span
          className={
            'h-3 w-3 rounded-full ' +
            (processing ? 'bg-amber-400 animate-pulse' : 'bg-rose-400 animate-pulse')
          }
          title={processing ? 'Processing speech' : 'Listening for speech'}
        ></span>
      </div>
      <div ref={transcriptRef} className="text-left flex-1 min-h-64 overflow-y-auto rounded-2xl border border-sky-200 bg-white/85 p-5 shadow-sm space-y-5">
        {segments.length === 0 && !partialText && (
          <p className="text-slate-500 italic">
            Speak naturally — phrases appear here as soon as you pause.
          </p>
        )}
        {segments.map((segment) => (
          <div key={segment.id} className="flex flex-col gap-1.5 text-left text-base leading-7 text-slate-900">
            <span className="text-xs text-slate-500 font-mono">
              [{formatTimestamp(segment.start)}]
            </span>
            <p className="whitespace-pre-wrap break-words">{segment.text}</p>
          </div>
        ))}
        {partialText && (
          <p className="border-l-2 border-blue-400 pl-3 text-base leading-7 text-blue-800 whitespace-pre-wrap break-words" aria-live="polite">{partialText}</p>
        )}
      </div>
      <div className="flex justify-end">
        {stopping ? (
          <p className="text-slate-400 italic text-sm">Finishing last phrase…</p>
        ) : (
          <button
            type="button"
            onClick={onStop}
            className="specialBtn px-4 py-2 rounded-lg text-blue-700 flex items-center gap-2"
          >
            <p>Finish &amp; review</p>
            <i className="fa-solid fa-stop"></i>
          </button>
        )}
      </div>
    </main>
  )
}
