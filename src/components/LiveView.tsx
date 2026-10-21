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
  return (
    <main className="flex-1 p-4 flex flex-col gap-4 text-center pb-20 w-full max-w-prose mx-auto">
      <div className="flex items-center justify-between gap-4">
        <h1 className="font-semibold text-4xl sm:text-5xl">
          <span className="text-blue-400 bold">Listening…</span>
        </h1>
        <span
          className={
            'h-3 w-3 rounded-full ' +
            (processing ? 'bg-amber-400 animate-pulse' : 'bg-rose-400 animate-pulse')
          }
          title={processing ? 'Processing speech' : 'Listening for speech'}
        ></span>
      </div>
      <div className="text-left flex-1 overflow-y-auto rounded-xl border border-slate-200 p-4 space-y-3 bg-white min-h-48">
        {segments.length === 0 && !partialText && (
          <p className="text-slate-400 italic">
            Speak naturally — phrases appear here as soon as you pause.
          </p>
        )}
        {segments.map((segment) => (
          <div key={segment.id} className="flex gap-3 text-sm">
            <span className="text-slate-400 font-mono whitespace-nowrap">
              [{formatTimestamp(segment.start)}]
            </span>
            <span>{segment.text}</span>
          </div>
        ))}
        {partialText && (
          <p className="text-slate-500 italic text-sm">{partialText}</p>
        )}
      </div>
      <div className="flex justify-end">
        {stopping ? (
          <p className="text-slate-400 italic text-sm">Finishing last phrase…</p>
        ) : (
          <button
            type="button"
            onClick={onStop}
            className="specialBtn px-4 py-2 rounded-lg text-blue-400 flex items-center gap-2"
          >
            <p>Finish &amp; review</p>
            <i className="fa-solid fa-stop"></i>
          </button>
        )}
      </div>
    </main>
  )
}
