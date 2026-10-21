import type { VoiceCaption, VoiceStatus } from '../app/types'

interface VoiceViewProps {
  status: VoiceStatus
  captions: VoiceCaption[]
  onStop: () => void
}

export default function VoiceView({ status, captions, onStop }: VoiceViewProps) {
  const connecting = status === 'connecting'
  return (
    <main className="flex-1 p-4 flex flex-col gap-4 text-center pb-20 w-full max-w-prose mx-auto">
      <div className="flex items-center justify-between gap-4">
        <h1 className="font-semibold text-4xl sm:text-5xl">
          <span className="text-blue-400 bold">Voice Chat</span>
        </h1>
        <span
          className={
            'h-3 w-3 rounded-full ' +
            (connecting ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400 animate-pulse')
          }
        ></span>
      </div>
      {connecting ? (
        <p className="text-slate-500 italic">Connecting to the voice model…</p>
      ) : (
        <div className="text-left flex-1 overflow-y-auto rounded-xl border border-slate-200 p-4 space-y-4 bg-white min-h-48">
          {captions.length === 0 && (
            <p className="text-slate-400 italic">
              Ask about your recording — speak naturally, XScribe will answer out loud.
            </p>
          )}
          {captions.map((caption, index) => (
            <div
              key={index}
              className={
                'flex ' + (caption.role === 'user' ? 'justify-end' : 'justify-start')
              }
            >
              <div
                className={
                  'max-w-[85%] rounded-xl px-3 py-2 text-sm whitespace-pre-wrap ' +
                  (caption.role === 'user'
                    ? 'bg-blue-50 text-blue-700'
                    : 'bg-slate-100 text-slate-700')
                }
              >
                {caption.text || '…'}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center justify-between gap-4 text-xs text-slate-400">
        <p>
          <i className="fa-solid fa-microphone mr-1"></i> Microphone is live. No audio is stored.
        </p>
        <button
          type="button"
          onClick={onStop}
          className="specialBtn px-4 py-2 rounded-lg text-blue-400 text-sm flex items-center gap-2"
        >
          <p>End conversation</p>
          <i className="fa-solid fa-phone-slash"></i>
        </button>
      </div>
    </main>
  )
}
