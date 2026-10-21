import { useEffect, useMemo, useRef, useState } from 'react'
import type { Transcript } from '../app/types'
import ChatPanel from './ChatPanel'
import { EXPORT_FORMATS, formatTimestamp, renderExport } from '../lib/export/captions'

type Tab = 'transcript' | 'ask' | 'voice'

interface InformationProps {
  transcript: Transcript
  audioUrl: string
  onVoiceStart: () => void
}

export default function Information({ transcript, audioUrl, onVoiceStart }: InformationProps) {
  const [tab, setTab] = useState<Tab>('ask')
  const [hideFlagged, setHideFlagged] = useState(false)
  const [activeKey, setActiveKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const flaggedCount = useMemo(
    () => transcript.segments.filter((segment) => segment.flagged).length,
    [transcript],
  )

  const visibleSegments = useMemo(
    () => (hideFlagged ? transcript.segments.filter((segment) => !segment.flagged) : transcript.segments),
    [transcript, hideFlagged],
  )

  const displayText = useMemo(
    () => visibleSegments.map((segment) => segment.text).join('\n'),
    [visibleSegments],
  )

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return

    const onTimeUpdate = () => {
      const time = audio.currentTime
      let found: string | null = null
      for (const segment of transcript.segments) {
        if (time < segment.start || time > segment.end) continue
        for (const [wordIndex, word] of segment.words.entries()) {
          if (time >= word.start && time <= word.end) {
            found = `${segment.id}-${wordIndex}`
            break
          }
        }
        if (found) break
      }
      setActiveKey(found)
    }

    audio.addEventListener('timeupdate', onTimeUpdate)
    return () => audio.removeEventListener('timeupdate', onTimeUpdate)
  }, [transcript])

  function seekTo(time: number) {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = Math.max(0, time - 0.05)
    void audio.play().catch(() => {})
  }

  function download(format: (typeof EXPORT_FORMATS)[number]['format']) {
    const content = renderExport(transcript, format)
    const mime = EXPORT_FORMATS.find((entry) => entry.format === format)?.mime ?? 'text/plain'
    const blob = new Blob([content], { type: `${mime};charset=utf-8` })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `xscribe-transcript.${format}`
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  async function copyText() {
    try {
      await navigator.clipboard.writeText(displayText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <main className="flex-1 p-4 flex flex-col gap-4 pb-20 w-full max-w-3xl mx-auto min-h-0">
      <h1 className="font-semibold text-4xl sm:text-5xl">
        <span className="text-blue-400 bold">Done</span>{' '}
        <span className="text-slate-400 text-2xl align-middle">
          {transcript.segments.length} segments · {formatTimestamp(transcript.duration)}
        </span>
      </h1>

      <audio ref={audioRef} src={audioUrl} className="w-full" controls></audio>

      <div className="flex gap-1 border-b border-slate-200">
        {(
          [
            ['transcript', 'Transcript'],
            ['ask', 'Ask'],
            ['voice', 'Voice'],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setTab(key)}
            className={
              'px-4 py-2 text-sm font-medium duration-200 rounded-t-lg ' +
              (tab === key
                ? 'text-blue-500 border-b-2 border-blue-400'
                : 'text-slate-400 hover:text-slate-600')
            }
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'transcript' && (
        <div className="flex flex-col gap-3 min-h-0">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {EXPORT_FORMATS.map((entry) => (
              <button
                key={entry.format}
                type="button"
                onClick={() => download(entry.format)}
                className="specialBtn px-3 py-1 rounded-lg text-blue-400"
              >
                {entry.label}
              </button>
            ))}
            <button
              type="button"
              onClick={() => void copyText()}
              className="specialBtn px-3 py-1 rounded-lg text-blue-400"
            >
              {copied ? 'Copied ✓' : 'Copy'}
            </button>
            {flaggedCount > 0 && (
              <label className="flex items-center gap-1 ml-auto text-slate-500 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hideFlagged}
                  onChange={(event) => setHideFlagged(event.target.checked)}
                />
                Hide {flaggedCount} flagged segment{flaggedCount === 1 ? '' : 's'}
              </label>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-4 overflow-y-auto max-h-[50vh]">
            {visibleSegments.length === 0 && (
              <p className="text-slate-400 italic text-sm">
                Every segment was flagged — uncheck the box above to review them.
              </p>
            )}
            {visibleSegments.map((segment) => (
              <div
                key={segment.id}
                className={'text-sm ' + (segment.flagged ? 'opacity-60' : '')}
              >
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={() => seekTo(segment.start)}
                    className="text-slate-400 font-mono whitespace-nowrap hover:text-blue-500 duration-200"
                    title="Seek to segment"
                  >
                    [{formatTimestamp(segment.start)}]
                  </button>
                  <p className="flex-1 leading-relaxed">
                    {segment.words.map((word, wordIndex) => (
                      <span
                        key={wordIndex}
                        onClick={() => seekTo(word.start)}
                        className={
                          'cursor-pointer duration-100 rounded px-0.5 ' +
                          (activeKey === `${segment.id}-${wordIndex}`
                            ? 'bg-blue-100 text-blue-700'
                            : 'hover:bg-blue-50') +
                          (segment.flagged && hideFlagged ? ' line-through' : '')
                        }
                        title={formatTimestamp(word.start)}
                      >
                        {word.text}
                      </span>
                    ))}
                  </p>
                </div>
                {segment.flagged && (
                  <p className="text-xs text-amber-600 mt-1 pl-16">
                    <i className="fa-solid fa-triangle-exclamation mr-1"></i>
                    Likely unreliable ({segment.flagReasons.join(', ')})
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'ask' && <ChatPanel transcript={transcript} />}

      {tab === 'voice' && (
        <div className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-6 text-center">
          <p className="text-slate-500 text-sm">
            Talk out loud about this recording. XScribe answers with real speech via OpenAI&apos;s
            realtime voice model, using your transcript as context.
          </p>
          <p className="text-xs text-slate-400">
            Microphone audio streams directly to the model over WebRTC — nothing is recorded or
            stored.
          </p>
          <button
            type="button"
            onClick={onVoiceStart}
            className="specialBtn px-4 py-2 rounded-lg text-blue-400 mx-auto flex items-center gap-2 font-medium"
          >
            <p>Start voice chat</p>
            <i className="fa-solid fa-headset"></i>
          </button>
        </div>
      )}
    </main>
  )
}
