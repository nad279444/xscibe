import { useEffect, useRef, useState } from 'react'

interface HomePageProps {
  onSelectFile: (file: File) => void
  onSelectRecording: (blob: Blob) => void
  onStartLive: () => void
}

type RecordingStatus = 'inactive' | 'recording'

const MIME_TYPE = 'audio/webm'

export default function HomePage({
  onSelectFile,
  onSelectRecording,
  onStartLive,
}: HomePageProps) {
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>('inactive')
  const [duration, setDuration] = useState(0)
  const [recordingError, setRecordingError] = useState<string | null>(null)

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const mediaStream = useRef<MediaStream | null>(null)
  const chunks = useRef<Blob[]>([])

  async function startRecording() {
    setRecordingError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      mediaStream.current = stream
      chunks.current = []
      let media: MediaRecorder
      try {
        media = new MediaRecorder(stream, { mimeType: MIME_TYPE })
      } catch {
        media = new MediaRecorder(stream)
      }
      mediaRecorder.current = media
      media.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunks.current.push(event.data)
      }
      media.onstop = () => {
        const blob = new Blob(chunks.current, { type: media.mimeType || MIME_TYPE })
        chunks.current = []
        setRecordingStatus('inactive')
        setDuration(0)
        if (blob.size > 0) onSelectRecording(blob)
      }
      media.start()
      setRecordingStatus('recording')
    } catch (error) {
      setRecordingError(
        error instanceof Error ? error.message : 'Microphone permission was denied',
      )
      setRecordingStatus('inactive')
    }
  }

  function stopRecording() {
    mediaRecorder.current?.stop()
    for (const track of mediaStream.current?.getTracks() ?? []) track.stop()
    mediaStream.current = null
  }

  useEffect(() => {
    if (recordingStatus !== 'recording') return
    const interval = setInterval(() => setDuration((current) => current + 1), 1000)
    return () => clearInterval(interval)
  }, [recordingStatus])

  useEffect(() => {
    return () => {
      if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
        mediaRecorder.current.stop()
      }
      for (const track of mediaStream.current?.getTracks() ?? []) track.stop()
    }
  }, [])

  return (
    <main className="relative isolate flex-1 overflow-hidden px-5 py-10 text-slate-800 sm:px-8 sm:py-14">
      <div className="pointer-events-none absolute inset-0 -z-10 robot-grid opacity-40" />
      <div className="mx-auto grid min-h-[min(720px,calc(100vh-150px))] max-w-6xl items-center gap-12 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white/70 px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-blue-700">
            <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500 shadow-[0_0_12px_#60a5fa]" />
            Audio intelligence online
          </div>
          <div className="space-y-4">
            <h1 className="max-w-2xl text-5xl font-semibold tracking-tight sm:text-6xl lg:text-7xl">
              Meet your <span className="text-blue-600">audio</span> co-pilot.
            </h1>
            <p className="max-w-xl text-base leading-7 text-slate-600 sm:text-lg">
              Turn recordings into searchable transcripts, clear answers, and natural voice conversations.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="group flex cursor-pointer items-center gap-4 rounded-2xl border border-sky-200 bg-white/75 p-5 shadow-sm transition hover:border-blue-300 hover:bg-white">
              <span className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-cyan-300 text-slate-950 shadow-[0_0_24px_rgba(103,232,249,0.18)]">
                <i className="fa-solid fa-file-arrow-up text-lg" />
              </span>
              <span>
                <span className="block font-semibold text-slate-900">Transcribe a file</span>
                <span className="mt-1 block text-xs text-slate-600">Cloud transcription · up to 4 MB</span>
              </span>
              <input
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) onSelectFile(file)
                  event.target.value = ''
                }}
                className="sr-only"
                type="file"
                accept=".mp3,.mp4,.mpeg,.mpga,.m4a,.wav,.webm,.ogg,.flac,audio/*"
              />
            </label>

            <button
              type="button"
              onClick={recordingStatus === 'recording' ? stopRecording : startRecording}
              className="flex items-center gap-4 rounded-2xl border border-sky-200 bg-white/75 p-5 text-left shadow-sm transition hover:border-rose-300 hover:bg-white"
            >
              <span className="grid h-12 w-12 shrink-0 place-items-center rounded-xl border border-white/10 bg-slate-900 text-rose-300">
                <i className={'fa-solid ' + (recordingStatus === 'recording' ? 'fa-stop' : 'fa-microphone')} />
              </span>
              <span>
                <span className="block font-semibold text-slate-900">
                  {recordingStatus === 'recording' ? `Recording · ${duration}s` : 'Record audio'}
                </span>
                <span className="mt-1 block text-xs text-slate-600">Capture from your microphone</span>
              </span>
            </button>
          </div>
          {recordingError && <p className="text-sm text-rose-300">{recordingError}</p>}

          <button
            type="button"
            onClick={onStartLive}
            className="flex w-full items-center gap-3 rounded-xl border border-sky-200 bg-white/75 px-4 py-3 text-left shadow-sm transition hover:border-blue-300 hover:bg-white"
          >
            <i className="fa-solid fa-wave-square text-cyan-300" />
            <span className="text-sm font-medium text-slate-800">Start live transcription</span>
            <span className="ml-auto font-mono text-[10px] uppercase tracking-widest text-slate-500">Live mode</span>
          </button>
          <p className="text-xs leading-relaxed text-slate-600">
            Audio files and live speech are sent to our transcription service. Do not upload
            recordings unless you have permission to share them.
          </p>

          <div className="grid gap-3 border-t border-sky-200 pt-6 sm:grid-cols-3">
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-blue-600">01 / Decode</p>
              <p className="mt-1 text-xs text-slate-600">Searchable transcripts with seek links</p>
            </div>
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-blue-600">02 / Ask</p>
              <p className="mt-1 text-xs text-slate-600">Chat with your recording</p>
            </div>
            <div>
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-blue-600">03 / Speak</p>
              <p className="mt-1 text-xs text-slate-600">Continue by voice</p>
            </div>
          </div>
        </section>

        <aside className="relative mx-auto w-full max-w-md">
          <div className="absolute inset-8 rounded-full bg-cyan-400/10 blur-3xl" />
          <div className="relative overflow-hidden rounded-[2rem] border border-cyan-200/15 bg-[#0b1828]/90 p-6 shadow-[0_24px_90px_rgba(0,0,0,0.35)] sm:p-8">
            <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
              <span>XS / Unit 01</span>
              <span className="text-emerald-300">● Ready</span>
            </div>
            <div className="robot-face mx-auto my-9">
              <div className="robot-antenna" />
              <div className="robot-ear robot-ear-left" />
              <div className="robot-ear robot-ear-right" />
              <div className="robot-eyes"><span /><span /></div>
              <div className="robot-mouth" />
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono uppercase tracking-widest text-slate-400">Core systems</span>
                <span className="text-cyan-300">3 / 3 active</span>
              </div>
              {[
                ['AUDIO TRANSCRIPTION', 'Cloud powered'],
                ['TRANSCRIPT CHAT', 'Context aware'],
                ['VOICE LINK', 'On demand'],
              ].map(([name, status]) => (
                <div key={name} className="flex items-center justify-between border-t border-white/[0.07] pt-3">
                  <span className="font-mono text-[10px] tracking-wider text-slate-400">{name}</span>
                  <span className="flex items-center gap-2 text-xs text-slate-200"><span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />{status}</span>
                </div>
              ))}
            </div>
            <p className="mt-6 rounded-xl bg-black/20 px-4 py-3 text-xs leading-5 text-slate-400">
              Start with any recording. Your co-pilot will transcribe it, then help you explore what was said.
            </p>
          </div>
        </aside>
      </div>
    </main>
  )
}
