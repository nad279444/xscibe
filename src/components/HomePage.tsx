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
    <main className="flex-1 p-4 flex flex-col gap-3 text-center sm:gap-4 justify-center pb-20">
      <h1 className="font-semibold text-5xl sm:text-6xl md:text-7xl">
        X<span className="text-blue-400 bold">scribe</span>
      </h1>
      <h3 className="font-medium md:text-lg">
        Record <span className="text-blue-400">&rarr;</span> Transcribe{' '}
        <span className="text-blue-400">&rarr;</span> Ask &amp; Speak
      </h3>
      <button
        type="button"
        onClick={recordingStatus === 'recording' ? stopRecording : startRecording}
        className="flex specialBtn px-4 py-2 rounded-xl items-center text-base justify-between gap-4 mx-auto w-72 max-w-full my-4"
      >
        <p className="text-blue-400">
          {recordingStatus === 'inactive' ? 'Record' : `Recording ${duration}s`}
        </p>
        <div className="flex items-center gap-2">
          <i
            className={
              'fa-solid duration-200 fa-microphone ' +
              (recordingStatus === 'recording' ? 'text-rose-300' : '')
            }
          ></i>
        </div>
      </button>
      {recordingError && <p className="text-rose-500 text-sm">{recordingError}</p>}
      <p className="text-base">
        Or{' '}
        <label className="text-blue-400 cursor-pointer hover:text-blue-600 duration-200">
          upload{' '}
          <input
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) onSelectFile(file)
              event.target.value = ''
            }}
            className="hidden"
            type="file"
            accept=".mp3,.wav,.m4a,.webm,.ogg,.flac,audio/*"
          />
        </label>{' '}
        an audio file
      </p>
      <button
        type="button"
        onClick={onStartLive}
        className="mx-auto flex items-center gap-2 text-blue-400 hover:text-blue-600 duration-200"
        title="Transcribe your microphone live, phrase by phrase"
      >
        <i className="fa-solid fa-bolt"></i>
        <span>or transcribe live from your microphone</span>
      </button>
      <p className="italic text-slate-400">Runs entirely in your browser</p>
    </main>
  )
}
