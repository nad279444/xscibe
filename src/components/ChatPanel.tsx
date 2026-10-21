import { useRef, useState } from 'react'
import type { ChatTurn, Transcript } from '../app/types'
import { streamChat, type ChatMessage } from '../lib/openai/client'

interface ChatPanelProps {
  transcript: Transcript
}

const SUGGESTIONS = [
  'Summarise this recording',
  'What action items came up?',
  'Who speaks the most?',
]

let nextTurnId = 0

export default function ChatPanel({ transcript }: ChatPanelProps) {
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [question, setQuestion] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  async function ask(text: string) {
    const trimmed = text.trim()
    if (!trimmed || streaming) return
    setQuestion('')
    setError(null)
    setStreaming(true)

    const userId = nextTurnId++
    const assistantId = nextTurnId++
    setTurns((current) => [
      ...current,
      { id: userId, role: 'user', text: trimmed },
      { id: assistantId, role: 'assistant', text: '' },
    ])

    const history: ChatMessage[] = turns
      .filter((turn) => turn.text)
      .map((turn) => ({ role: turn.role, text: turn.text }))

    try {
      for await (const delta of streamChat({ transcript, question: trimmed, history })) {
        setTurns((current) =>
          current.map((turn) =>
            turn.id === assistantId ? { ...turn, text: turn.text + delta } : turn,
          ),
        )
        const scroller = scrollRef.current
        if (scroller) scroller.scrollTop = scroller.scrollHeight
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      setTurns((current) =>
        current.map((turn) =>
          turn.id === assistantId && !turn.text
            ? { ...turn, text: '(no answer — please try again)' }
            : turn,
        ),
      )
    } finally {
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto rounded-xl border border-slate-200 p-3 space-y-3 bg-white min-h-40"
      >
        {turns.length === 0 && (
          <div className="flex flex-col gap-2 items-start">
            <p className="text-slate-400 italic text-sm">
              Ask anything about this transcript — answers stream back instantly.
            </p>
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => void ask(suggestion)}
                className="text-xs border border-blue-200 text-blue-500 rounded-full px-3 py-1 hover:bg-blue-50 duration-200"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn) => (
          <div key={turn.id} className={'flex ' + (turn.role === 'user' ? 'justify-end' : '')}>
            <div
              className={
                'max-w-[85%] rounded-xl px-3 py-2 text-sm whitespace-pre-wrap ' +
                (turn.role === 'user'
                  ? 'bg-blue-50 text-blue-700'
                  : 'bg-slate-100 text-slate-700')
              }
            >
              {turn.text || (streaming ? '…' : '')}
            </div>
          </div>
        ))}
      </div>
      {error && <p className="text-rose-500 text-xs">{error}</p>}
      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          void ask(question)
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about your recording…"
          className="flex-1 border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-300 duration-200 bg-white"
          disabled={streaming}
        />
        <button
          type="submit"
          disabled={streaming || !question.trim()}
          className="specialBtn px-4 py-2 rounded-lg text-blue-400 text-sm disabled:opacity-50"
        >
          <i className="fa-solid fa-paper-plane"></i>
        </button>
      </form>
    </div>
  )
}
