interface HeaderProps {
  canStartNew: boolean
  onNew: () => void
}

export default function Header({ canStartNew, onNew }: HeaderProps) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-sky-200/70 bg-white/65 px-5 py-4 shadow-sm backdrop-blur sm:px-8">
      <button
        type="button"
        onClick={onNew}
        className="flex items-center gap-2 text-left"
        title="Back to start"
      >
        <h1 className="font-medium tracking-wide text-slate-800">
          X<span className="text-blue-600 bold">Scribe</span>
        </h1>
      </button>
      <div className="flex flex-wrap items-center justify-end gap-3">
        {canStartNew && (
          <button
            type="button"
            onClick={onNew}
            className="flex items-center gap-2 rounded-lg border border-sky-200 bg-white/75 px-3 py-2 text-blue-700 transition hover:border-blue-300"
          >
            <p>New</p>
            <i className="fa-solid fa-plus"></i>
          </button>
        )}
      </div>
    </header>
  )
}
