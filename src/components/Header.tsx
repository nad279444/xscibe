interface HeaderProps {
  canStartNew: boolean
  onNew: () => void
}

export default function Header({ canStartNew, onNew }: HeaderProps) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-white/[0.08] bg-[#07111f]/90 px-5 py-4 sm:px-8">
      <button
        type="button"
        onClick={onNew}
        className="flex items-center gap-2 text-left"
        title="Back to start"
      >
        <h1 className="font-medium tracking-wide text-slate-100">
          X<span className="text-cyan-300 bold">Scribe</span>
        </h1>
      </button>
      <div className="flex flex-wrap items-center justify-end gap-3">
        {canStartNew && (
          <button
            type="button"
            onClick={onNew}
            className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.05] px-3 py-2 text-cyan-200 transition hover:border-cyan-300/40"
          >
            <p>New</p>
            <i className="fa-solid fa-plus"></i>
          </button>
        )}
      </div>
    </header>
  )
}
