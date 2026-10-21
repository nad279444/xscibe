import type { Device } from '../app/types'

interface HeaderProps {
  device: Device | null
  deviceWarning: string | null
  canStartNew: boolean
  onNew: () => void
}

export default function Header({ device, deviceWarning, canStartNew, onNew }: HeaderProps) {
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
        {device && (
          <span
            className={
              'text-xs px-2 py-1 rounded-full border ' +
              (device === 'webgpu'
                ? 'border-cyan-300/30 text-cyan-200 bg-cyan-300/[0.08]'
                : 'border-white/10 text-slate-400 bg-white/[0.04]')
            }
            title={deviceWarning ?? (device === 'webgpu' ? 'WebGPU adapter detected; model initialization will confirm GPU support.' : 'Running models on WebAssembly.')}
          >
            {device === 'webgpu' ? 'WebGPU' : deviceWarning ? 'WASM · GPU unavailable' : 'WASM'}
          </span>
        )}
        {deviceWarning && (
          <span className="max-w-[min(55vw,420px)] text-right text-[10px] leading-4 text-amber-200/80" title={deviceWarning}>
            {deviceWarning}
          </span>
        )}
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
