import type { Device } from '../app/types'

interface HeaderProps {
  device: Device | null
  canStartNew: boolean
  onNew: () => void
}

export default function Header({ device, canStartNew, onNew }: HeaderProps) {
  return (
    <header className="flex items-center justify-between gap-4 p-4">
      <button
        type="button"
        onClick={onNew}
        className="flex items-center gap-2 text-left"
        title="Back to start"
      >
        <h1 className="font-medium">
          X<span className="text-blue-400 bold">Scribe</span>
        </h1>
      </button>
      <div className="gap-3 flex items-center">
        {device && (
          <span
            className={
              'text-xs px-2 py-1 rounded-full border ' +
              (device === 'webgpu'
                ? 'border-blue-300 text-blue-500 bg-blue-50'
                : 'border-slate-300 text-slate-500 bg-white')
            }
            title={
              device === 'webgpu'
                ? 'Running models on your GPU'
                : 'Running models on WebAssembly (no compatible GPU found)'
            }
          >
            {device === 'webgpu' ? 'WebGPU' : 'WASM'}
          </span>
        )}
        {canStartNew && (
          <button
            type="button"
            onClick={onNew}
            className="flex items-center gap-2 specialBtn px-3 py-2 rounded-lg text-blue-400"
          >
            <p>New</p>
            <i className="fa-solid fa-plus"></i>
          </button>
        )}
      </div>
    </header>
  )
}
