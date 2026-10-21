import type { DeviceDetection } from '../../app/types'

interface GpuNavigator {
  gpu?: {
    requestAdapter(): Promise<{
      features: { has(feature: string): boolean }
    } | null>
  }
}

export async function detectDevice(): Promise<DeviceDetection> {
  const nav = navigator as unknown as GpuNavigator
  if (!isSecureContext) {
    return { device: 'wasm', warning: 'WebGPU requires a secure context (HTTPS or localhost).' }
  }
  if (!nav.gpu) {
    return { device: 'wasm', warning: 'This browser does not expose WebGPU. Try a current version of Chrome or Edge.' }
  }
  try {
    const adapter = await nav.gpu.requestAdapter()
    if (!adapter) {
      return { device: 'wasm', warning: 'The browser found no available WebGPU adapter. Check browser settings and GPU drivers.' }
    }
    return { device: 'webgpu' }
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    return { device: 'wasm', warning: `WebGPU adapter request failed: ${reason}` }
  }
}
