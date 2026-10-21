import type { Device } from '../../app/types'

interface GpuNavigator {
  gpu?: {
    requestAdapter(): Promise<{
      features: { has(feature: string): boolean }
    } | null>
  }
}

export async function detectDevice(): Promise<Device> {
  const nav = navigator as unknown as GpuNavigator
  if (!nav.gpu) return 'wasm'
  try {
    const adapter = await nav.gpu.requestAdapter()
    if (!adapter) return 'wasm'
    return 'webgpu'
  } catch {
    return 'wasm'
  }
}
