export interface IHardwareInfo {
  ram_bytes: number | null
  vram_bytes: number | null
  // null until a container start has shown whether GPUs work on this host.
  gpu_available?: boolean | null
}

export interface IDiskUsageEntry {
  model: string
  parameters: string
  status: string
  size_bytes: number | null
}

export interface IDiskUsage {
  total_bytes: number
  models: IDiskUsageEntry[]
}

export type FitLabel = 'runs_well' | 'tight' | 'wont_fit' | 'unknown'

export interface FitResult {
  label: FitLabel
  availableBytes: number | null
  source: 'vram' | 'ram' | 'none'
}

// Approximate on purpose — Ollama's actual runtime memory use includes KV
// cache overhead on top of the model's disk size, so these ratios lean
// conservative rather than claiming precision the inputs don't have.
const RUNS_WELL_RATIO = 0.6
const TIGHT_RATIO = 1.0

export function computeFitLabel(sizeBytes: number | null, hardware: IHardwareInfo): FitResult {
  if (sizeBytes === null)
    return { label: 'unknown', availableBytes: null, source: 'none' }

  // Models run on the CPU (system RAM) when the host has no usable GPU,
  // however much VRAM nvidia-smi might report.
  const vramBytes = hardware.gpu_available === false
    ? null
    : hardware.vram_bytes
  const availableBytes = vramBytes ?? hardware.ram_bytes
  const source: FitResult['source'] = vramBytes !== null
    ? 'vram'
    : hardware.ram_bytes !== null
      ? 'ram'
      : 'none'

  if (availableBytes === null)
    return { label: 'unknown', availableBytes: null, source: 'none' }

  const ratio = sizeBytes / availableBytes

  if (ratio <= RUNS_WELL_RATIO)
    return { label: 'runs_well', availableBytes, source }
  if (ratio <= TIGHT_RATIO)
    return { label: 'tight', availableBytes, source }

  return { label: 'wont_fit', availableBytes, source }
}
