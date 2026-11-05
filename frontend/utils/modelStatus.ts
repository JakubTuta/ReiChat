import type { FitLabel } from '~/models/hardware'

interface StatusMeta {
  // Name of a --color-* token in assets/css/main.css.
  dot: string
  label: string
}

const STATUS_META: Record<string, StatusMeta> = {
  running: { dot: 'mint', label: 'ready to chat' },
  exited: { dot: 'grey-dot', label: 'container stopped' },
  paused: { dot: 'grey-dot', label: 'container stopped' },
  restarting: { dot: 'amber', label: 'restarting' },
  pulling_model: { dot: 'amber', label: 'downloading' },
  starting: { dot: 'amber', label: 'starting up' },
  incomplete: { dot: 'amber', label: 'download didn\'t finish' },
  not_found: { dot: 'grey-dot', label: 'not installed' },
}

export function statusMeta(status: string): StatusMeta {
  return STATUS_META[status] ?? STATUS_META.not_found
}

// Embedding models never chat — "ready to chat" would send people looking
// for them in the chat picker.
export function statusLabel(status: string, isEmbedding: boolean): string {
  if (isEmbedding && status === 'running')
    return 'ready · used for searching your files'

  return statusMeta(status).label
}

export const FIT_LABEL: Record<FitLabel, string> = {
  runs_well: 'runs well',
  tight: 'tight',
  wont_fit: 'won\'t fit',
  unknown: 'unknown',
}

export function gpuShareLabel(
  loaded: { size_bytes: number, vram_bytes: number },
  gpuAvailable: boolean | null | undefined,
): string {
  if (!loaded.size_bytes || gpuAvailable === false)
    return 'on CPU'

  const percent = Math.round((loaded.vram_bytes / loaded.size_bytes) * 100)
  if (percent >= 99)
    return 'fully on GPU'
  if (percent <= 0)
    return 'on CPU'

  return `${percent}% on GPU`
}
