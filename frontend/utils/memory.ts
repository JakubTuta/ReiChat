import type { ContextReport } from '~/constants/websocket'

// Mirrors MEMORY_PRESETS in backend/app/chat/context_window.py. Sizes are tokens;
// everything the user sees is in "pages" (~500 words ≈ 650 tokens).
export interface MemoryPreset {
  id: string
  label: string
  tokens: number
  description: string
}

export const TOKENS_PER_PAGE = 650

export const MEMORY_PRESETS: MemoryPreset[] = [
  { id: 'short', label: 'Short', tokens: 4096, description: 'Fastest and lightest. Fine for quick questions.' },
  { id: 'standard', label: 'Standard', tokens: 8192, description: 'A good balance for most conversations.' },
  { id: 'long', label: 'Long', tokens: 16384, description: 'For long chats and documents. Uses more memory.' },
  { id: 'very_long', label: 'Very long', tokens: 32768, description: 'Needs a lot of memory and replies start slower.' },
]

export const DEFAULT_PRESET_TOKENS = 8192

export function pagesFor(tokens: number): number {
  return Math.max(1, Math.round(tokens / TOKENS_PER_PAGE))
}

export function formatPages(tokens: number): string {
  if (tokens < TOKENS_PER_PAGE / 2)
    return 'less than a page'

  const pages = pagesFor(tokens)

  return `about ${pages} page${pages === 1
    ? ''
    : 's'}`
}

export type MemoryLevel = 'ok' | 'filling' | 'full'

export function memoryPercent(report: Pick<ContextReport, 'used' | 'limit'>): number {
  if (!report.limit)
    return 0

  return Math.min(100, Math.round((report.used / report.limit) * 100))
}

export function memoryLevel(report: Pick<ContextReport, 'used' | 'limit' | 'dropped'>): MemoryLevel {
  if (report.dropped > 0)
    return 'full'

  const percent = memoryPercent(report)
  if (percent >= 90)
    return 'full'
  if (percent >= 70)
    return 'filling'

  return 'ok'
}

// The next bigger preset the model can actually use, or null if the chat is
// already at the largest one available.
export function nextPreset(currentLimit: number, modelMax: number | null): MemoryPreset | null {
  return MEMORY_PRESETS.find(p => p.tokens > currentLimit && (modelMax === null || p.tokens <= modelMax)) ?? null
}

// The next smaller preset, for when a big memory pushes part of the model
// out of GPU memory and slows every reply down.
export function previousPreset(currentLimit: number): MemoryPreset | null {
  return [...MEMORY_PRESETS].reverse().find(p => p.tokens < currentLimit) ?? null
}

// Below this, enough of the model runs on the CPU that it's worth saying so.
export const GPU_SPILL_THRESHOLD = 0.95

export function spillsOutOfGpu(report: { gpu_share?: number | null }): boolean {
  return report.gpu_share !== undefined && report.gpu_share !== null && report.gpu_share < GPU_SPILL_THRESHOLD
}

export function availablePresets(modelMax: number | null): MemoryPreset[] {
  return MEMORY_PRESETS.filter(p => modelMax === null || p.tokens <= modelMax)
}
