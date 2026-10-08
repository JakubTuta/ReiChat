import type { GenerationStats } from '~/constants/websocket'

export function formatStats(stats?: GenerationStats): string {
  if (!stats)
    return ''

  const parts: string[] = []

  if (stats.tokens_per_second !== undefined)
    parts.push(`${stats.tokens_per_second.toFixed(1)} tok/s`)

  if (stats.completion_tokens !== undefined)
    parts.push(`${stats.completion_tokens} tokens`)

  return parts.join(' · ')
}
