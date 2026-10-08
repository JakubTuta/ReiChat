import { describe, expect, it } from 'vitest'
import { formatStats } from '~/utils/formatStats'

describe('formatStats', () => {
  it('returns an empty string when there are no stats', () => {
    expect(formatStats(undefined)).toBe('')
  })

  it('returns an empty string for an empty stats object', () => {
    expect(formatStats({})).toBe('')
  })

  it('formats tokens/sec and token count', () => {
    expect(formatStats({ tokens_per_second: 25.4, completion_tokens: 30 })).toBe('25.4 tok/s · 30 tokens')
  })

  it('only shows the fields that are actually present', () => {
    expect(formatStats({ completion_tokens: 12 })).toBe('12 tokens')
    expect(formatStats({ tokens_per_second: 9.99 })).toBe('10.0 tok/s')
  })
})
