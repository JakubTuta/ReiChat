import { describe, expect, it } from 'vitest'
import { availablePresets, formatPages, memoryLevel, memoryPercent, nextPreset, previousPreset, spillsOutOfGpu } from '~/utils/memory'

describe('memory helpers', () => {
  it('describes sizes in pages, never tokens', () => {
    expect(formatPages(8192)).toBe('about 13 pages')
    expect(formatPages(500)).toBe('about 1 page')
    expect(formatPages(100)).toBe('less than a page')
  })

  it('computes a capped percentage', () => {
    expect(memoryPercent({ used: 2048, limit: 8192 })).toBe(25)
    expect(memoryPercent({ used: 9000, limit: 8192 })).toBe(100)
    expect(memoryPercent({ used: 10, limit: 0 })).toBe(0)
  })

  it('treats any dropped message as full, regardless of percentage', () => {
    expect(memoryLevel({ used: 100, limit: 8192, dropped: 0 })).toBe('ok')
    expect(memoryLevel({ used: 6000, limit: 8192, dropped: 0 })).toBe('filling')
    expect(memoryLevel({ used: 7500, limit: 8192, dropped: 0 })).toBe('full')
    expect(memoryLevel({ used: 100, limit: 8192, dropped: 3 })).toBe('full')
  })

  it('offers the next preset only when the model supports it', () => {
    expect(nextPreset(8192, null)?.id).toBe('long')
    expect(nextPreset(8192, 131072)?.id).toBe('long')
    expect(nextPreset(8192, 8192)).toBeNull()
    expect(nextPreset(32768, null)).toBeNull()
  })

  it('suggests a smaller preset only when one exists', () => {
    expect(previousPreset(16384)?.id).toBe('standard')
    expect(previousPreset(4096)).toBeNull()
  })

  it('flags GPU spill only when a GPU is actually involved', () => {
    expect(spillsOutOfGpu({ gpu_share: 0.6 })).toBe(true)
    expect(spillsOutOfGpu({ gpu_share: 1 })).toBe(false)
    expect(spillsOutOfGpu({ gpu_share: null })).toBe(false)
    expect(spillsOutOfGpu({})).toBe(false)
  })

  it('hides presets bigger than the model can use', () => {
    expect(availablePresets(8192).map(p => p.id)).toEqual(['short', 'standard'])
    expect(availablePresets(null)).toHaveLength(4)
  })
})
