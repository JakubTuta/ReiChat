import { describe, expect, it } from 'vitest'
import { FIT_LABEL, gpuShareLabel, statusLabel, statusMeta } from '~/utils/modelStatus'

describe('statusMeta', () => {
  it('gives every container state a plain-language label', () => {
    expect(statusMeta('running').label).toBe('ready to chat')
    expect(statusMeta('exited').label).toBe('container stopped')
    expect(statusMeta('incomplete').label).toBe('download didn\'t finish')
  })

  it('falls back to "not installed" for a state it has never heard of', () => {
    expect(statusMeta('something-new')).toEqual(statusMeta('not_found'))
  })

  it('marks transitional states amber so they read as "in progress"', () => {
    for (const status of ['restarting', 'pulling_model', 'starting', 'incomplete'])
      expect(statusMeta(status).dot).toBe('amber')
  })
})

describe('statusLabel', () => {
  it('does not tell people an embedding model is ready to chat', () => {
    expect(statusLabel('running', true)).toBe('ready · used for searching your files')
    expect(statusLabel('running', false)).toBe('ready to chat')
  })

  it('only changes the wording for a running embedding model', () => {
    expect(statusLabel('exited', true)).toBe('container stopped')
  })
})

describe('gpuShareLabel', () => {
  const GB = 1024 ** 3

  it('says CPU when the host has no GPU, whatever VRAM is reported', () => {
    expect(gpuShareLabel({ size_bytes: 4 * GB, vram_bytes: 4 * GB }, false)).toBe('on CPU')
  })

  it('says CPU when nothing is in VRAM or the size is unknown', () => {
    expect(gpuShareLabel({ size_bytes: 4 * GB, vram_bytes: 0 }, true)).toBe('on CPU')
    expect(gpuShareLabel({ size_bytes: 0, vram_bytes: 0 }, true)).toBe('on CPU')
  })

  it('says fully on GPU once at least 99% fits', () => {
    expect(gpuShareLabel({ size_bytes: 100, vram_bytes: 99 }, true)).toBe('fully on GPU')
  })

  it('reports a split as a percentage', () => {
    expect(gpuShareLabel({ size_bytes: 100, vram_bytes: 60 }, true)).toBe('60% on GPU')
  })

  it('still reports a share when GPU support has not been detected yet', () => {
    expect(gpuShareLabel({ size_bytes: 100, vram_bytes: 60 }, null)).toBe('60% on GPU')
  })
})

describe('fit labels', () => {
  it('has wording for every fit result', () => {
    expect(Object.keys(FIT_LABEL).sort()).toEqual(['runs_well', 'tight', 'unknown', 'wont_fit'])
  })
})
