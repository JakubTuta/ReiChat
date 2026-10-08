import type { IAIModel } from '~/models/aiModel'
import { describe, expect, it } from 'vitest'
import { recommendModels } from '~/utils/recommend'

const GB = 1024 ** 3

function model(name: string, versions: [string, number][]): IAIModel {
  return {
    id: name,
    name,
    model: name,
    description: '',
    popularity: 0,
    can_process_image: false,
    is_embedding: false,
    can_use_tools: false,
    is_custom: false,
    index: 0,
    versions: versions.map(([parameters, gb]) => ({ parameters, size: '', size_bytes: gb * GB, verified_at: null })),
  }
}

const catalog = [
  model('qwen3', [['0.6b', 0.5], ['4b', 2.5], ['8b', 5.2], ['14b', 9.3]]),
  model('gemma3', [['1b', 0.8], ['4b', 3.3], ['12b', 8.1]]),
  model('llama3.2', [['1b', 1.3], ['3b', 2.0]]),
]

describe('recommendModels', () => {
  it('picks a bigger model for a bigger GPU', () => {
    const { best } = recommendModels(catalog, { ram_bytes: 32 * GB, vram_bytes: 24 * GB, gpu_available: true })

    expect(`${best?.model.model}:${best?.version.parameters}`).toBe('qwen3:14b')
  })

  it('falls back to what fits in RAM without a GPU', () => {
    const { best, lighter } = recommendModels(catalog, { ram_bytes: 8 * GB, vram_bytes: 8 * GB, gpu_available: false })

    expect(`${best?.model.model}:${best?.version.parameters}`).toBe('gemma3:4b')
    expect(lighter).not.toBeNull()
    expect(lighter!.version.size_bytes!).toBeLessThan(best!.version.size_bytes!)
  })

  it('offers the lightest models when hardware is unknown', () => {
    const { best } = recommendModels(catalog, { ram_bytes: null, vram_bytes: null })

    expect(['gemma3:1b', 'llama3.2:1b', 'qwen3:0.6b']).toContain(`${best?.model.model}:${best?.version.parameters}`)
  })
})
