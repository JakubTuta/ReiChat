import type { IAIModel } from '~/models/aiModel'
import { describe, expect, it } from 'vitest'
import { mapAIModel } from '~/models/aiModel'
import { formatCount, installedVersionCount, nextSort, selectCatalogModels, sortLabel } from '~/utils/modelCatalog'

function model(name: string, popularity: number, sizes: string[], extra: Partial<IAIModel> = {}): IAIModel {
  return mapAIModel({
    id: name,
    name,
    model: name,
    popularity,
    versions: sizes.map(parameters => ({ parameters, size: '1 GB', size_bytes: 1, verified_at: null })),
    ...extra,
  })
}

const catalogue = [
  model('llama3.1', 300, ['8b', '70b']),
  model('deepseek-r1', 200, ['1.5b', '7b']),
  model('llava', 100, ['7b'], { can_process_image: true }),
  model('qwen', 50, ['7b']),
]

// containers: deepseek 1.5b running, llama3.1 8b stopped, llava 7b downloading
const statuses: Record<string, string> = {
  'deepseek-r1:1.5b': 'running',
  'llama3.1:8b': 'exited',
  'llava:7b': 'pulling_model',
}
const statusOf = (m: IAIModel, v: { parameters: string }) => statuses[`${m.model}:${v.parameters}`] ?? 'not_found'

const defaults = { search: '', sort: 'popularityDecreasing', filter: 'all', canProcessImages: false }
const names = (models: IAIModel[]) => models.map(m => m.name)

describe('selectCatalogModels', () => {
  it('puts models the user already has first, keeping the chosen order within each group', () => {
    expect(names(selectCatalogModels(catalogue, defaults, statusOf))).toEqual(['llama3.1', 'deepseek-r1', 'llava', 'qwen'])

    const byName = selectCatalogModels(catalogue, { ...defaults, sort: 'nameAlphabetically' }, statusOf)
    expect(names(byName)).toEqual(['deepseek-r1', 'llama3.1', 'llava', 'qwen'])
  })

  it('searches model names case-insensitively', () => {
    expect(names(selectCatalogModels(catalogue, { ...defaults, search: 'LLA' }, statusOf))).toEqual(['llama3.1', 'llava'])
  })

  it('narrows to models that handle images', () => {
    expect(names(selectCatalogModels(catalogue, { ...defaults, canProcessImages: true }, statusOf))).toEqual(['llava'])
  })

  it('keeps only the sizes whose container matches the state filter', () => {
    const running = selectCatalogModels(catalogue, { ...defaults, filter: 'runningContainers' }, statusOf)

    expect(names(running)).toEqual(['deepseek-r1'])
    expect(running[0].versions.map(v => v.parameters)).toEqual(['1.5b'])
  })

  it('treats a stopped container as stopped, and a downloading one as neither stopped nor running', () => {
    expect(names(selectCatalogModels(catalogue, { ...defaults, filter: 'stoppedContainers' }, statusOf))).toEqual(['llama3.1'])
  })

  it('lists only what is not installed under "not installed"', () => {
    const notInstalled = selectCatalogModels(catalogue, { ...defaults, filter: 'notFoundContainers' }, statusOf)

    expect(names(notInstalled)).toEqual(['llama3.1', 'deepseek-r1', 'qwen'])
    expect(notInstalled[0].versions.map(v => v.parameters)).toEqual(['70b'])
  })

  it('does not reorder or modify the catalogue it was given', () => {
    const before = names(catalogue)

    selectCatalogModels(catalogue, { ...defaults, sort: 'nameReverseAlphabetically' }, statusOf)

    expect(names(catalogue)).toEqual(before)
    expect(catalogue[0].versions).toHaveLength(2)
  })

  it('shows everything for a filter value it does not know', () => {
    expect(selectCatalogModels(catalogue, { ...defaults, filter: 'bogus' }, statusOf)).toHaveLength(4)
  })
})

describe('installedVersionCount', () => {
  it('counts sizes that have any container, whatever its state', () => {
    expect(installedVersionCount(catalogue[0], statusOf)).toBe(1)
    expect(installedVersionCount(catalogue[3], statusOf)).toBe(0)
  })
})

describe('sort cycling', () => {
  it('walks through every order and wraps around', () => {
    expect(nextSort('popularityDecreasing')).toBe('popularityIncreasing')
    expect(nextSort('nameReverseAlphabetically')).toBe('popularityDecreasing')
  })

  it('labels each order, with a safe fallback', () => {
    expect(sortLabel('nameAlphabetically')).toBe('Name A→Z')
    expect(sortLabel('nonsense')).toBe('Sort')
  })
})

describe('formatCount', () => {
  it('abbreviates thousands and millions', () => {
    expect(formatCount(999)).toBe('999')
    expect(formatCount(1500)).toBe('1.5K')
    expect(formatCount(118_300_000)).toBe('118.3M')
  })
})
