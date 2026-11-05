import type { IAIModel, IAIModelVersion } from '~/models/aiModel'

export const STATE_PILLS = [
  { label: 'All', value: 'all' },
  { label: 'Installed', value: 'allMyModels' },
  { label: 'Ready to chat', value: 'runningContainers' },
  { label: 'Stopped', value: 'stoppedContainers' },
  { label: 'Not installed', value: 'notFoundContainers' },
]

const SORT_CYCLE = [
  { label: 'Most popular', value: 'popularityDecreasing' },
  { label: 'Least popular', value: 'popularityIncreasing' },
  { label: 'Name A→Z', value: 'nameAlphabetically' },
  { label: 'Name Z→A', value: 'nameReverseAlphabetically' },
]

export function sortLabel(sort: string): string {
  return SORT_CYCLE.find(s => s.value === sort)?.label ?? 'Sort'
}

export function nextSort(sort: string): string {
  const idx = SORT_CYCLE.findIndex(s => s.value === sort)

  return SORT_CYCLE[(idx + 1) % SORT_CYCLE.length].value
}

export interface CatalogQuery {
  search: string
  sort: string
  filter: string
  canProcessImages: boolean
}

// Container status of one downloadable size of a model; 'not_found' when
// nothing has been created for it yet.
export type VersionStatus = (model: IAIModel, version: IAIModelVersion) => string

function matchesState(filter: string, status: string): boolean {
  if (filter === 'allMyModels')
    return status !== 'not_found'
  if (filter === 'runningContainers')
    return status === 'running'
  if (filter === 'stoppedContainers')
    return ['exited', 'paused', 'restarting'].includes(status)
  if (filter === 'notFoundContainers')
    return status === 'not_found'

  return true
}

function compareBySort(sort: string): (a: IAIModel, b: IAIModel) => number {
  return (a, b) => {
    if (sort === 'popularityDecreasing')
      return b.popularity - a.popularity
    if (sort === 'popularityIncreasing')
      return a.popularity - b.popularity
    if (sort === 'nameAlphabetically')
      return a.name.localeCompare(b.name)
    if (sort === 'nameReverseAlphabetically')
      return b.name.localeCompare(a.name)

    return 0
  }
}

export function installedVersionCount(model: IAIModel, statusOf: VersionStatus): number {
  return model.versions.filter(v => statusOf(model, v) !== 'not_found').length
}

// What the model grid shows: the catalogue narrowed by search, image
// support and container state (a model keeps only the sizes that match),
// sorted, with models the user already has on top.
export function selectCatalogModels(models: IAIModel[], query: CatalogQuery, statusOf: VersionStatus): IAIModel[] {
  const needle = query.search.toLowerCase()

  const withVersions = models
    .filter(model => !query.search || model.name.toLowerCase().includes(needle))
    .filter(model => (query.canProcessImages
      ? model.can_process_image
      : true))
    .sort(compareBySort(query.sort))
    .map(model => ({
      ...model,
      versions: model.versions.filter(version => matchesState(query.filter, statusOf(model, version))),
    }))
    .filter(model => model.versions.length > 0)

  const installed = withVersions.filter(m => installedVersionCount(m, statusOf) > 0)
  const rest = withVersions.filter(m => installedVersionCount(m, statusOf) === 0)

  return [...installed, ...rest]
}

export function formatCount(num: number): string {
  if (num >= 1_000_000)
    return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000)
    return `${(num / 1_000).toFixed(1)}K`

  return num.toString()
}
