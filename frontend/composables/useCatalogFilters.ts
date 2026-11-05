import type { ShellType } from '~/utils/dockerCommands'
import { isShellType } from '~/utils/dockerCommands'
import { nextSort } from '~/utils/modelCatalog'

// The Models page's search, sort and filter state, mirrored into the URL so
// a filtered view survives a reload and can be shared.
export function useCatalogFilters() {
  const router = useRouter()
  const query = router.currentRoute.value.query

  const search = ref(typeof query.search === 'string'
    ? query.search
    : '')
  const sort = ref(typeof query.sort === 'string'
    ? query.sort
    : 'popularityDecreasing')
  const filter = ref(typeof query.filter === 'string'
    ? query.filter
    : 'all')
  const canProcessImages = ref(query.canProcessImages === 'true')

  // Default to the shell the user is most likely in: PowerShell on Windows.
  const shellType = ref<ShellType>(isShellType(query.shell)
    ? query.shell
    : /Windows/i.test(navigator.userAgent)
      ? 'powershell'
      : 'bash')

  function updateQuery(patch: Record<string, string | undefined>, mode: 'push' | 'replace' = 'push') {
    router[mode]({ query: { ...router.currentRoute.value.query, ...patch } })
  }

  // `push`ing per keystroke used to add one history entry per character
  // typed — Back needed as many presses as the search term was long.
  // `replace` updates the current entry instead, and the debounce keeps
  // fast typing from spamming the router at all.
  const syncSearchToQuery = useDebounceFn((value: string) => {
    updateQuery({ search: value || undefined }, 'replace')
  }, 400)

  function cycleSort() {
    sort.value = nextSort(sort.value)
    updateQuery({ sort: sort.value })
  }

  function setFilter(value: string) {
    filter.value = value
    updateQuery({ filter: value })
  }

  function setShell(value: ShellType) {
    shellType.value = value
    updateQuery({ shell: value })
  }

  function toggleCanProcessImages() {
    canProcessImages.value = !canProcessImages.value
    updateQuery({ canProcessImages: canProcessImages.value
      ? 'true'
      : undefined })
  }

  return {
    search,
    sort,
    filter,
    canProcessImages,
    shellType,
    syncSearchToQuery,
    cycleSort,
    setFilter,
    setShell,
    toggleCanProcessImages,
  }
}
