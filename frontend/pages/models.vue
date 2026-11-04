<script setup lang="ts">
import type { IAIModel, IAIModelVersion } from '~/models/aiModel'
import type { FitLabel } from '~/models/hardware'
import { computeFitLabel } from '~/models/hardware'
import { OLLAMA_UPDATE_KEY } from '~/stores/containerStore'
import { containerNameFor } from '~/utils/containerName'

const router = useRouter()

const containerStore = useContainerStore()
const { containers, loadingOperation, pullProgress, scrapeProgress, loadedModels, isExternal, connection } = storeToRefs(containerStore)
const connectionDialogOpen = ref(false)

const chatStore = useChatStore()
const { aiModels, loading } = storeToRefs(chatStore)

const hardwareStore = useHardwareStore()
const { hardware, diskUsage } = storeToRefs(hardwareStore)

const snackbarStore = useSnackbarStore()

const isOpenPullModelsDialog = ref(false)
const addModelDialogOpen = ref(false)

const search = ref('')
function installAddedModel(added: { model: string, parameters: string }) {
  containerStore.runContainer(added)
  search.value = added.model.split('/').pop() ?? added.model
  snackbarStore.showSnackbarInfo('Downloading — progress shows on the model\'s card.')
}
const sort = ref('popularityDecreasing')
const filter = ref('all')
const selectedVersions = ref<Record<string, IAIModelVersion | null>>({})

const ollamaUpdateProgress = computed(() => pullProgress.value[OLLAMA_UPDATE_KEY] ?? null)

let runtimeTimer: ReturnType<typeof setInterval> | null = null
const RUNTIME_POLL_MS = 10_000

function loadedInfo(model: IAIModel) {
  const version = selectedVersions.value[model.model]
  if (!version)
    return null

  return loadedModels.value.find(m => m.model === `${model.model}:${version.parameters}`) ?? null
}

function gpuShareLabel(loaded: { size_bytes: number, vram_bytes: number }): string {
  if (!loaded.size_bytes || hardware.value.gpu_available === false)
    return 'on CPU'

  const percent = Math.round((loaded.vram_bytes / loaded.size_bytes) * 100)
  if (percent >= 99)
    return 'fully on GPU'
  if (percent <= 0)
    return 'on CPU'

  return `${percent}% on GPU`
}
// cmd.exe uses `^` for line continuation, but PowerShell — the default
// shell on Windows 11 — uses a backtick, and `^` just fails there. A plain
// "Windows commands" checkbox defaulting to true assumed cmd unconditionally
// and broke for anyone on the actual Win11 default.
type ShellType = 'powershell' | 'cmd' | 'bash'
const shellItems: { title: string, value: ShellType }[] = [
  { title: 'bash/zsh', value: 'bash' },
  { title: 'PowerShell', value: 'powershell' },
  { title: 'Command Prompt', value: 'cmd' },
]
const shellType = ref<ShellType>('bash')
const canProcessImages = ref(false)
const escapeCharacter = computed(() => {
  if (shellType.value === 'cmd')
    return '^'
  if (shellType.value === 'powershell')
    return '`'

  return '\\'
})

onMounted(async () => {
  const query = router.currentRoute.value.query
  if (query.search)
    search.value = query.search as string
  if (query.sort)
    sort.value = query.sort as string
  if (query.filter)
    filter.value = query.filter as string
  if (query.canProcessImages)
    canProcessImages.value = query.canProcessImages === 'true'

  if (query.shell === 'powershell' || query.shell === 'cmd' || query.shell === 'bash') {
    shellType.value = query.shell
  }
  else {
    shellType.value = /Windows/i.test(navigator.userAgent)
      ? 'powershell'
      : 'bash'
  }

  // Instant paint before socket connects
  await containerStore.getUserContainers()
  containerStore.connectSocket()

  await chatStore.fetchAIModels()

  hardwareStore.fetchHardware()
  hardwareStore.fetchDiskUsage()

  containerStore.fetchConnection()
  containerStore.fetchRuntime()
  runtimeTimer = setInterval(() => containerStore.fetchRuntime(), RUNTIME_POLL_MS)
})

onUnmounted(() => {
  containerStore.disconnectSocket()
  containerStore.stopPolling()
  if (runtimeTimer !== null)
    clearInterval(runtimeTimer)
})

watch(
  () => scrapeProgress.value.running,
  (running, wasRunning) => {
    if (wasRunning && !running && !scrapeProgress.value.error) {
      chatStore.fetchAIModels()
      snackbarStore.showSnackbarSuccess('Models updated successfully!')
    }
  },
)

const preparedAIModels = computed(() => {
  const searchFn = (model: IAIModel) => !search.value || model.name.toLowerCase().includes(search.value.toLowerCase())

  const sortFn = (a: IAIModel, b: IAIModel) => {
    if (sort.value === 'popularityDecreasing')
      return b.popularity - a.popularity
    if (sort.value === 'popularityIncreasing')
      return a.popularity - b.popularity
    if (sort.value === 'nameAlphabetically')
      return a.name.localeCompare(b.name)
    if (sort.value === 'nameReverseAlphabetically')
      return b.name.localeCompare(a.name)

    return 0
  }

  const filterFn = (containerStatus: string) => {
    if (filter.value === 'all')
      return true
    if (filter.value === 'allMyModels')
      return containerStatus !== 'not_found'
    if (filter.value === 'runningContainers')
      return containerStatus === 'running'
    if (filter.value === 'stoppedContainers')
      return ['exited', 'paused', 'restarting'].includes(containerStatus)
    if (filter.value === 'notFoundContainers')
      return containerStatus === 'not_found'

    return true
  }

  const canProcessImageFn = (model: IAIModel) => (canProcessImages.value
    ? model.can_process_image
    : true)

  const withVersions = aiModels.value
    .filter(searchFn)
    .filter(canProcessImageFn)
    .sort(sortFn)
    .map((model) => {
      const filteredVersions = model.versions.filter((version) => {
        const containerName = containerNameFor(model.model, version.parameters)
        const containerStatus = getContainerStatusByName(containerName)

        return filterFn(containerStatus)
      })

      return { ...model, versions: filteredVersions }
    })
    .filter(model => model.versions.length > 0)

  const installed = withVersions.filter(m => getInstalledVersionCount(m) > 0)
  const rest = withVersions.filter(m => getInstalledVersionCount(m) === 0)

  return [...installed, ...rest]
})

const statePills = [
  { label: 'All', value: 'all' },
  { label: 'Installed', value: 'allMyModels' },
  { label: 'Ready to chat', value: 'runningContainers' },
  { label: 'Stopped', value: 'stoppedContainers' },
  { label: 'Not installed', value: 'notFoundContainers' },
]

const sortCycle = [
  { label: 'Most popular', value: 'popularityDecreasing' },
  { label: 'Least popular', value: 'popularityIncreasing' },
  { label: 'Name A→Z', value: 'nameAlphabetically' },
  { label: 'Name Z→A', value: 'nameReverseAlphabetically' },
]

const currentSortLabel = computed(() => sortCycle.find(s => s.value === sort.value)?.label ?? 'Sort')

function cycleSort() {
  const idx = sortCycle.findIndex(s => s.value === sort.value)
  const next = sortCycle[(idx + 1) % sortCycle.length]
  sort.value = next.value
  addSortToQuery(next.value)
}

function getContainerName(model: IAIModel): string | null {
  if (!selectedVersions.value[model.model])
    return null

  return containerNameFor(model.model, selectedVersions.value[model.model]!.parameters)
}

function getContainerStatusByName(containerName: string): string {
  return containers.value.find(c => c.name === containerName)?.status ?? 'not_found'
}

function getContainerStatus(model: IAIModel): string {
  const name = getContainerName(model)
  if (!name)
    return 'not_found'

  return getContainerStatusByName(name)
}

function getModelPullProgress(model: IAIModel) {
  const name = getContainerName(model)
  if (!name)
    return null

  return pullProgress.value[name] ?? null
}

function isOperationLoading(model: IAIModel): boolean {
  const name = getContainerName(model)
  if (!name || !loadingOperation.value)
    return false

  // operationKey is `${action}_${model}_${parameters}` — anchor on the
  // full container name, not a bare substring of the model name, or e.g.
  // "llama3" would read as loading whenever "llama3.1_8b" was.
  return loadingOperation.value.endsWith(`_${name}`)
}

function getInstalledVersionCount(model: IAIModel): number {
  return model.versions.filter(v => getContainerStatusByName(containerNameFor(model.model, v.parameters)) !== 'not_found',
  ).length
}

function selectVersion(model: IAIModel, version: IAIModelVersion) {
  selectedVersions.value[model.model] = selectedVersions.value[model.model]?.parameters === version.parameters
    ? null
    : version
}

// A returning user with an installed model saw "Select a version to manage
// this model" on a model they'd already set up — auto-pick the first
// installed version instead of leaving the card looking unconfigured.
watch([aiModels, containers], () => {
  for (const model of aiModels.value) {
    if (selectedVersions.value[model.model])
      continue

    const installedVersion = model.versions.find(
      v => getContainerStatusByName(containerNameFor(model.model, v.parameters)) !== 'not_found',
    )

    if (installedVersion)
      selectedVersions.value[model.model] = installedVersion
  }
}, { immediate: true, deep: true })

// No chat model installed yet (embedding-only doesn't count): the
// first-run "Get started" card shows instead of making a newcomer pick from
// the whole catalog.
const hasChatModelInstalled = computed(() => containers.value.some((c) => {
  const catalogEntry = aiModels.value.find(m => m.model === c.environment?.model)

  return catalogEntry && !catalogEntry.is_embedding
}))

const STATUS_META: Record<string, { dot: string, label: string }> = {
  running: { dot: 'mint', label: 'ready to chat' },
  exited: { dot: 'grey-dot', label: 'container stopped' },
  paused: { dot: 'grey-dot', label: 'container stopped' },
  restarting: { dot: 'amber', label: 'restarting' },
  pulling_model: { dot: 'amber', label: 'downloading' },
  starting: { dot: 'amber', label: 'starting up' },
  incomplete: { dot: 'amber', label: 'download didn\'t finish' },
  not_found: { dot: 'grey-dot', label: 'not installed' },
}

function statusMeta(status: string) {
  return STATUS_META[status] ?? STATUS_META.not_found
}

// Embedding models never chat — "ready to chat" would send people looking
// for them in the chat picker.
function statusLabel(model: IAIModel): string {
  const status = getContainerStatus(model)
  if (model.is_embedding && status === 'running')
    return 'ready · used for searching your files'

  return statusMeta(status).label
}

const FIT_LABEL: Record<FitLabel, string> = {
  runs_well: 'runs well',
  tight: 'tight',
  wont_fit: 'won\'t fit',
  unknown: 'unknown',
}

function fitFor(model: IAIModel) {
  const version = selectedVersions.value[model.model]
  if (!version)
    return null

  return computeFitLabel(version.size_bytes, hardware.value)
}

interface ActionConfig {
  label: string
  primary: boolean
  disabled: boolean
  loading: boolean
  handler: () => void
}

function getPrimaryAction(model: IAIModel): ActionConfig {
  if (!selectedVersions.value[model.model]) {
    return { label: 'Select a version first', primary: false, disabled: true, loading: false, handler: () => {} }
  }

  const containerStatus = getContainerStatus(model)
  const progress = getModelPullProgress(model)
  const operationLoading = isOperationLoading(model)

  if (progress) {
    return { label: 'Downloading…', primary: false, disabled: true, loading: true, handler: () => {} }
  }

  if (operationLoading || containerStatus === 'restarting') {
    return { label: 'Please wait…', primary: false, disabled: true, loading: true, handler: () => {} }
  }

  if (containerStatus === 'not_found') {
    return {
      label: 'Create container',
      primary: true,
      disabled: false,
      loading: false,
      handler: () => createContainer(model),
    }
  }

  if (containerStatus === 'exited' || containerStatus === 'paused') {
    return {
      label: 'Start container',
      primary: false,
      disabled: false,
      loading: false,
      handler: () => startContainer(model),
    }
  }

  if (containerStatus === 'incomplete') {
    return {
      label: 'Resume download',
      primary: true,
      disabled: false,
      loading: false,
      handler: () => resumeDownload(model),
    }
  }

  if (containerStatus === 'starting') {
    return { label: 'Starting…', primary: false, disabled: true, loading: true, handler: () => {} }
  }

  if (containerStatus === 'running' && isExternal.value)
    return { label: 'Installed', primary: false, disabled: true, loading: false, handler: () => {} }

  if (containerStatus === 'running') {
    return {
      label: 'Stop container',
      primary: false,
      disabled: false,
      loading: false,
      handler: () => stopContainer(model),
    }
  }

  if (containerStatus === 'pulling_model') {
    return { label: 'Pulling model…', primary: false, disabled: true, loading: true, handler: () => {} }
  }

  return { label: 'N/A', primary: false, disabled: true, loading: false, handler: () => {} }
}

function canRemove(model: IAIModel): boolean {
  const s = getContainerStatus(model)

  return ['running', 'exited', 'paused', 'restarting', 'pulling_model', 'starting', 'incomplete'].includes(s)
}

function createContainer(model: IAIModel) {
  if (!selectedVersions.value[model.model])
    return
  containerStore.runContainer({ model: model.model, parameters: selectedVersions.value[model.model]!.parameters })
  snackbarStore.showSnackbarInfo('Container creation started — progress shown on card')
}

function startContainer(model: IAIModel) {
  if (!selectedVersions.value[model.model])
    return
  containerStore.runContainer({ model: model.model, parameters: selectedVersions.value[model.model]!.parameters })
  snackbarStore.showSnackbarSuccess('Starting container')
}

function resumeDownload(model: IAIModel) {
  if (!selectedVersions.value[model.model])
    return
  containerStore.runContainer({ model: model.model, parameters: selectedVersions.value[model.model]!.parameters })
  snackbarStore.showSnackbarInfo('Resuming download — it continues from where it stopped')
}

function stopContainer(model: IAIModel) {
  if (!selectedVersions.value[model.model])
    return
  containerStore.stopContainer({ model: model.model, parameters: selectedVersions.value[model.model]!.parameters })
  snackbarStore.showSnackbarSuccess('Stopping container')
}

// Deleting a chat has a confirmation; deleting a multi-GB model didn't.
const confirmRemoveDialog = ref(false)
const modelToRemove = ref<IAIModel | null>(null)

const removeConfirmMessage = computed(() => {
  const model = modelToRemove.value
  const version = model
    ? selectedVersions.value[model.model]
    : null

  if (!model || !version)
    return ''

  return isExternal.value
    ? `This deletes "${model.name}" (${version.parameters}) from your Ollama and frees ~${version.size} of disk space. This cannot be undone.`
    : `This deletes the "${model.name}" (${version.parameters}) container and frees ~${version.size} of disk space. This cannot be undone.`
})

function removeContainer(model: IAIModel) {
  if (!selectedVersions.value[model.model])
    return
  modelToRemove.value = model
  confirmRemoveDialog.value = true
}

async function confirmRemoveContainer() {
  confirmRemoveDialog.value = false

  const model = modelToRemove.value
  modelToRemove.value = null

  if (!model || !selectedVersions.value[model.model])
    return

  await containerStore.removeContainer({ model: model.model, parameters: selectedVersions.value[model.model]!.parameters })
  snackbarStore.showSnackbarSuccess('Removing container')
  hardwareStore.fetchDiskUsage()
}

function cancelRemoveContainer() {
  confirmRemoveDialog.value = false
  modelToRemove.value = null
}

// The backend allocates ports from a collision-free table, not a formula —
// `11434 + model.index` (the old scheme) stopped meaning anything once that
// changed, and index gets renumbered on every catalog refresh anyway. Show
// the container's real published port if it already exists; otherwise this
// is just a starting point for a manual `docker run` — the app doesn't
// reserve it.
function getContainerPort(model: IAIModel): number {
  const name = getContainerName(model)
  const existing = name
    ? containers.value.find(c => c.name === name)
    : null

  if (existing?.port)
    return Number.parseInt(existing.port, 10)

  return 11434
}

function createContainerCommand(model: IAIModel): string[] {
  if (!selectedVersions.value[model.model])
    return []
  const parameters = selectedVersions.value[model.model]!.parameters
  const containerName = containerNameFor(model.model, parameters)
  const containerPort = getContainerPort(model)

  const gpuFlag = hardware.value.gpu_available === false
    ? ''
    : `  --gpus=all ${escapeCharacter.value}\n`
  const createCommand = `docker run -d ${escapeCharacter.value}
  --name ${containerName} ${escapeCharacter.value}
  --network chatbot-network ${escapeCharacter.value}
${gpuFlag}  -p 127.0.0.1:${containerPort}:11434 ${escapeCharacter.value}
  -v reichat-ollama-${containerName}:/root/.ollama ${escapeCharacter.value}
  --label reichat.managed=true ${escapeCharacter.value}
  --label reichat.model=${model.model} ${escapeCharacter.value}
  --label reichat.parameters=${parameters} ${escapeCharacter.value}
  --label reichat.port=${containerPort} ${escapeCharacter.value}
  ollama/ollama:latest`

  const pullModelCommand = `docker exec -d ${containerName} ollama pull ${model.model}:${parameters}`

  return [createCommand, pullModelCommand]
}

function startContainerCommand(model: IAIModel): string[] {
  if (!selectedVersions.value[model.model])
    return []
  const parameters = selectedVersions.value[model.model]!.parameters
  const containerName = containerNameFor(model.model, parameters)

  return [`docker start ${containerName}`]
}

function stopContainerCommand(model: IAIModel): string[] {
  if (!selectedVersions.value[model.model])
    return []
  const parameters = selectedVersions.value[model.model]!.parameters
  const containerName = containerNameFor(model.model, parameters)

  return [`docker stop ${containerName}`]
}

function removeContainerCommand(model: IAIModel): string[] {
  if (!selectedVersions.value[model.model])
    return []
  const containerStatus = getContainerStatus(model)
  const stopCmds = ['running', 'pulling_model', 'starting', 'incomplete'].includes(containerStatus)
    ? stopContainerCommand(model)
    : []
  const containerName = containerNameFor(model.model, selectedVersions.value[model.model]!.parameters)

  return [...stopCmds, `docker rm ${containerName}`, `docker volume rm reichat-ollama-${containerName}`]
}

function allCliCommands(model: IAIModel): string[] {
  if (!selectedVersions.value[model.model])
    return []
  const containerStatus = getContainerStatus(model)
  if (containerStatus === 'not_found')
    return createContainerCommand(model)
  const cmds: string[] = []
  if (['exited', 'paused'].includes(containerStatus))
    cmds.push(...startContainerCommand(model))
  if (['running', 'paused', 'restarting', 'pulling_model', 'starting', 'incomplete'].includes(containerStatus))
    cmds.push(...stopContainerCommand(model))
  cmds.push(...removeContainerCommand(model))

  return cmds
}

function copyToClipboard(text: string): void {
  navigator.clipboard.writeText(text)
}

function formatNumber(num: number): string {
  if (num >= 1_000_000)
    return `${(num / 1_000_000).toFixed(1)}M`
  if (num >= 1_000)
    return `${(num / 1_000).toFixed(1)}K`

  return num.toString()
}

// `push`ing per keystroke used to add one history entry per character
// typed — Back needed as many presses as the search term was long.
// `replace` updates the current entry instead, and the debounce keeps
// fast typing from spamming the router at all.
const addSearchToQuery = useDebounceFn((value: string) => {
  router.replace({ query: { ...router.currentRoute.value.query, search: value || undefined } })
}, 400)

function addSortToQuery(value: string) {
  router.push({ query: { ...router.currentRoute.value.query, sort: value } })
}

function addFilterToQuery(value: string) {
  filter.value = value
  router.push({ query: { ...router.currentRoute.value.query, filter: value } })
}

function addShellToQuery(value: ShellType) {
  shellType.value = value
  router.push({ query: { ...router.currentRoute.value.query, shell: value } })
}

function toggleCanProcessImages() {
  canProcessImages.value = !canProcessImages.value
  router.push({ query: { ...router.currentRoute.value.query, canProcessImages: canProcessImages.value
    ? 'true'
    : undefined } })
}

function openPullModelsDialog() {
  if (loading.value)
    return
  isOpenPullModelsDialog.value = true
}

const catalogueButtonLabel = computed(() => {
  if (scrapeProgress.value.running) {
    return `Refreshing… ${scrapeProgress.value.total
      ? `${scrapeProgress.value.completed}/${scrapeProgress.value.total}`
      : ''}`
  }
  if (loading.value)
    return 'Loading…'

  return aiModels.value.length
    ? 'Refresh model list'
    : 'Fetch model list'
})
</script>

<template>
  <AppTopBar />

  <div class="models-page">
    <SystemStatusBanner />

    <div class="models-header">
      <div>
        <h1 class="page-h1">
          Models
        </h1>

        <div
          v-if="diskUsage.models.length"
          class="mono-kicker models-subhead"
        >
          {{ diskUsage.models.length }} model{{ diskUsage.models.length === 1
            ? ''
            : 's' }} installed · {{ formatBytes(diskUsage.total_bytes) }}
        </div>
      </div>

      <div class="models-header-actions">
        <v-btn
          variant="text"
          rounded="lg"
          prepend-icon="mdi-server"
          title="Change where models run"
          @click="connectionDialogOpen = true"
        >
          {{ isExternal
            ? 'Your Ollama'
            : 'Managed by ReiChat' }}
        </v-btn>

        <v-btn
          variant="outlined"
          rounded="lg"
          @click="addModelDialogOpen = true"
        >
          Add by name
        </v-btn>

        <v-btn
          v-if="!isExternal"
          variant="outlined"
          rounded="lg"
          :loading="!!ollamaUpdateProgress"
          title="Download the newest Ollama and move your models onto it"
          @click="containerStore.updateOllama()"
        >
          Update Ollama
          <template #loader>
            {{ ollamaUpdateProgress?.detail || 'Updating…' }}
          </template>
        </v-btn>

        <v-btn
          color="mint-btn"
          variant="flat"
          rounded="lg"
          @click="openPullModelsDialog"
        >
          {{ catalogueButtonLabel }}
        </v-btn>
      </div>
    </div>

    <GetStartedCard
      v-if="!loading && aiModels.length && !hasChatModelInstalled"
      class="models-get-started"
    />

    <div
      v-if="isExternal"
      class="cpu-hint"
    >
      <v-icon size="16">
        mdi-server
      </v-icon>

      <span>
        Using your own Ollama at <strong>{{ connection.url }}</strong>. Models you install here are downloaded
        there, and everything already on it is listed.
      </span>
    </div>

    <div
      v-if="hardware.gpu_available === false && !isExternal"
      class="cpu-hint"
    >
      <v-icon size="16">
        mdi-chip
      </v-icon>

      <span>
        No compatible graphics card (NVIDIA GPU) was found, so models run on your processor (CPU).
        Everything works — answers just arrive more slowly. Smaller models (1–4B) feel best here.
      </span>
    </div>

    <div class="filter-row">
      <div class="search-field">
        <v-icon size="16">
          mdi-magnify
        </v-icon>

        <input
          v-model="search"
          type="text"
          placeholder="Search models…"
          aria-label="Search models"
          @input="addSearchToQuery(search)"
        >
      </div>

      <button
        v-for="pill in statePills"
        :key="pill.value"
        type="button"
        class="filter-pill"
        :class="{'filter-pill--active': filter === pill.value}"
        @click="addFilterToQuery(pill.value)"
      >
        {{ pill.label }}
      </button>

      <span class="filter-row-spacer" />

      <button
        type="button"
        class="filter-pill"
        :class="{'filter-pill--active': canProcessImages}"
        @click="toggleCanProcessImages"
      >
        Can process images
      </button>

      <button
        type="button"
        class="filter-pill font-mono"
        @click="cycleSort"
      >
        {{ currentSortLabel }}
      </button>
    </div>

    <div
      v-if="!loading && !aiModels.length"
      class="page-empty"
    >
      <div class="page-empty-headline">
        No models loaded yet
      </div>

      <div class="page-empty-body">
        Click <strong>Fetch model list</strong>
        to load the catalogue from
        <a
          href="https://ollama.com/library"
          target="_blank"
          rel="noopener noreferrer"
        >ollama.com</a>

        . Then pick a version on a model card and click <strong>Create container</strong>
        to download and run it.
      </div>

      <v-btn
        color="mint-btn"
        variant="flat"
        @click="openPullModelsDialog"
      >
        Fetch model list
      </v-btn>
    </div>

    <div
      v-else-if="!loading && !preparedAIModels.length"
      class="page-empty"
    >
      <div class="page-empty-headline">
        Nothing matches those filters
      </div>
    </div>

    <div class="models-grid">
      <div
        v-for="model in preparedAIModels"
        :key="model.model"
        class="model-card"
      >
        <div class="model-card-title-row">
          <span class="model-card-name">{{ model.name }}</span>

          <span
            v-if="model.can_process_image"
            class="badge badge--mint"
          >VISION</span>

          <span
            v-if="getInstalledVersionCount(model)"
            class="badge badge--grey"
          >{{ getInstalledVersionCount(model) }} INSTALLED</span>

          <span
            v-if="model.is_custom"
            class="badge badge--grey"
          >ADDED BY YOU</span>

          <span class="model-card-popularity font-mono">★ {{ formatNumber(model.popularity) }}</span>
        </div>

        <p class="model-card-description">
          {{ model.description }}
        </p>

        <div class="version-pills">
          <button
            v-for="version in model.versions"
            :key="version.parameters"
            type="button"
            class="version-pill font-mono"
            :class="{'version-pill--active': selectedVersions[model.model]?.parameters === version.parameters}"
            @click="selectVersion(model, version)"
          >
            {{ version.parameters }} · {{ version.size }}
          </button>
        </div>

        <template v-if="selectedVersions[model.model]">
          <div class="status-line">
            <span
              class="status-dot"
              :style="{'backgroundColor': `var(--color-${statusMeta(getContainerStatus(model)).dot})`}"
            />

            <span class="status-label">{{ statusLabel(model) }}</span>

            <template v-if="fitFor(model) && fitFor(model)!.label !== 'unknown'">
              <span class="status-sep">·</span>

              <span class="fit-label">
                {{ FIT_LABEL[fitFor(model)!.label] }}

                <v-tooltip
                  activator="parent"
                  location="top"
                >
                  Approximate — based on this model's download size vs.
                  {{ fitFor(model)!.source === 'vram'
                    ? 'GPU memory'
                    : 'system RAM' }}
                  ({{ formatBytes(fitFor(model)!.availableBytes) }} available).
                </v-tooltip>
              </span>
            </template>
          </div>

          <div
            v-if="loadedInfo(model)"
            class="loaded-line"
          >
            <span>
              In memory: {{ formatBytes(loadedInfo(model)!.size_bytes) }} · {{ gpuShareLabel(loadedInfo(model)!) }}
            </span>

            <button
              type="button"
              class="loaded-free"
              title="Unload it now instead of waiting a few idle minutes"
              @click="containerStore.unloadModel({'model': model.model,
                                                  'parameters': selectedVersions[model.model]!.parameters})"
            >
              Free memory
            </button>
          </div>

          <template v-if="getModelPullProgress(model)">
            <div class="pull-progress-row">
              <span>{{ getModelPullProgress(model)!.phase === 'image_pull_progress'
                ? 'Downloading base image'
                : 'Downloading model' }}</span>

              <span class="font-mono">{{ getModelPullProgress(model)!.percent }}%</span>
            </div>

            <div class="pull-progress-track">
              <div
                class="pull-progress-fill"
                :style="{'width': `${getModelPullProgress(model)!.percent}%`}"
              />
            </div>

            <div class="pull-progress-detail">
              {{ getModelPullProgress(model)!.detail }}
            </div>
          </template>

          <div class="action-row">
            <v-btn
              :color="getPrimaryAction(model).primary
                ? 'mint-btn'
                : undefined"
              :variant="getPrimaryAction(model).primary
                ? 'flat'
                : 'outlined'"
              :disabled="getPrimaryAction(model).disabled"
              :loading="getPrimaryAction(model).loading"
              size="small"
              rounded="lg"
              style="min-width: 150px"
              @click="getPrimaryAction(model).handler()"
            >
              {{ getPrimaryAction(model).label }}
            </v-btn>

            <v-btn
              v-if="getContainerStatus(model) === 'running' && !getModelPullProgress(model)"
              variant="text"
              size="small"
              title="Download a newer version of this model if there is one"
              @click="containerStore.updateModel({'model': model.model,
                                                  'parameters': selectedVersions[model.model]!.parameters})"
            >
              Check for update
            </v-btn>

            <v-btn
              v-if="canRemove(model)"
              variant="text"
              color="red"
              size="small"
              :disabled="isOperationLoading(model)"
              @click="removeContainer(model)"
            >
              Remove
            </v-btn>

            <span class="action-row-spacer" />

            <v-expansion-panels
              v-if="!isExternal && allCliCommands(model).length"
              class="advanced-panel"
              variant="accordion"
            >
              <v-expansion-panel>
                <v-expansion-panel-title class="advanced-toggle font-mono">
                  advanced: run manually
                </v-expansion-panel-title>

                <v-expansion-panel-text>
                  <div class="shell-pills">
                    <button
                      v-for="shell in shellItems"
                      :key="shell.value"
                      type="button"
                      class="shell-pill font-mono"
                      :class="{'shell-pill--active': shellType === shell.value}"
                      @click="addShellToQuery(shell.value)"
                    >
                      {{ shell.title }}
                    </button>
                  </div>

                  <div
                    v-for="(cmd, idx) in allCliCommands(model)"
                    :key="idx"
                    class="cli-command"
                  >
                    <pre class="font-mono">{{ cmd }}</pre>

                    <button
                      type="button"
                      class="cli-copy"
                      title="Copy"
                      @click="copyToClipboard(cmd)"
                    >
                      <v-icon size="14">
                        mdi-content-copy
                      </v-icon>
                    </button>
                  </div>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
          </div>
        </template>

        <div
          v-else
          class="no-version-hint"
        >
          Select a version to manage this model.
        </div>
      </div>
    </div>
  </div>

  <PullModelsDialog v-model:is-show="isOpenPullModelsDialog" />

  <ConfirmDialog
    v-model="confirmRemoveDialog"
    title="Remove model"
    :message="removeConfirmMessage"
    confirm-label="Remove"
    confirm-color="red"
    @confirm="confirmRemoveContainer"
    @cancel="cancelRemoveContainer"
  />

  <OllamaConnectionDialog v-model="connectionDialogOpen" />

  <AddModelDialog
    v-model="addModelDialogOpen"
    @added="installAddedModel"
  />
</template>

<style scoped>
.models-page {
  max-width: 1120px;
  margin: 0 auto;
  padding: 28px 20px 60px;
}

.models-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 34px;
  flex-wrap: wrap;
}

.page-h1 {
  font-size: 30px;
  font-weight: 400;
  letter-spacing: -0.025em;
  color: var(--color-ink);
  margin: 0 0 6px;
}

.models-subhead {
  color: var(--color-ink-3);
}

.filter-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 22px;
}

.models-get-started {
  margin-bottom: 22px;
  max-width: none;
}

.models-header-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.loaded-line {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -4px 0 10px;
  font-size: 12px;
  color: var(--color-mint-deep);
}

.loaded-free {
  border: none;
  background: transparent;
  padding: 0;
  font-size: 12px;
  color: var(--color-ink-3);
  text-decoration: underline;
  cursor: pointer;
}

.cpu-hint {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 14px;
  margin-bottom: 18px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-size: 13px;
  line-height: 1.5;
}

.search-field {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 210px;
  padding: 7px 11px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-3);
}

.search-field input {
  flex: 1;
  min-width: 0;
  border: none;
  outline: none;
  background: transparent;
  font: inherit;
  font-size: 13px;
  color: var(--color-ink);
}

.filter-row-spacer {
  flex: 1;
}

.filter-pill {
  padding: 7px 13px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.filter-pill:hover {
  border-color: var(--color-mint-border);
}

.filter-pill--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.page-empty {
  padding: 60px 16px;
  text-align: center;
}

.page-empty-headline {
  font-size: 20px;
  color: var(--color-ink);
  margin-bottom: 10px;
}

.page-empty-body {
  max-width: 480px;
  margin: 0 auto 20px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--color-ink-2);
}

.page-empty-body a {
  color: var(--color-mint-deep);
}

.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(390px, 1fr));
  gap: 20px;
}

.model-card {
  background: var(--color-card);
  border: 1px solid var(--color-line-2);
  border-radius: 18px;
  padding: 24px 24px 20px;
  display: flex;
  flex-direction: column;
}

.model-card-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}

.model-card-name {
  font-size: 14.5px;
  font-weight: 600;
  color: var(--color-ink);
}

.badge {
  font-family: var(--font-mono);
  font-size: 9.5px;
  padding: 2px 6px;
  border-radius: 5px;
}

.badge--mint {
  background: var(--color-mint-tint);
  color: var(--color-mint-deep);
}

.badge--grey {
  background: var(--color-soft-2);
  color: var(--color-ink-2);
}

.model-card-popularity {
  margin-left: auto;
  font-size: 10.5px;
  color: var(--color-ink-3);
}

.model-card-description {
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
  margin: 0 0 14px;
}

.version-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.version-pill {
  padding: 5px 10px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  color: var(--color-ink-2);
  font-size: 11px;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.version-pill--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.status-line {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 12px;
  color: var(--color-ink-2);
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-sep {
  color: var(--color-ink-3);
}

.fit-label {
  text-decoration: underline dotted var(--color-ink-3);
  text-underline-offset: 3px;
  cursor: help;
}

.pull-progress-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--color-ink-2);
  margin-bottom: 4px;
}

.pull-progress-track {
  height: 4px;
  border-radius: 4px;
  background: var(--color-soft-2);
  overflow: hidden;
  margin-bottom: 4px;
}

.pull-progress-fill {
  height: 100%;
  background: var(--color-mint);
  transition: width 0.2s ease;
}

.pull-progress-detail {
  font-size: 11px;
  color: var(--color-ink-3);
  margin-bottom: 12px;
}

.action-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.action-row-spacer {
  flex: 1;
}

.no-version-hint {
  font-size: 12.5px;
  color: var(--color-ink-3);
}

.advanced-panel {
  width: auto;
  flex-basis: 100%;
}

.advanced-toggle {
  font-size: 10.5px;
  color: var(--color-ink-3);
  min-height: 0 !important;
  padding: 8px !important;
}

.shell-pills {
  display: flex;
  gap: 6px;
  margin-bottom: 10px;
}

.shell-pill {
  padding: 4px 9px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-size: 10.5px;
  cursor: pointer;
}

.shell-pill--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.cli-command {
  position: relative;
  background: var(--color-soft);
  border: 1px solid var(--color-line-2);
  border-radius: 10px;
  padding: 10px 34px 10px 10px;
  margin-bottom: 8px;
}

.cli-command pre {
  margin: 0;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--color-ink);
}

.cli-copy {
  position: absolute;
  top: 8px;
  right: 8px;
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  cursor: pointer;
}

.cli-copy:hover {
  color: var(--color-mint-deep);
}

@media (max-width: 900px) {
  .models-grid {
    grid-template-columns: 1fr;
  }
}
</style>
