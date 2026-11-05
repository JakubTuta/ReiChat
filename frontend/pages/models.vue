<script setup lang="ts">
import type { IAIModel, IAIModelVersion } from '~/models/aiModel'
import { OLLAMA_UPDATE_KEY } from '~/stores/containerStore'
import { containerNameFor } from '~/utils/containerName'
import { manualCommands } from '~/utils/dockerCommands'
import { installedVersionCount, selectCatalogModels, sortLabel, STATE_PILLS } from '~/utils/modelCatalog'

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

const {
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
} = useCatalogFilters()

function installAddedModel(added: { model: string, parameters: string }) {
  containerStore.runContainer(added)
  search.value = added.model.split('/').pop() ?? added.model
  snackbarStore.showSnackbarInfo('Downloading — progress shows on the model\'s card.')
}

const selectedVersions = ref<Record<string, IAIModelVersion | null>>({})

const ollamaUpdateProgress = computed(() => pullProgress.value[OLLAMA_UPDATE_KEY] ?? null)

let runtimeTimer: ReturnType<typeof setInterval> | null = null
const RUNTIME_POLL_MS = 10_000

onMounted(async () => {
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

function statusOfName(containerName: string): string {
  return containers.value.find(c => c.name === containerName)?.status ?? 'not_found'
}

function statusOfVersion(model: IAIModel, version: IAIModelVersion): string {
  return statusOfName(containerNameFor(model.model, version.parameters))
}

const preparedAIModels = computed(() => selectCatalogModels(
  aiModels.value,
  {
    search: search.value,
    sort: sort.value,
    filter: filter.value,
    canProcessImages: canProcessImages.value,
  },
  statusOfVersion,
))

function containerNameOf(model: IAIModel): string | null {
  const version = selectedVersions.value[model.model]

  return version
    ? containerNameFor(model.model, version.parameters)
    : null
}

function statusOf(model: IAIModel): string {
  const name = containerNameOf(model)

  return name
    ? statusOfName(name)
    : 'not_found'
}

function pullProgressOf(model: IAIModel) {
  const name = containerNameOf(model)

  return name
    ? pullProgress.value[name] ?? null
    : null
}

function isOperationLoading(model: IAIModel): boolean {
  const name = containerNameOf(model)
  if (!name || !loadingOperation.value)
    return false

  // operationKey is `${action}_${model}_${parameters}` — anchor on the
  // full container name, not a bare substring of the model name, or e.g.
  // "llama3" would read as loading whenever "llama3.1_8b" was.
  return loadingOperation.value.endsWith(`_${name}`)
}

function loadedInfo(model: IAIModel) {
  const version = selectedVersions.value[model.model]
  if (!version)
    return null

  return loadedModels.value.find(m => m.model === `${model.model}:${version.parameters}`) ?? null
}

// A returning user with an installed model saw "Select a version to manage
// this model" on a model they'd already set up — auto-pick the first
// installed version instead of leaving the card looking unconfigured.
watch([aiModels, containers], () => {
  for (const model of aiModels.value) {
    if (selectedVersions.value[model.model])
      continue

    const installedVersion = model.versions.find(v => statusOfVersion(model, v) !== 'not_found')

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

// The backend allocates ports from a collision-free table, not a formula —
// `11434 + model.index` (the old scheme) stopped meaning anything once that
// changed, and index gets renumbered on every catalog refresh anyway. Show
// the container's real published port if it already exists; otherwise this
// is just a starting point for a manual `docker run` — the app doesn't
// reserve it.
function portOf(model: IAIModel): number {
  const name = containerNameOf(model)
  const existing = name
    ? containers.value.find(c => c.name === name)
    : null

  if (existing?.port)
    return Number.parseInt(existing.port, 10)

  return 11434
}

function commandsFor(model: IAIModel): string[] {
  const version = selectedVersions.value[model.model]
  if (!version)
    return []

  return manualCommands({
    model: model.model,
    parameters: version.parameters,
    status: statusOf(model),
    port: portOf(model),
    gpuAvailable: hardware.value.gpu_available,
    shell: shellType.value,
  })
}

// The card only emits once a size is selected, so these never see a model
// without one — but the page owns that state, so each still checks.
function runWithSelectedVersion(model: IAIModel, run: (target: { model: string, parameters: string }) => void) {
  const version = selectedVersions.value[model.model]
  if (version)
    run({ model: model.model, parameters: version.parameters })
}

function createContainer(model: IAIModel) {
  runWithSelectedVersion(model, (target) => {
    containerStore.runContainer(target)
    snackbarStore.showSnackbarInfo('Container creation started — progress shown on card')
  })
}

function startContainer(model: IAIModel) {
  runWithSelectedVersion(model, (target) => {
    containerStore.runContainer(target)
    snackbarStore.showSnackbarSuccess('Starting container')
  })
}

function resumeDownload(model: IAIModel) {
  runWithSelectedVersion(model, (target) => {
    containerStore.runContainer(target)
    snackbarStore.showSnackbarInfo('Resuming download — it continues from where it stopped')
  })
}

function stopContainer(model: IAIModel) {
  runWithSelectedVersion(model, (target) => {
    containerStore.stopContainer(target)
    snackbarStore.showSnackbarSuccess('Stopping container')
  })
}

function updateModel(model: IAIModel) {
  runWithSelectedVersion(model, target => containerStore.updateModel(target))
}

function unloadModel(model: IAIModel) {
  runWithSelectedVersion(model, target => containerStore.unloadModel(target))
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
          prepend-icon="$mdiServer"
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
      <v-icon
        size="16"
        icon="$mdiServer"
      />

      <span>
        Using your own Ollama at <strong>{{ connection.url }}</strong>. Models you install here are downloaded
        there, and everything already on it is listed.
      </span>
    </div>

    <div
      v-if="hardware.gpu_available === false && !isExternal"
      class="cpu-hint"
    >
      <v-icon
        size="16"
        icon="$mdiChip"
      />

      <span>
        No compatible graphics card (NVIDIA GPU) was found, so models run on your processor (CPU).
        Everything works — answers just arrive more slowly. Smaller models (1–4B) feel best here.
      </span>
    </div>

    <div class="filter-row">
      <div class="search-field">
        <v-icon
          size="16"
          icon="$mdiMagnify"
        />

        <input
          v-model="search"
          type="text"
          placeholder="Search models…"
          aria-label="Search models"
          @input="syncSearchToQuery(search)"
        >
      </div>

      <button
        v-for="pill in STATE_PILLS"
        :key="pill.value"
        type="button"
        class="filter-pill"
        :class="{'filter-pill--active': filter === pill.value}"
        @click="setFilter(pill.value)"
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
        {{ sortLabel(sort) }}
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
      <ModelCard
        v-for="model in preparedAIModels"
        :key="model.model"
        v-model:selected-version="selectedVersions[model.model]"
        :shell="shellType"
        :model="model"
        :installed-version-count="installedVersionCount(model, statusOfVersion)"
        :status="statusOf(model)"
        :pull-progress="pullProgressOf(model)"
        :operation-loading="isOperationLoading(model)"
        :loaded="loadedInfo(model)"
        :hardware="hardware"
        :is-external="isExternal"
        :manual-commands="commandsFor(model)"
        @update:shell="setShell"
        @create="createContainer(model)"
        @start="startContainer(model)"
        @resume="resumeDownload(model)"
        @stop="stopContainer(model)"
        @remove="removeContainer(model)"
        @update="updateModel(model)"
        @unload="unloadModel(model)"
      />
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

@media (max-width: 900px) {
  .models-grid {
    grid-template-columns: 1fr;
  }
}
</style>
