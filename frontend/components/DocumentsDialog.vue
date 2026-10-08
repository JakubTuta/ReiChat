<script setup lang="ts">
import type { IAIModel } from '~/models/aiModel'
import type { IDocument, IDocumentCollection } from '~/models/document'
import { containerNameFor } from '~/utils/containerName'

const props = defineProps<{
  chatId: string
  activeCollections: { id: number, name: string }[]
  embeddingModels: IAIModel[]
}>()

const emit = defineEmits<{
  (e: 'update:activeCollections', collections: { id: number, name: string }[]): void
}>()
const containerStore = useContainerStore()
const { containers, pullProgress } = storeToRefs(containerStore)

// The catalog lists every embedding model ollama.com has; only the ones
// with a container can actually turn files into something searchable.
const RECOMMENDED_EMBEDDER = { model: 'nomic-embed-text', parameters: 'latest' }

function isInstalled(model: string, parameters: string): boolean {
  return containers.value.some(c => c.environment?.model === model && c.environment?.parameters === parameters)
}

const installedEmbeddingModels = computed(() => props.embeddingModels
  .map(m => ({ ...m, versions: m.versions.filter(v => isInstalled(m.model, v.parameters)) }))
  .filter(m => m.versions.length))

const recommendedAvailable = computed(() => props.embeddingModels.some(m => m.model === RECOMMENDED_EMBEDDER.model
  && m.versions.some(v => v.parameters === RECOMMENDED_EMBEDDER.parameters)))

const recommendedProgress = computed(() => pullProgress.value[containerNameFor(RECOMMENDED_EMBEDDER.model, RECOMMENDED_EMBEDDER.parameters)] ?? null)

async function installRecommendedEmbedder() {
  await containerStore.runContainer(RECOMMENDED_EMBEDDER)
  containerStore.connectSocket()
}

const isOpen = defineModel<boolean>({ default: false, required: true })

const documentStore = useDocumentStore()
const { collections, documentsByCollection } = storeToRefs(documentStore)

const creating = ref(false)
const newName = ref('')
const newEmbeddingModelId = ref<string | null>(null)
const newEmbeddingParameters = ref<string | null>(null)
const newIsGlobal = ref(true)

const expandedId = ref<number | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const uploadingId = ref<number | null>(null)

const deleteCollectionConfirm = ref(false)
const collectionToDelete = ref<IDocumentCollection | null>(null)
const deleteDocumentConfirm = ref(false)
const documentToDelete = ref<{ collectionId: number, document: IDocument } | null>(null)

let pollTimer: ReturnType<typeof setTimeout> | null = null
const POLL_INTERVAL_MS = 2000

const availableVersions = computed(() => {
  const model = installedEmbeddingModels.value.find(m => m.id === newEmbeddingModelId.value)

  return model?.versions ?? []
})

watch(newEmbeddingModelId, () => {
  newEmbeddingParameters.value = availableVersions.value[0]?.parameters ?? null
})

watch(installedEmbeddingModels, (models) => {
  if (creating.value && !newEmbeddingModelId.value && models.length)
    newEmbeddingModelId.value = models[0].id
})

function isActive(collection: IDocumentCollection) {
  return props.activeCollections.some(c => c.id === collection.id)
}

function toggleActive(collection: IDocumentCollection) {
  const next = isActive(collection)
    ? props.activeCollections.filter(c => c.id !== collection.id)
    : [...props.activeCollections, { id: collection.id, name: collection.name }]

  emit('update:activeCollections', next)
}

function stopPolling() {
  if (pollTimer !== null) {
    clearTimeout(pollTimer)
    pollTimer = null
  }
}

function scheduleNextPoll() {
  stopPolling()

  const docs = expandedId.value !== null
    ? documentsByCollection.value[expandedId.value]
    : null
  if (!docs?.some(d => d.status === 'pending' || d.status === 'processing'))
    return

  pollTimer = setTimeout(async () => {
    if (expandedId.value !== null) {
      await documentStore.fetchDocuments(expandedId.value)
      scheduleNextPoll()
    }
  }, POLL_INTERVAL_MS)
}

async function toggleExpand(collection: IDocumentCollection) {
  stopPolling()

  if (expandedId.value === collection.id) {
    expandedId.value = null

    return
  }

  expandedId.value = collection.id
  await documentStore.fetchDocuments(collection.id)
  scheduleNextPoll()
}

watch(isOpen, (open) => {
  if (open) {
    documentStore.fetchCollections(props.chatId)
    if (!containers.value.length)
      containerStore.getUserContainers()
  }
  else {
    stopPolling()
    expandedId.value = null
    creating.value = false
  }
})

onUnmounted(stopPolling)

function openCreateForm() {
  creating.value = true
  newName.value = ''
  newEmbeddingModelId.value = installedEmbeddingModels.value[0]?.id ?? null
  newEmbeddingParameters.value = availableVersions.value[0]?.parameters ?? null
  newIsGlobal.value = true
}

async function createCollection() {
  const name = newName.value.trim()
  if (!name || !newEmbeddingModelId.value || !newEmbeddingParameters.value)
    return

  const modelId = Number(newEmbeddingModelId.value)
  const created = await documentStore.createCollection(
    name,
    modelId,
    newEmbeddingParameters.value,
    newIsGlobal.value
      ? undefined
      : props.chatId,
  )

  if (created) {
    creating.value = false
    // A freshly created file set is empty and useless until documents are
    // attached to it — auto-activate it for this chat so the user doesn't
    // have to find the checkbox again right after making it.
    emit('update:activeCollections', [...props.activeCollections, { id: created.id, name: created.name }])
  }
}

function requestDeleteCollection(collection: IDocumentCollection) {
  collectionToDelete.value = collection
  deleteCollectionConfirm.value = true
}

async function confirmDeleteCollection() {
  deleteCollectionConfirm.value = false
  if (!collectionToDelete.value)
    return

  const deletedId = collectionToDelete.value.id
  const deleted = await documentStore.deleteCollection(deletedId)
  collectionToDelete.value = null

  if (deleted) {
    if (expandedId.value === deletedId)
      expandedId.value = null

    if (props.activeCollections.some(c => c.id === deletedId))
      emit('update:activeCollections', props.activeCollections.filter(c => c.id !== deletedId))
  }
}

function triggerUpload(collectionId: number) {
  uploadingId.value = collectionId
  fileInput.value?.click()
}

async function handleFileSelected(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  target.value = ''

  if (!file || uploadingId.value === null)
    return

  const collectionId = uploadingId.value
  uploadingId.value = null

  const succeeded = await documentStore.uploadDocument(collectionId, file)
  if (succeeded && expandedId.value === collectionId)
    scheduleNextPoll()
}

function requestDeleteDocument(collectionId: number, document: IDocument) {
  documentToDelete.value = { collectionId, document }
  deleteDocumentConfirm.value = true
}

async function confirmDeleteDocument() {
  deleteDocumentConfirm.value = false
  if (!documentToDelete.value)
    return

  const { collectionId, document } = documentToDelete.value
  documentToDelete.value = null
  await documentStore.deleteDocument(collectionId, document.id)
}

const STATUS_META: Record<string, { dot: string, text: string }> = {
  pending: { dot: 'grey-dot', text: 'Queued' },
  processing: { dot: 'amber', text: 'Processing…' },
  ready: { dot: 'mint', text: 'Ready' },
  failed: { dot: 'red', text: 'Failed' },
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="540"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Files</span>

        <button
          type="button"
          class="reichat-dialog-close"
          title="Close"
          aria-label="Close"
          @click="isOpen = false"
        >
          <v-icon size="16">
            mdi-close
          </v-icon>
        </button>
      </div>

      <v-card-text class="reichat-dialog-body">
        <div
          v-if="!collections.length && !creating"
          class="files-empty"
        >
          No file sets yet — create one below to give this chat something to reference.
        </div>

        <template
          v-for="collection in collections"
          :key="collection.id"
        >
          <div
            class="file-set-row"
            :class="{'file-set-row--active': isActive(collection)}"
          >
            <input
              type="checkbox"
              class="file-set-checkbox"
              :checked="isActive(collection)"
              title="Use this file set in the current chat"
              aria-label="Use this file set in the current chat"
              @click.stop="toggleActive(collection)"
            >

            <div class="file-set-main">
              <span class="file-set-name">{{ collection.name }}</span>

              <span class="file-set-meta font-mono">
                {{ collection.embedding_model_name }}:{{ collection.embedding_parameters }} ·
                {{ collection.document_count }} file{{ collection.document_count === 1
                  ? ''
                  : 's' }}
              </span>
            </div>

            <button
              type="button"
              class="file-set-expander font-mono"
              @click="toggleExpand(collection)"
            >
              {{ expandedId === collection.id
                ? 'hide'
                : 'files' }}
            </button>

            <button
              type="button"
              class="file-set-delete font-mono"
              @click="requestDeleteCollection(collection)"
            >
              Delete
            </button>
          </div>

          <div
            v-if="expandedId === collection.id"
            class="file-set-documents"
          >
            <div
              v-for="document in documentsByCollection[collection.id] || []"
              :key="document.id"
              class="document-row"
            >
              <span
                class="status-dot"
                :style="{'backgroundColor': `var(--color-${STATUS_META[document.status].dot})`}"
                :title="document.status === 'failed'
                  ? document.error_message
                  : undefined"
              />

              <span class="document-filename font-mono">{{ document.filename }}</span>

              <span class="document-status font-mono">
                {{ STATUS_META[document.status].text }}<template v-if="document.status === 'processing' && document.progress"> {{ document.progress }}%</template>
              </span>

              <button
                type="button"
                class="reichat-dialog-close"
                title="Remove document"
                aria-label="Remove document"
                @click="requestDeleteDocument(collection.id, document)"
              >
                <v-icon size="14">
                  mdi-close
                </v-icon>
              </button>
            </div>

            <div
              v-if="!documentsByCollection[collection.id]?.length"
              class="document-row-empty"
            >
              No files yet.
            </div>

            <button
              type="button"
              class="upload-target font-mono"
              @click="triggerUpload(collection.id)"
            >
              upload .txt .md .pdf .docx
            </button>
          </div>
        </template>

        <input
          ref="fileInput"
          type="file"
          accept=".txt,.md,.pdf,.docx"
          style="display: none"
          @change="handleFileSelected"
        >

        <template v-if="creating">
          <div
            v-if="!installedEmbeddingModels.length"
            class="embedding-warning"
          >
            <p class="embedding-warning-text">
              To search your files, the app needs a small helper model (an "embedding model") that turns
              text into something it can look up. None is installed yet.
            </p>

            <template v-if="recommendedProgress">
              Installing {{ RECOMMENDED_EMBEDDER.model }}… {{ recommendedProgress.percent }}%
            </template>

            <v-btn
              v-else-if="recommendedAvailable"
              size="small"
              color="mint-btn"
              variant="flat"
              @click="installRecommendedEmbedder"
            >
              Install {{ RECOMMENDED_EMBEDDER.model }} (about 270 MB)
            </v-btn>

            <template v-else>
              Install one (e.g. <strong>nomic-embed-text</strong>
              ) on the
              <NuxtLink to="/models">
                Models page
              </NuxtLink>.
            </template>
          </div>

          <v-text-field
            v-model="newName"
            label="File set name"
            density="compact"
            autofocus
            :disabled="!installedEmbeddingModels.length"
            class="mb-2 mt-3"
          />

          <v-select
            v-model="newEmbeddingModelId"
            label="Embedding model"
            :items="installedEmbeddingModels"
            item-title="name"
            item-value="id"
            density="compact"
            class="mb-2"
          />

          <v-select
            v-model="newEmbeddingParameters"
            label="Version"
            :items="availableVersions"
            item-title="parameters"
            item-value="parameters"
            density="compact"
            :disabled="!availableVersions.length"
            hide-details
            class="mb-3"
          />

          <div class="scope-toggle-row">
            <span>Available in every chat (not just this one)</span>

            <v-switch
              v-model="newIsGlobal"
              density="compact"
              hide-details
              color="mint-btn"
            />
          </div>
        </template>

        <template v-if="!creating">
          <button
            type="button"
            class="dashed-add-row"
            @click="openCreateForm"
          >
            + New file set
          </button>
        </template>
      </v-card-text>

      <v-card-actions
        v-if="creating"
        class="reichat-dialog-actions"
      >
        <v-btn
          variant="text"
          @click="creating = false"
        >
          Cancel
        </v-btn>

        <v-spacer />

        <v-btn
          color="mint-btn"
          variant="flat"
          :disabled="!newName.trim() || !newEmbeddingModelId || !newEmbeddingParameters"
          @click="createCollection"
        >
          Create
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <ConfirmDialog
    v-model="deleteCollectionConfirm"
    title="Delete file set"
    message="Every file and embedding in this set is deleted too. This can't be undone."
    confirm-label="Delete"
    confirm-color="red"
    @confirm="confirmDeleteCollection"
    @cancel="collectionToDelete = null"
  />

  <ConfirmDialog
    v-model="deleteDocumentConfirm"
    title="Remove file"
    message="This removes the file and everything the chat learned from it. This can't be undone."
    confirm-label="Remove"
    confirm-color="red"
    @confirm="confirmDeleteDocument"
    @cancel="documentToDelete = null"
  />
</template>

<style scoped>
.files-empty {
  padding: 16px 4px;
  text-align: center;
  font-size: 12.5px;
  color: var(--color-ink-2);
}

.file-set-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 11px;
  border-radius: 10px;
}

.file-set-row--active {
  background: var(--color-mint-tint);
}

.file-set-checkbox {
  width: 15px;
  height: 15px;
  border-radius: 4px;
  accent-color: var(--color-mint-btn);
  flex-shrink: 0;
}

.file-set-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.file-set-name {
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink);
}

.file-set-meta {
  font-size: 10.5px;
  color: var(--color-ink-3);
}

.file-set-expander,
.file-set-delete {
  flex-shrink: 0;
  font-size: 10.5px;
  color: var(--color-mint-deep);
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 4px 6px;
}

.file-set-delete {
  color: var(--color-ink-3);
}

.file-set-delete:hover {
  color: var(--color-red);
}

.file-set-documents {
  padding: 4px 12px 10px 37px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.document-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.document-filename {
  flex: 1;
  min-width: 0;
  font-size: 11px;
  color: var(--color-ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-status {
  font-size: 10px;
  color: var(--color-ink-3);
  flex-shrink: 0;
}

.document-row-empty {
  font-size: 11.5px;
  color: var(--color-ink-3);
  padding: 4px 0;
}

.upload-target {
  margin-top: 4px;
  padding: 8px;
  border-radius: 8px;
  border: 1px dashed var(--color-line-dash);
  background: transparent;
  color: var(--color-ink-3);
  font-size: 10.5px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease;
}

.upload-target:hover {
  border-color: var(--color-mint-border);
  color: var(--color-mint-deep);
}

.embedding-warning {
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-banner-bg);
  border: 1px solid var(--color-banner-border);
  color: var(--color-banner-text);
  font-size: 12px;
}

.embedding-warning-text {
  margin: 0 0 8px;
  line-height: 1.5;
}

.scope-toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12.5px;
  color: var(--color-ink);
}
</style>
