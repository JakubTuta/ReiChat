<script setup lang="ts">
interface ImportReport {
  chats: number
  messages: number
  personas: number
  prompt_templates: number
  mcp_servers: number
  collections: number
  documents: number
  notes: string[]
}

const serverUrl = useRuntimeConfig().public.serverUrl
const apiStore = useApiStore()
const { api } = storeToRefs(apiStore)
const snackbarStore = useSnackbarStore()

const chatStore = useChatStore()
const { aiModels } = storeToRefs(chatStore)
const containerStore = useContainerStore()
const { containers } = storeToRefs(containerStore)

const includeDocuments = ref(false)
const restoring = ref(false)
const report = ref<ImportReport | null>(null)
const backupInput = ref<HTMLInputElement | null>(null)

const chatInput = ref<HTMLInputElement | null>(null)
const importModel = ref<string | null>(null)
const importingChat = ref(false)

onMounted(async () => {
  if (!aiModels.value.length)
    await chatStore.fetchAIModels()
  if (!containers.value.length)
    await containerStore.getUserContainers()
})

const chatModels = computed(() => {
  const installed = new Set(containers.value.map(c => c.environment?.model))

  return aiModels.value.filter(m => !m.is_embedding && installed.has(m.model))
})

watch(chatModels, (list) => {
  if (!importModel.value && list.length)
    importModel.value = list[0].model
}, { immediate: true })

const exportUrl = computed(() => `${serverUrl.replace(/\/$/, '')}/backup/?documents=${includeDocuments.value}`)

const reportLines = computed(() => {
  const r = report.value
  if (!r)
    return []

  return [
    [r.chats, 'chat'],
    [r.messages, 'message'],
    [r.personas, 'persona'],
    [r.prompt_templates, 'prompt template'],
    [r.mcp_servers, 'tool server'],
    [r.collections, 'file set'],
    [r.documents, 'document'],
  ]
    .filter(([count]) => (count as number) > 0)
    .map(([count, noun]) => `${count} ${noun}${count === 1
      ? ''
      : 's'}`)
})

function uploadFile(url: string, file: File) {
  const formData = new FormData()
  formData.append('file', file)

  return api.value.post(url, formData, { timeout: 300_000 })
}

async function restoreBackup(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  target.value = ''
  if (!file)
    return

  restoring.value = true
  report.value = null
  try {
    const response = await uploadFile('backup/', file)
    report.value = response.data
    snackbarStore.showSnackbarSuccess('Backup restored.')
    chatStore.fetchAIModels()
  }
  catch (error: any) {
    console.error(error)
    snackbarStore.showSnackbarError(error.response?.data?.error || 'Restoring the backup failed.')
  }
  finally {
    restoring.value = false
  }
}

async function importChat(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  target.value = ''
  if (!file || !importModel.value)
    return

  importingChat.value = true
  try {
    const response = await uploadFile(`chat-import/${importModel.value}`, file)
    snackbarStore.showSnackbarSuccess(`"${response.data.title}" imported.`)
    await navigateTo({
      path: '/chat',
      query: {
        model: response.data.model,
        parameters: containers.value.find(c => c.environment?.model === response.data.model)?.environment.parameters,
        chat: String(response.data.id),
      },
    })
  }
  catch (error: any) {
    console.error(error)
    snackbarStore.showSnackbarError(error.response?.data?.error || 'Importing the conversation failed.')
  }
  finally {
    importingChat.value = false
  }
}
</script>

<template>
  <AppTopBar />

  <div class="data-page">
    <h1 class="page-h1">
      Your data
    </h1>

    <p class="page-lede">
      Everything you create here stays on this computer. Save a backup to keep it safe or move it to another
      computer, and restore it there.
    </p>

    <section class="data-card">
      <div class="data-card-title">
        Save a backup
      </div>

      <p class="data-card-body">
        Downloads one file with all your chats (including every branch), personas, prompt templates and tool
        servers. Models themselves aren't included — they can be downloaded again.
      </p>

      <label class="data-check">
        <input
          v-model="includeDocuments"
          type="checkbox"
        >
        Include your file sets and documents (makes the file much larger)
      </label>

      <v-btn
        color="mint-btn"
        variant="flat"
        :href="exportUrl"
      >
        Download backup
      </v-btn>
    </section>

    <section class="data-card">
      <div class="data-card-title">
        Restore a backup
      </div>

      <p class="data-card-body">
        Adds everything from a backup file to what's already here — nothing you have now is changed or deleted.
        Tool servers come in switched off, so you can check them first.
      </p>

      <v-btn
        variant="outlined"
        :loading="restoring"
        @click="backupInput?.click()"
      >
        Choose backup file…
      </v-btn>

      <input
        ref="backupInput"
        type="file"
        accept=".json,application/json"
        style="display: none"
        @change="restoreBackup"
      >

      <div
        v-if="report"
        class="data-report"
      >
        <div>Restored {{ reportLines.join(', ') || 'nothing new' }}.</div>

        <ul v-if="report.notes.length">
          <li
            v-for="(note, i) in report.notes"
            :key="i"
          >
            {{ note }}
          </li>
        </ul>
      </div>
    </section>

    <section class="data-card">
      <div class="data-card-title">
        Import one conversation
      </div>

      <p class="data-card-body">
        Opens a conversation saved with <strong>Export conversation → json</strong> in a chat's + menu.
      </p>

      <div
        v-if="!chatModels.length"
        class="data-card-body"
      >
        Install a chat model first — the conversation needs a model to continue with.
      </div>

      <div
        v-else
        class="data-import-row"
      >
        <v-select
          v-model="importModel"
          :items="chatModels"
          item-title="name"
          item-value="model"
          label="Continue it with"
          density="compact"
          hide-details
          class="data-import-select"
        />

        <v-btn
          variant="outlined"
          :loading="importingChat"
          @click="chatInput?.click()"
        >
          Choose conversation file…
        </v-btn>
      </div>

      <input
        ref="chatInput"
        type="file"
        accept=".json,application/json"
        style="display: none"
        @change="importChat"
      >
    </section>
  </div>
</template>

<style scoped>
.data-page {
  max-width: 760px;
  margin: 0 auto;
  padding: 28px 20px 60px;
}

.page-h1 {
  font-size: 30px;
  font-weight: 400;
  letter-spacing: -0.025em;
  color: var(--color-ink);
  margin: 0 0 6px;
}

.page-lede {
  font-size: 14px;
  line-height: 1.6;
  color: var(--color-ink-2);
  margin: 0 0 22px;
}

.data-card {
  margin-bottom: 14px;
  padding: 16px 18px;
  border-radius: 13px;
  border: 1px solid var(--color-line-2);
  background: var(--color-card);
}

.data-card-title {
  font-size: 15px;
  font-weight: 500;
  color: var(--color-ink);
  margin-bottom: 4px;
}

.data-card-body {
  margin: 0 0 12px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
}

.data-check {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--color-ink-2);
  cursor: pointer;
}

.data-report {
  margin-top: 12px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-mint-deep);
}

.data-report ul {
  margin: 6px 0 0;
  padding-left: 18px;
  color: var(--color-ink-2);
}

.data-import-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.data-import-select {
  max-width: 260px;
}
</style>
