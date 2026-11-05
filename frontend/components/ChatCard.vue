<script setup lang="ts">
import type ChatDock from '~/components/ChatDock.vue'
import type ChatMessageList from '~/components/ChatMessageList.vue'
import type { WebsocketMessage } from '~/composables/websocket'
import type { IContainer } from '~/models/container'

const props = defineProps<{
  selectedChatId: string
  reset: boolean
}>()

const emit = defineEmits<{
  (e: 'softReset'): void
  (e: 'selectChat', chatId: string): void
}>()

const selectedModel = defineModel<IContainer | null>('selectedModel', { default: null, required: true })
const drawerOpen = defineModel<boolean>('drawerOpen', { default: false })

const { selectedChatId, reset } = toRefs(props)

const composerMessage = ref('')
const listRef = ref<InstanceType<typeof ChatMessageList> | null>(null)
const dockRef = ref<InstanceType<typeof ChatDock> | null>(null)

const editingIndex = ref<number | null>(null)
const editConfirmDialog = ref(false)
const pendingEditIndex = ref<number | null>(null)
const pendingEditContent = ref('')

const chatStore = useChatStore()
const { chatHistoryPerModel, chatContext } = storeToRefs(chatStore)

const containerStore = useContainerStore()

const chatSocketStore = useChatSocketStore()
const {
  websocket,
  waitingForResponse,
  isReconnecting,
  reconnectFailed,
  failedPrompt,
} = storeToRefs(chatSocketStore)

const snackbarStore = useSnackbarStore()

const wsBaseURL = useRuntimeConfig().public.serverUrl

const current = useCurrentChat(selectedModel, selectedChatId)
const { title: currentChatTitle, continuedFrom } = current
const { setMemorySize, switchToSummarize, continueInFreshChat } = useChatSettings(selectedModel, selectedChatId, current)

const suggestions = [
  'Explain quantum computing simply',
  'Write a Python quicksort function',
  'Give me some productivity tips',
  'Summarize the history of the internet',
]

const modelRunning = computed(() => selectedModel.value?.status === 'running')

const chatHistory = computed(() => {
  if (!selectedModel.value || !chatHistoryPerModel.value[selectedModel.value.model])
    return []

  return chatHistoryPerModel.value[selectedModel.value.model]
})

watch([selectedChatId, () => selectedModel.value?.model, () => selectedModel.value?.status], () => {
  chatContext.value = null
  current.refreshContext()
}, { immediate: true })

watch(selectedModel, (newModel, oldModel) => {
  if (!newModel
    || (newModel.model === (oldModel?.model || '')
      && newModel.parameters === (oldModel?.parameters || ''))) {
    return
  }

  softReset()

  containerStore.runContainer(newModel)
  chatStore.fetchAllChats(newModel.model)
}, { immediate: true })

watch(reset, (newValue) => {
  if (newValue) {
    softReset()
  }
})

// The socket itself lives in chatSocketStore, not here — a plain
// component-local ref got closed on every ChatCard unmount, which cancelled
// whatever was streaming any time the user navigated away from /chat. The
// store connection survives navigation; this watcher just tells it which
// room to be on, and reconnects when either the room or (via the immediate
// watch on selectedModel above) the model behind it changes.
watch([selectedChatId, selectedModel], ([newChatId, newModel]) => {
  if (!newChatId || !newModel)
    return

  chatSocketStore.connectToChat(newChatId, newModel.model, wsBaseURL)
}, { immediate: true })

// A prompt the server couldn't answer is gone from the message list (it was
// never stored) — put it back in the composer rather than losing it.
watch(failedPrompt, (text) => {
  if (text && !composerMessage.value.trim())
    composerMessage.value = text
})

function scrollToBottom() {
  listRef.value?.scrollToBottom()
}

function reconnectManually() {
  chatSocketStore.reconnectManually(wsBaseURL)
}

function resetComposerUi() {
  editingIndex.value = null
  editConfirmDialog.value = false
  pendingEditIndex.value = null
  dockRef.value?.clear()
  emit('softReset')
}

function softReset() {
  chatSocketStore.resetStreamState()
  resetComposerUi()
}

function useSuggestion(text: string) {
  composerMessage.value = text
  dockRef.value?.focus()
}

// The JSON shape a reply must follow, when the user turned that on.
function structuredOutputPayload() {
  const fields = dockRef.value?.structuredOutput

  return fields
    ? { structured_output: fields }
    : {}
}

function handleSend(payload: { message: string, image: string }) {
  if (!selectedModel.value)
    return

  const websocketMessage: WebsocketMessage = {
    message: payload.message,
    ai_model: selectedModel.value.model,
    ai_model_parameters: selectedModel.value.parameters,
    ...structuredOutputPayload(),
  }

  if (payload.image)
    websocketMessage.image = payload.image

  if (!chatSocketStore.sendMessage(websocketMessage)) {
    snackbarStore.showSnackbarError('Connection lost — please wait for reconnection or refresh the page.')

    return
  }

  // Not softReset() here — that calls chatSocketStore.resetStreamState(),
  // which would immediately flip waitingForResponse back to false right
  // after sendMessage() just set it true, so the typing indicator never had
  // a chance to render.
  resetComposerUi()

  // The just-sent user message is pushed into chatHistoryPerModel in place
  // (array.push, same reference) — the message list's watch on its history
  // never fires for that because Vue's default (non-deep) watch compares
  // by reference, so it's a no-op on in-place mutation. Scroll explicitly
  // here rather than relying on it.
  nextTick(() => scrollToBottom())
}

function handleStop() {
  chatSocketStore.stopGeneration()
}

function regenerateResponse() {
  if (!selectedModel.value || waitingForResponse.value)
    return

  if (!websocket.value || websocket.value.readyState !== WebSocket.OPEN) {
    snackbarStore.showSnackbarError('Connection lost — please wait for reconnection or refresh the page.')

    return
  }

  // A false return here means there was nothing valid to regenerate (empty
  // history, or the last message isn't an assistant reply) — not a
  // connection problem, so nothing to tell the user.
  chatSocketStore.regenerate({
    ai_model: selectedModel.value.model,
    ai_model_parameters: selectedModel.value.parameters,
    ...structuredOutputPayload(),
  })
}

function handleStartEdit(index: number) {
  if (waitingForResponse.value)
    return

  editingIndex.value = index
}

function handleCancelEdit() {
  editingIndex.value = null
}

function handleRequestSaveEdit({ index, content }: { index: number, content: string }) {
  const list = selectedModel.value && chatHistoryPerModel.value[selectedModel.value.model]
  if (!list)
    return

  pendingEditContent.value = content

  // Editing anything but the last message discards every turn after it —
  // confirm before doing something that destructive.
  if (index < list.length - 1) {
    pendingEditIndex.value = index
    editConfirmDialog.value = true
  }
  else {
    confirmSaveEdit(index)
  }
}

function cancelSaveEdit() {
  editConfirmDialog.value = false
  pendingEditIndex.value = null
}

function confirmSaveEdit(index: number) {
  editConfirmDialog.value = false
  pendingEditIndex.value = null

  if (!selectedModel.value || waitingForResponse.value)
    return

  const newContent = pendingEditContent.value
  if (!newContent)
    return

  if (!websocket.value || websocket.value.readyState !== WebSocket.OPEN) {
    snackbarStore.showSnackbarError('Connection lost — please wait for reconnection or refresh the page.')

    return
  }

  editingIndex.value = null

  // A false return means `index` no longer points at an editable user
  // message (the conversation changed underneath the dialog) — not a
  // connection problem, so nothing to tell the user.
  chatSocketStore.editResend({
    index,
    message: newContent,
    ai_model: selectedModel.value.model,
    ai_model_parameters: selectedModel.value.parameters,
    ...structuredOutputPayload(),
  })
}

async function handleSwitchBranch({ index, siblingIndex }: { index: number, siblingIndex: number }) {
  if (!selectedModel.value || waitingForResponse.value)
    return

  await chatStore.switchBranch(selectedModel.value.model, selectedChatId.value, index, siblingIndex)
  current.refreshContext()
}

async function continueFresh() {
  const newChatId = await continueInFreshChat()
  if (newChatId)
    emit('selectChat', newChatId)
}

function exportChat(format: 'md' | 'json') {
  if (!selectedModel.value || !chatHistory.value.length)
    return

  const model = selectedModel.value.model
  const title = current.chat.value?.title ?? 'chat'
  const messages = chatHistory.value.map(m => ({ role: m.role, content: m.content, image: m.image }))

  if (format === 'md')
    downloadTextFile(title, 'md', buildMarkdownExport(title, messages), 'text/markdown')
  else
    downloadTextFile(title, 'json', buildJsonExport(title, model, messages), 'application/json')
}
</script>

<template>
  <div class="chat-card">
    <ChatStatusBanners
      :model="selectedModel"
      :reconnecting="isReconnecting"
      :reconnect-failed="reconnectFailed"
      @reconnect="reconnectManually"
    />

    <div
      v-if="selectedModel"
      class="thread-header"
    >
      <button
        type="button"
        class="thread-toggle"
        title="Toggle conversations list"
        aria-label="Toggle conversations list"
        @click="drawerOpen = !drawerOpen"
      >
        <v-icon
          size="18"
          icon="$mdiMenu"
        />
      </button>

      <span
        v-if="currentChatTitle"
        class="thread-title"
      >{{ currentChatTitle }}</span>
    </div>

    <ChatMessageList
      ref="listRef"
      :model="selectedModel"
      :chat-history="chatHistory"
      :continued-from="continuedFrom"
      :editing-index="editingIndex"
      :model-running="modelRunning"
      :suggestions="suggestions"
      @regenerate="regenerateResponse"
      @start-edit="handleStartEdit"
      @request-save-edit="handleRequestSaveEdit"
      @cancel-edit="handleCancelEdit"
      @switch-branch="handleSwitchBranch"
      @more-memory="setMemorySize"
      @less-memory="setMemorySize"
      @summarize-memory="switchToSummarize"
      @continue-fresh="continueFresh"
      @use-suggestion="useSuggestion"
    />

    <ChatDock
      ref="dockRef"
      v-model:message="composerMessage"
      :model="selectedModel"
      :chat-id="selectedChatId"
      :has-history="!!chatHistory.length"
      @send="handleSend"
      @stop="handleStop"
      @image-attached="scrollToBottom"
      @export="exportChat"
      @continue-fresh="continueFresh"
    />

    <ConfirmDialog
      v-model="editConfirmDialog"
      title="Edit and resend"
      message="This sends your edited message as a new branch. The original message and its replies aren't deleted — switch back to them any time with the ‹ › arrows."
      confirm-label="Edit and resend"
      confirm-color="mint-btn"
      @confirm="pendingEditIndex !== null && confirmSaveEdit(pendingEditIndex)"
      @cancel="cancelSaveEdit"
    />
  </div>
</template>

<style scoped>
.chat-card {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  min-height: 0;
}

.thread-header {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  padding-bottom: 6px;
}

.thread-toggle {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--color-ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  flex-shrink: 0;
}

.thread-toggle:hover {
  background: oklch(0.94 0.014 168);
}

.thread-title {
  font-size: 13px;
  font-weight: 500;
  color: oklch(0.42 0.016 168);
}
</style>
