<script setup lang="ts">
import type { WebsocketMessage } from '~/constants/websocket'
import type { IContainer } from '~/models/container'
import type { IGenerationParams } from '~/stores/chatStore'

const props = defineProps<{
  selectedChatId: string
  reset: boolean
}>()

const emit = defineEmits<{
  (e: 'softReset'): void
}>()

const selectedModel = defineModel<IContainer | null>('selectedModel', { default: null, required: true })
const drawerOpen = defineModel<boolean>('drawerOpen', { default: false })

const { selectedChatId, reset } = toRefs(props)

const composerMessage = ref('')
const composerRef = ref<{ focus: () => void, clear: () => void } | null>(null)
const scrollToMe = ref<HTMLDivElement | null>(null)
const messagesContainer = ref<HTMLDivElement | null>(null)
const structuredOutputFormat = ref([])
const isFormValid = ref(false)
const jsonEnforced = ref(false)
const showScrollFab = ref(false)

const editingIndex = ref<number | null>(null)
const editConfirmDialog = ref(false)
const pendingEditIndex = ref<number | null>(null)
const pendingEditContent = ref('')

const chatStore = useChatStore()
const { chatHistoryPerModel, aiModels, allChats } = storeToRefs(chatStore)

const containerStore = useContainerStore()

const chatSocketStore = useChatSocketStore()
const {
  botResponse,
  toolCallTrace,
  websocket,
  waitingForResponse,
  isReconnecting,
  reconnectFailed,
} = storeToRefs(chatSocketStore)

const snackbarStore = useSnackbarStore()

const wsBaseURL = useRuntimeConfig().public.serverUrl

const suggestions = [
  'Explain quantum computing simply',
  'Write a Python quicksort function',
  'Give me some productivity tips',
  'Summarize the history of the internet',
]

const currentChatTitle = computed(() => {
  if (!selectedModel.value || !selectedChatId.value)
    return ''

  return allChats.value[selectedModel.value.model]?.find(c => c.id === selectedChatId.value)?.title ?? ''
})

const personaDialogOpen = ref(false)

const currentChatPersona = computed(() => {
  if (!selectedModel.value || !selectedChatId.value)
    return null

  return allChats.value[selectedModel.value.model]?.find(c => c.id === selectedChatId.value)?.persona ?? null
})

async function handlePersonaSelect(persona: { id: string, name: string } | null) {
  if (!selectedModel.value || !selectedChatId.value)
    return

  await chatStore.setChatPersona(selectedModel.value.model, { id: selectedChatId.value }, persona)
}

const parameterPanelOpen = ref(false)

const currentChatGenerationParams = computed<IGenerationParams>(() => {
  const found = selectedModel.value && selectedChatId.value
    ? allChats.value[selectedModel.value.model]?.find(c => c.id === selectedChatId.value)
    : null

  return {
    temperature: found?.temperature ?? null,
    num_ctx: found?.num_ctx ?? null,
    top_p: found?.top_p ?? null,
    seed: found?.seed ?? null,
  }
})

async function handleGenerationParamsSave(params: IGenerationParams) {
  if (!selectedModel.value || !selectedChatId.value)
    return

  await chatStore.setChatGenerationParams(selectedModel.value.model, { id: selectedChatId.value }, params)
}

const promptTemplateDialogOpen = ref(false)

const documentsDialogOpen = ref(false)
const jsonDialogOpen = ref(false)
const plusMenuOpen = ref(false)

const embeddingModels = computed(() => aiModels.value.filter(m => m.is_embedding))

const currentChatActiveCollections = computed(() => {
  if (!selectedModel.value || !selectedChatId.value)
    return []

  return allChats.value[selectedModel.value.model]?.find(c => c.id === selectedChatId.value)?.active_collections ?? []
})

async function handleActiveCollectionsUpdate(collections: { id: number, name: string }[]) {
  if (!selectedModel.value || !selectedChatId.value)
    return

  await chatStore.setChatActiveCollections(selectedModel.value.model, { id: selectedChatId.value }, collections)
}

const currentChatToolsEnabled = computed(() => {
  if (!selectedModel.value || !selectedChatId.value)
    return false

  return allChats.value[selectedModel.value.model]?.find(c => c.id === selectedChatId.value)?.tools_enabled ?? false
})

async function toggleTools() {
  if (!selectedModel.value || !selectedChatId.value)
    return

  await chatStore.setChatToolsEnabled(
    selectedModel.value.model,
    { id: selectedChatId.value },
    !currentChatToolsEnabled.value,
  )
}

type DialogTarget = 'persona' | 'parameters' | 'files' | 'templates' | 'json' | 'tools'

function openDialog(target: DialogTarget) {
  if (target === 'persona')
    personaDialogOpen.value = true
  else if (target === 'parameters')
    parameterPanelOpen.value = true
  else if (target === 'files')
    documentsDialogOpen.value = true
  else if (target === 'templates')
    promptTemplateDialogOpen.value = true
  else if (target === 'json')
    jsonDialogOpen.value = true
  else if (target === 'tools')
    plusMenuOpen.value = true
}

const modelRunning = computed(() => selectedModel.value?.status === 'running')

const sendGateOk = computed(() => !!selectedChatId.value && !!websocket.value && !isReconnecting.value && !reconnectFailed.value,
)

const canUseStructuredOutput = computed(() => (isFormValid.value && structuredOutputFormat.value.length > 0))

const chatHistory = computed(() => {
  if (!selectedModel.value || !chatHistoryPerModel.value[selectedModel.value.model])
    return []

  return chatHistoryPerModel.value[selectedModel.value.model]
})

watch(chatHistory, () => {
  nextTick(() => {
    scrollToBottom()
  })
})

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

watch(botResponse, () => {
  // Fires once per streamed token — respect showScrollFab (the user has
  // manually scrolled up) instead of yanking them back to the bottom on
  // every token, which made it impossible to reread earlier text while a
  // response was still streaming in.
  if (showScrollFab.value)
    return

  nextTick(() => {
    scrollToBottom()
  })
})

function scrollToBottom() {
  if (scrollToMe.value)
    scrollToMe.value.scrollIntoView({ behavior: 'smooth' })
}

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

function reconnectManually() {
  chatSocketStore.reconnectManually(wsBaseURL)
}

function resetComposerUi() {
  editingIndex.value = null
  editConfirmDialog.value = false
  pendingEditIndex.value = null
  composerRef.value?.clear()
  emit('softReset')
}

function softReset() {
  chatSocketStore.resetStreamState()
  resetComposerUi()
}

function useSuggestion(text: string) {
  composerMessage.value = text
  composerRef.value?.focus()
}

function onMessagesScroll() {
  const el = messagesContainer.value
  if (!el)
    return
  showScrollFab.value = el.scrollHeight - el.scrollTop - el.clientHeight > 120
}

function handleSend(payload: { message: string, image: string }) {
  if (!selectedModel.value)
    return

  const websocketMessage: WebsocketMessage = {
    message: payload.message,
    ai_model: selectedModel.value.model,
    ai_model_parameters: selectedModel.value.parameters,
  }

  if (payload.image)
    websocketMessage.image = payload.image

  if (canUseStructuredOutput.value && jsonEnforced.value)
    websocketMessage.structured_output = structuredOutputFormat.value

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
  // (array.push, same reference) — the `watch(chatHistory, ...)` below
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
    ...(canUseStructuredOutput.value && jsonEnforced.value
      ? { structured_output: structuredOutputFormat.value }
      : {}),
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
    ...(canUseStructuredOutput.value && jsonEnforced.value
      ? { structured_output: structuredOutputFormat.value }
      : {}),
  })
}

function handleSwitchBranch({ index, siblingIndex }: { index: number, siblingIndex: number }) {
  switchBranch(index, siblingIndex)
}

async function switchBranch(index: number, siblingIndex: number) {
  if (!selectedModel.value || waitingForResponse.value)
    return

  await chatStore.switchBranch(selectedModel.value.model, selectedChatId.value, index, siblingIndex)
}

function exportChat(format: 'md' | 'json') {
  if (!selectedModel.value || !chatHistory.value.length)
    return

  const model = selectedModel.value.model
  const title = allChats.value[model]?.find(c => c.id === selectedChatId.value)?.title ?? 'chat'
  const messages = chatHistory.value.map(m => ({ role: m.role, content: m.content, image: m.image }))

  if (format === 'md')
    downloadTextFile(title, 'md', buildMarkdownExport(title, messages), 'text/markdown')
  else
    downloadTextFile(title, 'json', buildJsonExport(title, model, messages), 'application/json')
}

watch(waitingForResponse, (newValue) => {
  if (newValue) {
    nextTick(() => {
      scrollToBottom()
    })
  }
})
</script>

<template>
  <div class="chat-card">
    <div
      v-if="selectedModel && selectedModel.status !== 'running'"
      class="status-banner"
    >
      <span class="status-banner-label">Model</span>

      <span class="status-banner-text">
        <template v-if="selectedModel.status === 'pulling_model'">
          Model is still being pulled — sending will be enabled once it's ready.
        </template>

        <template v-else-if="selectedModel.status === 'starting'">
          Model is starting up — this takes a few seconds.
        </template>

        <template v-else-if="selectedModel.status === 'incomplete'">
          This model's download didn't finish. Resume it on the Models page to start chatting.
        </template>

        <template v-else>
          Container is not running.
        </template>
      </span>

      <NuxtLink
        v-if="selectedModel.status !== 'pulling_model'"
        to="/models"
        class="status-banner-action"
        style="text-decoration: none"
      >
        Manage on Models
      </NuxtLink>
    </div>

    <div
      v-if="isReconnecting"
      class="status-banner"
    >
      <span class="status-banner-label">Connection</span>

      <span class="status-banner-text">Connection lost — reconnecting…</span>
    </div>

    <div
      v-if="reconnectFailed"
      class="status-banner"
    >
      <span class="status-banner-label">Connection</span>

      <span class="status-banner-text">Couldn't reconnect. Sending is paused until you reconnect.</span>

      <button
        type="button"
        class="status-banner-action"
        @click="reconnectManually"
      >
        Reconnect
      </button>
    </div>

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
        <v-icon size="18">
          mdi-menu
        </v-icon>
      </button>

      <span
        v-if="currentChatTitle"
        class="thread-title"
      >{{ currentChatTitle }}</span>
    </div>

    <div class="messages-area">
      <div
        ref="messagesContainer"
        class="messages-container"
        @scroll="onMessagesScroll"
      >
        <ChatEmptyState
          v-if="!selectedModel"
          variant="no-model"
        />

        <ChatEmptyState
          v-if="!chatHistory.length && !botResponse && !toolCallTrace.length && !waitingForResponse && selectedModel"
          variant="no-messages"
          :suggestions="suggestions"
          @use-suggestion="useSuggestion"
        />

        <TransitionGroup
          name="msg"
          tag="div"
          class="messages-list"
        >
          <ChatMessage
            v-for="(chatMessage, index) in chatHistory"
            :key="index"
            :message="chatMessage"
            :index="index"
            :is-last="index === chatHistory.length - 1"
            :editing="editingIndex === index"
            :actions-enabled="!waitingForResponse && modelRunning"
            :model="selectedModel"
            @regenerate="regenerateResponse"
            @start-edit="handleStartEdit"
            @request-save-edit="handleRequestSaveEdit"
            @cancel-edit="handleCancelEdit"
            @switch-branch="handleSwitchBranch"
          />

          <StreamingMessage
            v-if="botResponse || toolCallTrace.length"
            key="streaming"
            :content="botResponse"
            :tool-calls="toolCallTrace"
            :model="selectedModel"
          />

          <TypingIndicator
            v-if="waitingForResponse"
            key="typing"
          />
        </TransitionGroup>

        <div ref="scrollToMe" />
      </div>

      <Transition name="fade">
        <button
          v-if="showScrollFab"
          type="button"
          class="scroll-fab"
          title="Scroll to bottom"
          aria-label="Scroll to bottom"
          @click="scrollToBottom"
        >
          <v-icon size="18">
            mdi-chevron-down
          </v-icon>
        </button>
      </Transition>
    </div>

    <div class="composer-dock">
      <div class="composer-dock-inner">
        <div
          v-if="selectedChatId"
          class="chip-row"
        >
          <ChatSettingsChips
            :persona="currentChatPersona"
            :params="currentChatGenerationParams"
            :active-collections-count="currentChatActiveCollections.length"
            :tools-enabled="currentChatToolsEnabled"
            :json-enabled="jsonEnforced"
            @open="openDialog"
          />

          <ChatPlusMenu
            v-model="plusMenuOpen"
            :persona="currentChatPersona"
            :params="currentChatGenerationParams"
            :active-collections-count="currentChatActiveCollections.length"
            :tools-enabled="currentChatToolsEnabled"
            :tools-available="!!selectedModel?.canUseTools"
            :json-enabled="jsonEnforced"
            :json-field-count="structuredOutputFormat.length"
            :has-history="!!chatHistory.length"
            @open="openDialog"
            @toggle-tools="toggleTools"
            @export="exportChat"
          />
        </div>

        <ChatComposer
          ref="composerRef"
          v-model:message="composerMessage"
          :model-running="modelRunning"
          :waiting-for-response="waitingForResponse"
          :send-gate-ok="sendGateOk"
          :show-image-button="!!selectedModel?.canProcessImages"
          @send="handleSend"
          @stop="handleStop"
          @image-attached="scrollToBottom"
        />
      </div>
    </div>

    <ConfirmDialog
      v-model="editConfirmDialog"
      title="Edit and resend"
      message="This replaces this message and deletes every reply after it in this conversation. This can't be undone."
      confirm-label="Edit and resend"
      confirm-color="mint-btn"
      @confirm="pendingEditIndex !== null && confirmSaveEdit(pendingEditIndex)"
      @cancel="cancelSaveEdit"
    />

    <PersonaDialog
      v-model="personaDialogOpen"
      :current-persona-id="currentChatPersona?.id ?? null"
      @select="handlePersonaSelect"
    />

    <ParameterPanelDialog
      v-model="parameterPanelOpen"
      :current-params="currentChatGenerationParams"
      @save="handleGenerationParamsSave"
    />

    <DocumentsDialog
      v-if="selectedChatId"
      v-model="documentsDialogOpen"
      :chat-id="selectedChatId"
      :active-collections="currentChatActiveCollections"
      :embedding-models="embeddingModels"
      @update:active-collections="handleActiveCollectionsUpdate"
    />

    <PromptTemplateDialog
      v-model="promptTemplateDialogOpen"
      @insert="useSuggestion"
    />

    <StructuredOutputSelector
      v-model:is-open="jsonDialogOpen"
      v-model:format="structuredOutputFormat"
      v-model:is-form-valid="isFormValid"
      v-model:enforced="jsonEnforced"
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

.messages-area {
  position: relative;
  flex: 1;
  min-height: 0;
}

.messages-container {
  height: 100%;
  overflow-y: auto;
  padding: 6px 0 190px;
  display: flex;
  flex-direction: column;
}

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 18px;
  max-width: 660px;
  margin: 0 auto;
  width: 100%;
}

.scroll-fab {
  position: absolute;
  bottom: 196px;
  right: 16px;
  z-index: 5;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: var(--color-mint-btn);
  color: var(--color-mint-ink);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: var(--shadow-popover);
}

.composer-dock {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 0 24px 22px;
  background: linear-gradient(to top, var(--color-paper) 62%, transparent);
  pointer-events: none;
}

.composer-dock-inner {
  max-width: 708px;
  margin: 0 auto;
  pointer-events: auto;
}

.chip-row {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.msg-enter-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}

.msg-enter-from {
  opacity: 0;
  transform: translateY(10px);
}

.msg-enter-to {
  opacity: 1;
  transform: translateY(0);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (prefers-reduced-motion: reduce) {
  .msg-enter-active {
    transition: opacity 0.15s ease;
  }

  .msg-enter-from {
    transform: none;
  }
}

@media (max-width: 600px) {
  .composer-dock {
    padding: 0 12px 14px;
  }
}
</style>
