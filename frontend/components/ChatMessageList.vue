<script setup lang="ts">
import type { IContainer } from '~/models/container'
import type { IChatMessage } from '~/stores/chatStore'

const props = defineProps<{
  model: IContainer | null
  chatHistory: IChatMessage[]
  continuedFrom: { id: string, title: string } | null
  editingIndex: number | null
  modelRunning: boolean
  suggestions: string[]
}>()

const emit = defineEmits<{
  (e: 'regenerate'): void
  (e: 'startEdit', index: number): void
  (e: 'requestSaveEdit', payload: { index: number, content: string }): void
  (e: 'cancelEdit'): void
  (e: 'switchBranch', payload: { index: number, siblingIndex: number }): void
  (e: 'moreMemory' | 'lessMemory', tokens: number): void
  (e: 'summarizeMemory'): void
  (e: 'continueFresh'): void
  (e: 'useSuggestion', text: string): void
}>()

const chatSocketStore = useChatSocketStore()
const {
  botResponse,
  toolCallTrace,
  notices,
  thinkingResponse,
  pendingApproval,
  waitingForResponse,
} = storeToRefs(chatSocketStore)

const mcpServerStore = useMCPServerStore()

const scrollToMe = ref<HTMLDivElement | null>(null)
const messagesContainer = ref<HTMLDivElement | null>(null)
const showScrollFab = ref(false)

const actionsEnabled = computed(() => !waitingForResponse.value && props.modelRunning)

function scrollToBottom() {
  if (scrollToMe.value)
    scrollToMe.value.scrollIntoView({ behavior: 'smooth' })
}

function onMessagesScroll() {
  const el = messagesContainer.value
  if (!el)
    return

  showScrollFab.value = el.scrollHeight - el.scrollTop - el.clientHeight > 120
}

watch(() => props.chatHistory, () => {
  nextTick(() => {
    scrollToBottom()
  })
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

watch(waitingForResponse, (newValue) => {
  if (newValue) {
    nextTick(() => {
      scrollToBottom()
    })
  }
})

async function decideToolApproval({ approved, always }: { approved: boolean, always: boolean }) {
  const server = pendingApproval.value?.server
  chatSocketStore.respondToolApproval(approved)

  if (always && server) {
    if (!mcpServerStore.servers.length)
      await mcpServerStore.fetchServers()
    const found = mcpServerStore.servers.find(s => s.name === server)
    if (found)
      await mcpServerStore.setServerAutoApprove(found.id, true)
  }
}

defineExpose({ scrollToBottom })
</script>

<template>
  <div class="messages-area">
    <div
      ref="messagesContainer"
      class="messages-container"
      @scroll="onMessagesScroll"
    >
      <ChatEmptyState
        v-if="!model"
        variant="no-model"
      />

      <ChatEmptyState
        v-if="!chatHistory.length && !botResponse && !toolCallTrace.length && !waitingForResponse && model"
        variant="no-messages"
        :suggestions="suggestions"
        @use-suggestion="emit('useSuggestion', $event)"
      />

      <div
        v-if="continuedFrom"
        class="continued-banner"
      >
        <v-icon
          size="14"
          icon="$mdiFileDocumentRefreshOutline"
        />

        <span>
          Continues "{{ continuedFrom.title }}" — the model starts out knowing a summary
          of it.
        </span>
      </div>

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
          :actions-enabled="actionsEnabled"
          :model="model"
          @regenerate="emit('regenerate')"
          @start-edit="emit('startEdit', $event)"
          @request-save-edit="emit('requestSaveEdit', $event)"
          @cancel-edit="emit('cancelEdit')"
          @switch-branch="emit('switchBranch', $event)"
          @more-memory="emit('moreMemory', $event)"
          @less-memory="emit('lessMemory', $event)"
          @summarize-memory="emit('summarizeMemory')"
          @continue-fresh="emit('continueFresh')"
        />

        <StreamingMessage
          v-if="botResponse || toolCallTrace.length || notices.length || thinkingResponse"
          key="streaming"
          :content="botResponse"
          :tool-calls="toolCallTrace"
          :notices="notices"
          :thinking="thinkingResponse"
          :model="model"
        />

        <ToolApprovalCard
          v-if="pendingApproval"
          key="approval"
          :request="pendingApproval"
          @decide="decideToolApproval"
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
        <v-icon
          size="18"
          icon="$mdiChevronDown"
        />
      </button>
    </Transition>
  </div>
</template>

<style scoped>
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

.continued-banner {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  max-width: 760px;
  margin: 0 auto 14px;
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-size: 12.5px;
  line-height: 1.5;
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
</style>
