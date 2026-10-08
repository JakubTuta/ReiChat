<script setup lang="ts">
import type { IContainer } from '~/models/container'
import { useDisplay } from 'vuetify'

const { mobile } = useDisplay()

const loading = ref(false)
const isShowDrawer = ref(!mobile.value)
const temporaryChatTitle = ref('')
const changingChatTitleId = ref('')
const loadingChangeChatTitle = ref(false)
const selectedModel = ref<IContainer | null>(null)
const selectedChatId = ref('')
const forceReset = ref(false)

const confirmDeleteDialog = ref(false)
const chatToDelete = ref<{ id: string } | null>(null)
const searchDialogOpen = ref(false)

const chatStore = useChatStore()
const { aiModels, allChats, chatHistoryPerModel } = storeToRefs(chatStore)

const containerStore = useContainerStore()
const { containers } = storeToRefs(containerStore)

const hardwareStore = useHardwareStore()
const { diskUsage } = storeToRefs(hardwareStore)

onMounted(async () => {
  loading.value = true

  if (!aiModels.value.length)
    await chatStore.fetchAIModels()

  if (!containers.value.length)
    await containerStore.getUserContainers()

  containerStore.startPolling()
  hardwareStore.fetchDiskUsage()

  loading.value = false
})

onUnmounted(() => {
  containerStore.stopPolling()
})

// Models the chat page can actually talk to: installed, non-embedding,
// merged with their live container status. Drives both the top-bar model
// pill and the auto-select-first-model watcher below.
const userPulledModels = computed(() => {
  if (!containers.value.length || !aiModels.value.length)
    return []

  return containers.value.map((container) => {
    if (container.status === 'pulling_model')
      return null

    const containerModel = container.environment.model
    const foundAIModel = aiModels.value.find(model => model.model === containerModel)

    // Embedding-only models (e.g. nomic-embed-text) don't do chat completion —
    // keep them out of the chat picker even if the user has installed one.
    if (!foundAIModel || foundAIModel.is_embedding)
      return null

    return {
      name: foundAIModel.name,
      value: `${foundAIModel.model} - ${container.environment.parameters}`,
      model: containerModel,
      status: container.status,
      parameters: container.environment.parameters,
      canProcessImages: foundAIModel.can_process_image,
      canUseTools: foundAIModel.can_use_tools,
    } as IContainer
  })
    .filter(e => e !== null)
    .sort((a, b) => a.value.localeCompare(b.value))
})

const route = useRoute()
const router = useRouter()

watch(userPulledModels, (newValue) => {
  if (!newValue.length) {
    selectedModel.value = null

    return
  }

  // Arriving from "Continue in chat" on the Compare page: open that model
  // and chat instead of the default first one.
  const requested = route.query.model
    ? newValue.find(m => m.model === route.query.model && m.parameters === route.query.parameters)
    : null
  if (requested) {
    selectedModel.value = requested
    if (typeof route.query.chat === 'string')
      selectedChatId.value = route.query.chat
    router.replace({ query: {} })

    return
  }

  const stillExists = selectedModel.value
    && newValue.some(m => m.model === selectedModel.value!.model
      && m.parameters === selectedModel.value!.parameters)
  if (!stillExists)
    selectedModel.value = newValue[0]
}, { immediate: true })

const switchDialogOpen = ref(false)
const pendingModel = ref<IContainer | null>(null)

function requestModelChange(model: IContainer | null) {
  if (!model || (selectedModel.value?.model === model.model && selectedModel.value?.parameters === model.parameters))
    return

  const hasMessages = !!selectedModel.value
    && !!selectedChatId.value
    && !!chatHistoryPerModel.value[selectedModel.value.model]?.length

  if (!hasMessages) {
    selectedModel.value = model

    return
  }

  pendingModel.value = model
  switchDialogOpen.value = true
}

async function continueChatWithPendingModel() {
  switchDialogOpen.value = false
  const target = pendingModel.value
  const source = selectedModel.value
  const chatId = selectedChatId.value
  pendingModel.value = null
  if (!target || !source || !chatId)
    return

  // Same model family, different size: the chat already belongs to it —
  // only the size used for the next reply changes.
  if (target.model === source.model) {
    selectedModel.value = target

    return
  }

  if (await chatStore.moveChat(source.model, chatId, target.model)) {
    selectedModel.value = target
    await nextTick()
    await chatStore.fetchChatHistory(target.model, chatId)
  }
}

function startNewChatWithPendingModel() {
  switchDialogOpen.value = false
  if (pendingModel.value)
    selectedModel.value = pendingModel.value
  pendingModel.value = null
}

watch(selectedChatId, async (newChatId, oldChatId) => {
  if (!newChatId || !selectedModel.value || newChatId === oldChatId)
    return

  loading.value = true
  forceReset.value = true

  await chatStore.fetchChatHistory(selectedModel.value.model, newChatId)

  loading.value = false
}, { immediate: true })

watch(allChats, (newValue) => {
  if (!selectedModel.value || !newValue[selectedModel.value.model])
    return

  const chats = newValue[selectedModel.value.model]
  const currentValid = chats.some(c => c.id.toString() === selectedChatId.value)

  if (!currentValid) {
    if (chats.length)
      selectedChatId.value = chats[0].id.toString()
    else
      createNewChat(selectedModel.value.model)
  }
}, { deep: true })

function softReset() {
  temporaryChatTitle.value = ''
  changingChatTitleId.value = ''
  loadingChangeChatTitle.value = false
  forceReset.value = false
}

function changeChat(chat: any) {
  selectedChatId.value = chat.id.toString()
  if (mobile.value)
    isShowDrawer.value = false
}

function changeChatTitle(chat: { id: string, title: string }) {
  temporaryChatTitle.value = chat.title
  changingChatTitleId.value = chat.id
}

function isChangingMyTitle(chat: { id: string }) {
  if (!changingChatTitleId.value)
    return false

  return changingChatTitleId.value === chat.id
}

async function acceptChangeChatTitle(chat: { id: string, title: string }) {
  const newTitle = temporaryChatTitle.value.trim()

  if (!newTitle || !selectedModel.value)
    return

  loadingChangeChatTitle.value = true

  const model = selectedModel.value.model
  await chatStore.changeChatTitle(model, chat, newTitle)

  loadingChangeChatTitle.value = false
  changingChatTitleId.value = ''
  temporaryChatTitle.value = ''
}

function cancelChangeChatTitle() {
  changingChatTitleId.value = ''
  temporaryChatTitle.value = ''
}

// Vue collects an array of matches when a ref sits inside v-for, even
// though only one row is ever in rename mode at a time — normalise either
// shape defensively rather than assume which one shows up.
const renameInputRef = ref<HTMLInputElement | HTMLInputElement[] | null>(null)

function addSpace() {
  if (!changingChatTitleId.value)
    return

  const inputEl = Array.isArray(renameInputRef.value)
    ? renameInputRef.value[0]
    : renameInputRef.value

  if (!inputEl) {
    temporaryChatTitle.value += ' '

    return
  }

  const start = inputEl.selectionStart ?? temporaryChatTitle.value.length
  const end = inputEl.selectionEnd ?? start
  const value = temporaryChatTitle.value

  temporaryChatTitle.value = `${value.slice(0, start)} ${value.slice(end)}`

  nextTick(() => {
    inputEl.setSelectionRange(start + 1, start + 1)
    inputEl.focus()
  })
}

async function createNewChat(model: string) {
  const newChatId = await chatStore.createChat(model)

  if (newChatId !== null)
    selectedChatId.value = newChatId.toString()
}

function deleteChat(chat: { id: string }) {
  chatToDelete.value = chat
  confirmDeleteDialog.value = true
}

async function confirmDeleteChat() {
  confirmDeleteDialog.value = false

  if (!selectedModel.value || !chatToDelete.value)
    return

  const deletedChatId = chatToDelete.value.id
  const model = selectedModel.value.model
  const succeeded = await chatStore.deleteChat(model, deletedChatId)
  chatToDelete.value = null

  // On failure the chat is still in allChats — don't navigate away and
  // imply something happened when the delete actually failed.
  if (!succeeded || deletedChatId !== selectedChatId.value)
    return

  if (allChats.value[model]?.length)
    selectedChatId.value = allChats.value[model][0].id.toString()
  else
    await createNewChat(model)
}

function cancelDeleteChat() {
  confirmDeleteDialog.value = false
  chatToDelete.value = null
}

function handleSearchSelect(chatId: string) {
  changeChat({ id: chatId })
}
</script>

<template>
  <AppTopBar>
    <template #append>
      <ModelPillMenu
        :model-value="selectedModel"
        :models="userPulledModels"
        @update:model-value="requestModelChange"
      />
    </template>
  </AppTopBar>

  <v-navigation-drawer
    v-if="selectedModel"
    v-model="isShowDrawer"
    :temporary="mobile"
    width="262"
    class="chat-drawer"
  >
    <div class="drawer-header">
      <span class="mono-kicker">Chats · {{ selectedModel.name }}</span>

      <div class="drawer-header-actions">
        <button
          type="button"
          class="drawer-icon-btn"
          title="Search chats"
          aria-label="Search chats"
          @click="searchDialogOpen = true"
        >
          <v-icon size="15">
            mdi-magnify
          </v-icon>
        </button>

        <button
          type="button"
          class="drawer-icon-btn drawer-icon-btn--mint"
          title="New chat"
          @click="createNewChat(selectedModel!.model)"
        >
          <v-icon size="15">
            mdi-plus
          </v-icon>
        </button>
      </div>
    </div>

    <div class="drawer-list">
      <div
        v-for="chat in allChats[selectedModel.model] || []"
        :key="chat.id"
        class="drawer-row"
        :class="{'drawer-row--active': selectedChatId === chat.id.toString()}"
        @click="changeChat(chat)"
      >
        <input
          v-if="isChangingMyTitle(chat)"
          ref="renameInputRef"
          v-model="temporaryChatTitle"
          class="drawer-rename-input"
          autofocus
          @click.stop
          @keydown.enter="acceptChangeChatTitle(chat)"
          @keydown.esc="cancelChangeChatTitle"
          @keydown.space.prevent="addSpace"
        >

        <span
          v-else
          class="drawer-row-title"
        >
          {{ chat.title }}
        </span>

        <div class="drawer-row-actions">
          <template v-if="!isChangingMyTitle(chat)">
            <button
              type="button"
              class="drawer-icon-btn"
              title="Rename chat"
              aria-label="Rename chat"
              @click.stop="changeChatTitle(chat)"
            >
              <v-icon size="13">
                mdi-pencil
              </v-icon>
            </button>

            <button
              type="button"
              class="drawer-icon-btn drawer-icon-btn--danger"
              title="Delete chat"
              aria-label="Delete chat"
              @click.stop="deleteChat(chat)"
            >
              <v-icon size="13">
                mdi-close
              </v-icon>
            </button>
          </template>

          <button
            v-else
            type="button"
            class="drawer-icon-btn drawer-icon-btn--danger"
            title="Cancel rename"
            aria-label="Cancel rename"
            :disabled="loadingChangeChatTitle"
            @click.stop="cancelChangeChatTitle()"
          >
            <v-icon size="13">
              mdi-close
            </v-icon>
          </button>
        </div>
      </div>
    </div>

    <div class="drawer-footer mono-kicker">
      {{ diskUsage.models.length }} installed · {{ formatBytes(diskUsage.total_bytes) }}
    </div>
  </v-navigation-drawer>

  <v-dialog
    v-model="switchDialogOpen"
    max-width="440"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Switch to {{ pendingModel?.name }} {{ pendingModel?.parameters }}</span>
      </div>

      <v-card-text class="reichat-dialog-body switch-body">
        Keep talking in this conversation with the new model, or start a fresh chat? Earlier replies
        stay labeled with the model that wrote them.
      </v-card-text>

      <v-card-actions class="reichat-dialog-actions">
        <v-btn
          variant="text"
          @click="switchDialogOpen = false; pendingModel = null"
        >
          Cancel
        </v-btn>

        <v-spacer />

        <v-btn
          variant="outlined"
          @click="startNewChatWithPendingModel"
        >
          New chat
        </v-btn>

        <v-btn
          color="mint-btn"
          variant="flat"
          @click="continueChatWithPendingModel"
        >
          Continue this chat
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <ConfirmDialog
    v-model="confirmDeleteDialog"
    title="Delete chat"
    message="Are you sure you want to delete this chat? This action cannot be undone."
    confirm-label="Delete"
    confirm-color="red"
    @confirm="confirmDeleteChat"
    @cancel="cancelDeleteChat"
  />

  <SearchDialog
    v-if="selectedModel"
    v-model="searchDialogOpen"
    :model="selectedModel.model"
    @select="handleSearchSelect"
  />

  <div class="chat-page-content">
    <div
      v-if="loading"
      class="d-flex align-center justify-center"
      style="height: 100%"
    >
      <v-progress-circular
        indeterminate
        color="mint"
      />
    </div>

    <template v-else>
      <SystemStatusBanner class="chat-page-banner" />

      <ChatCard
        v-model:selected-model="selectedModel"
        v-model:drawer-open="isShowDrawer"
        :reset="forceReset"
        :selected-chat-id="selectedChatId"
        @soft-reset="softReset"
        @select-chat="(id: string) => selectedChatId = id"
      />
    </template>
  </div>
</template>

<style scoped>
.switch-body {
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
}

.chat-drawer {
  background: var(--color-soft) !important;
  border-color: var(--color-line-2) !important;
}

/* Vuetify's default-slot wrapper (.v-navigation-drawer__content) isn't a
   flex container on its own — without this, .drawer-list's `flex: 1` has
   no effect and .drawer-footer ends up right after the chat list instead
   of pinned to the bottom of the drawer. */
.chat-drawer :deep(.v-navigation-drawer__content) {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.drawer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 12px 8px;
}

.drawer-header-actions {
  display: flex;
  gap: 2px;
}

.drawer-icon-btn {
  width: 26px;
  height: 26px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--color-ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background-color 0.15s ease, opacity 0.15s ease, transform 0.5s ease;
}

.drawer-icon-btn:hover {
  background: var(--color-line-2);
}

.drawer-icon-btn--mint {
  color: var(--color-mint-deep);
}

.drawer-icon-btn--mint:hover {
  transform: rotate(90deg);
}

.drawer-icon-btn--danger:hover {
  color: var(--color-red);
}

.drawer-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.drawer-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 9px 11px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.drawer-row:hover {
  background: oklch(0.93 0.02 168);
}

.drawer-row--active {
  background: var(--color-mint-tint);
}

.drawer-row-title {
  flex: 1;
  min-width: 0;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.drawer-rename-input {
  flex: 1;
  min-width: 0;
  background: transparent;
  border: none;
  outline: none;
  font: inherit;
  font-size: 12.5px;
  color: var(--color-ink);
}

.drawer-row-actions {
  display: flex;
  gap: 0;
  flex-shrink: 0;
  opacity: 0.4;
  transition: opacity 0.15s ease;
}

.drawer-row:hover .drawer-row-actions,
.drawer-row:focus-within .drawer-row-actions {
  opacity: 1;
}

.drawer-footer {
  padding: 10px 14px;
  border-top: 1px solid var(--color-line-2);
  flex-shrink: 0;
}

.chat-page-content {
  display: flex;
  flex-direction: column;
  height: calc(100dvh - 56px);
  padding: 12px 24px 16px;
  overflow: hidden;
}

.chat-page-banner {
  flex-shrink: 0;
}
</style>
