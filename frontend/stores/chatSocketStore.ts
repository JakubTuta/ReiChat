import type { EditResendPayload, RegeneratePayload, ToolCall, WebsocketHandlers, WebsocketMessage, WebsocketResponse } from '~/constants/websocket'
import type { IChatMessage } from '~/stores/chatStore'
import { getWebsocket } from '~/constants/websocket'

// Owns the chat WebSocket as a store singleton instead of component state,
// so a response keeps streaming (and lands in chat history) while the user
// navigates to another page mid-generation — a plain `ref` inside
// ChatCard.vue got torn down (and the socket closed) on every unmount,
// which made every route change away from /chat cancel whatever was
// in-flight.
export const useChatSocketStore = defineStore('chatSocket', () => {
  const chatStore = useChatStore()
  const { chatHistoryPerModel } = storeToRefs(chatStore)
  const snackbarStore = useSnackbarStore()

  const activeChatId = ref('')
  const activeModel = ref('')
  const websocket = ref<WebSocketWrapper | null>(null)
  const botResponse = ref('')
  const toolCallTrace = ref<ToolCall[]>([])
  const notices = ref<string[]>([])
  const waitingForResponse = ref(false)
  const isReconnecting = ref(false)
  const reconnectFailed = ref(false)
  const pendingReplaceBackup = ref<{ index: number, messages: IChatMessage[] } | null>(null)
  // Text of a prompt the server rejected or couldn't answer — handed back
  // to the composer so the user doesn't have to retype it.
  const failedPrompt = ref('')
  let lastPromptText = ''
  let hasConnectedBefore = false

  // The local list is optimistic (a sent message shows before the server
  // stores anything), but edit/regenerate/branch-switch address messages by
  // their position in it. After a failed request — never persisted
  // server-side — or a dropped connection — which can persist a partial
  // reply the client never saw — positions would point at different
  // messages than the server's, so edits could land on the wrong one.
  // Reloading from the server makes its copy the truth again.
  async function resyncHistory() {
    if (activeModel.value && activeChatId.value)
      await chatStore.fetchChatHistory(activeModel.value, activeChatId.value)
  }

  function restorePendingReplace() {
    if (!pendingReplaceBackup.value || !activeModel.value)
      return

    const list = chatHistoryPerModel.value[activeModel.value]
    if (list) {
      const { index, messages } = pendingReplaceBackup.value
      list.splice(index, list.length - index, ...messages)
    }

    pendingReplaceBackup.value = null
  }

  function resetStreamState() {
    botResponse.value = ''
    toolCallTrace.value = []
    notices.value = []
    waitingForResponse.value = false
    pendingReplaceBackup.value = null
  }

  const handlers: WebsocketHandlers = {
    onConnect: () => {
      isReconnecting.value = false
      reconnectFailed.value = false

      if (hasConnectedBefore) {
        resetStreamState()
        resyncHistory()
      }
      hasConnectedBefore = true
    },

    onDisconnect: () => {
      waitingForResponse.value = false
    },

    onReconnecting: (_attempt: number) => {
      isReconnecting.value = true
    },

    onMaxReconnectFailed: () => {
      isReconnecting.value = false
      reconnectFailed.value = true
      waitingForResponse.value = false
    },

    onError: (errorMessage: string) => {
      const hadPendingPrompt = !!lastPromptText
      waitingForResponse.value = false
      restorePendingReplace()
      botResponse.value = ''
      toolCallTrace.value = []
      notices.value = []
      snackbarStore.showSnackbarError(errorMessage)

      if (hadPendingPrompt) {
        failedPrompt.value = lastPromptText
        lastPromptText = ''
      }
      resyncHistory()
    },

    onNotice: (notice: string) => {
      notices.value = [...notices.value, notice]
    },

    onSendMessage: (message: WebsocketMessage) => {
      const model = message.ai_model

      if (!chatHistoryPerModel.value[model])
        chatHistoryPerModel.value[model] = []

      chatHistoryPerModel.value[model].push({
        role: 'user',
        content: message.message,
        image: message.image ?? '',
      })
    },

    onReceiveMessage: (message: WebsocketResponse) => {
      if (message.done)
        lastPromptText = ''

      if (message.tool_call) {
        waitingForResponse.value = false
        toolCallTrace.value = [...toolCallTrace.value, message.tool_call]

        return
      }

      waitingForResponse.value = false
      if (message.done) {
        const content = message.stopped
          ? botResponse.value
          : message.message

        if (activeModel.value && content) {
          chatHistoryPerModel.value[activeModel.value].push({
            role: 'assistant',
            content,
            image: '',
            ...(message.stats
              ? { stats: message.stats }
              : {}),
            ...(message.citations
              ? { citations: message.citations }
              : {}),
            ...(toolCallTrace.value.length
              ? { tool_calls: toolCallTrace.value }
              : {}),
            ...(notices.value.length
              ? { notices: notices.value }
              : {}),
          })
          pendingReplaceBackup.value = null
        }
        else if (pendingReplaceBackup.value) {
          restorePendingReplace()
        }
        botResponse.value = ''
        toolCallTrace.value = []
        notices.value = []
      }
      else {
        botResponse.value += message.message
      }
    },
  }

  function connectToChat(chatId: string, model: string, baseURL: string) {
    if (!chatId)
      return

    // Already connected (or connecting/reconnecting) to this exact room —
    // e.g. ChatCard remounted after navigating back to /chat — reuse the
    // live socket and whatever streamed in while it was unmounted instead
    // of tearing it down and losing that state.
    if (activeChatId.value === chatId && websocket.value && websocket.value.readyState !== WebSocket.CLOSED) {
      activeModel.value = model

      return
    }

    if (websocket.value)
      websocket.value.closeConnection()

    activeChatId.value = chatId
    activeModel.value = model
    hasConnectedBefore = false
    lastPromptText = ''
    resetStreamState()
    isReconnecting.value = false
    reconnectFailed.value = false

    websocket.value = getWebsocket(handlers, chatId, baseURL)
  }

  function reconnectManually(baseURL: string) {
    if (!activeChatId.value)
      return

    if (websocket.value)
      websocket.value.closeConnection()

    reconnectFailed.value = false
    websocket.value = getWebsocket(handlers, activeChatId.value, baseURL)
  }

  function sendMessage(message: WebsocketMessage): boolean {
    if (!websocket.value || websocket.value.readyState !== WebSocket.OPEN)
      return false

    waitingForResponse.value = true
    activeModel.value = message.ai_model
    lastPromptText = message.message
    failedPrompt.value = ''
    websocket.value.sendMessage(message)

    return true
  }

  function stopGeneration() {
    if (websocket.value?.readyState === WebSocket.OPEN)
      websocket.value.stopGeneration()
  }

  function regenerate(payload: RegeneratePayload): boolean {
    if (!websocket.value || websocket.value.readyState !== WebSocket.OPEN)
      return false

    const model = payload.ai_model
    const list = chatHistoryPerModel.value[model]
    if (!list?.length)
      return false

    const lastIndex = list.length - 1
    const last = list[lastIndex]
    if (last.role !== 'assistant')
      return false

    // Removed optimistically so the streaming reply doesn't render alongside
    // the answer it's replacing; restored if the regenerate doesn't pan out
    // (see onError / onReceiveMessage above) since the backend never deletes
    // the original until a new response has actually been produced.
    pendingReplaceBackup.value = { index: lastIndex, messages: [last] }
    list.splice(lastIndex, 1)

    activeModel.value = model
    waitingForResponse.value = true
    botResponse.value = ''

    websocket.value.regenerate(payload)

    return true
  }

  function editResend(payload: EditResendPayload): boolean {
    if (!websocket.value || websocket.value.readyState !== WebSocket.OPEN)
      return false

    const model = payload.ai_model
    const list = chatHistoryPerModel.value[model]
    const target = list?.[payload.index]
    if (!list || !target || target.role !== 'user')
      return false

    // Same optimistic-remove-then-restore-on-failure pattern as regenerate,
    // but the backup can span several messages since editing an earlier turn
    // discards everything after it too.
    pendingReplaceBackup.value = { index: payload.index, messages: list.slice(payload.index) }
    list.splice(payload.index, list.length - payload.index, { role: 'user', content: payload.message, image: target.image })

    activeModel.value = model
    waitingForResponse.value = true
    botResponse.value = ''

    // The edit flow only ever changes the text — the original message's
    // image (if any) rides along automatically rather than requiring every
    // caller to look it up and pass it through.
    websocket.value.editResend(target.image
      ? { ...payload, image: target.image }
      : payload)

    return true
  }

  return {
    activeChatId,
    activeModel,
    websocket,
    botResponse,
    toolCallTrace,
    notices,
    waitingForResponse,
    isReconnecting,
    reconnectFailed,
    pendingReplaceBackup,
    failedPrompt,
    connectToChat,
    reconnectManually,
    sendMessage,
    stopGeneration,
    regenerate,
    editResend,
    restorePendingReplace,
    resetStreamState,
  }
})
