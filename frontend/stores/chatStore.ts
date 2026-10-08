import type { Citation, ContextReport, GenerationStats, ToolCall } from '~/constants/websocket'
import type { IAIModel } from '~/models/aiModel'
import type { ThinkingControls } from '~/utils/thinking'
import { mapAIModel } from '~/models/aiModel'

export interface IChatMessage {
  role: 'user' | 'assistant'
  content: string
  image: string
  // Only present once a message has been loaded from chat-history (GET or
  // the branch-switch PATCH) — optimistic messages pushed straight from a
  // send/regenerate/edit-resend don't know their sibling count yet, so a
  // just-created branch point doesn't show a switcher until the next fetch.
  sibling_count?: number
  sibling_index?: number
  // Ephemeral — only set on a message just streamed in this session, from
  // the WS "done" frame. The backend doesn't persist it, so it's gone after
  // a reload or chat switch.
  stats?: GenerationStats
  citations?: Citation[]
  // Same lifecycle as stats/citations — only set on a message just streamed
  // this session, in call order, never persisted server-side.
  tool_calls?: ToolCall[]
  // Same lifecycle again — reasons attached documents weren't searched.
  notices?: string[]
  context?: ContextReport
  // A thinking model's reasoning for this reply. Persisted, unlike the
  // fields above.
  thinking?: string
}

export type MemoryMode = 'trim' | 'summarize'

// What chat-context/ returns: memory use before anything is sent.
export interface IChatContext extends ContextReport {
  presets: Record<string, number>
  tokens_per_page: number
  // null when the model can't think.
  thinking: ThinkingControls | null
}

export interface IGenerationParams {
  temperature: number | null
  num_ctx: number | null
  top_p: number | null
  seed: number | null
}

export interface IChatSummary extends IGenerationParams {
  id: string
  title: string
  persona: { id: string, name: string } | null
  active_collections: { id: number, name: string }[]
  tools_enabled: boolean
  memory_mode: MemoryMode
  think_level: string
  continued_from: { id: string, title: string } | null
}

export interface ISearchResult {
  chat_id: string
  chat_title: string
  role: 'user' | 'assistant'
  snippet: string
}

export const useChatStore = defineStore('chat', () => {
  const aiModels = ref<IAIModel[]>([])
  const chatHistoryPerModel = ref<{
    [model: string]: IChatMessage[]
  }>({})
  const allChats = ref<{ [model: string]: IChatSummary[] }>({})
  // Memory use of the open chat — from chat-context/ when it's opened, then
  // from each reply's own report.
  const chatContext = ref<IChatContext | null>(null)

  const loading = ref(false)

  const apiStore = useApiStore()
  const snackbarStore = useSnackbarStore()
  const { api } = storeToRefs(apiStore)

  const resetState = () => {
    aiModels.value = []
    chatHistoryPerModel.value = {}
  }

  const fetchAIModels = async () => {
    const url = 'ai-models/'

    loading.value = true

    try {
      const response = await api.value.get(url)

      if (response?.status === 200) {
        aiModels.value = response.data.map(mapAIModel)
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load AI models.')
    }
    finally {
      loading.value = false
    }
  }

  const pullAIModels = async (minPullCount: number) => {
    loading.value = true

    try {
      await api.value.put(`ai-models/?minPullCount=${minPullCount}`)
      snackbarStore.showSnackbarInfo('Scraping models in background — progress shown on the page.')
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to pull AI models.')
    }
    finally {
      loading.value = false
    }
  }

  const fetchAllChats = async (model: string) => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.get(url, { params: { model } })

      // Ids arrive as numbers, but everything client-side (route state,
      // selectedChatId) holds them as strings — normalize once here, or every
      // `c.id === selectedChatId` lookup silently misses.
      if (response?.status === 200) {
        allChats.value[model] = response.data.map((chat: IChatSummary) => ({ ...chat, id: String(chat.id) }))
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load chats.')
    }
  }

  const fetchChatHistory = async (model: string, chatId: string) => {
    const url = `chat-history/${model}/${chatId}`

    try {
      const response = await api.value.get(url, { params: { model } })

      if (response?.status === 200) {
        chatHistoryPerModel.value[model] = response.data
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to load chat history.')
    }
  }

  const switchBranch = async (model: string, chatId: string, index: number, siblingIndex: number) => {
    const url = `chat-history/${model}/${chatId}`

    try {
      const response = await api.value.patch(url, { index, sibling_index: siblingIndex })

      if (response?.status === 200) {
        chatHistoryPerModel.value[model] = response.data
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to switch branch.')
    }
  }

  const createChat = async (model: string) => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.post(url, {})

      if (response?.status === 201) {
        const id = String(response.data.id)
        const { title } = response.data

        if (!allChats.value[model]) {
          allChats.value[model] = []
        }

        // Chats are sorted newest-first (by last_update_time) everywhere
        // else — a freshly created chat belongs at the top, not the bottom,
        // or the order visibly jumps on the next reload.
        allChats.value[model] = [
          {
            id,
            title,
            persona: null,
            temperature: null,
            num_ctx: null,
            top_p: null,
            seed: null,
            active_collections: [],
            tools_enabled: false,
            memory_mode: 'trim',
            think_level: '',
            continued_from: null,
          },
          ...allChats.value[model],
        ]

        return id
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to create a new chat.')
    }

    return null
  }

  const deleteChat = async (model: string, chatId: string): Promise<boolean> => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.delete(url, { data: { chat_id: chatId } })

      if (response.status === 200) {
        if (!allChats.value[model]) {
          allChats.value[model] = []
        }

        allChats.value[model] = allChats.value[model].filter(chat => chat.id !== chatId)

        return true
      }

      return false
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to delete the chat.')

      return false
    }
  }

  const changeChatTitle = async (model: string, chat: { id: string }, newTitle: string) => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.put(url, { id: chat.id, title: newTitle })

      if (response.status === 200) {
        if (!allChats.value[model]) {
          allChats.value[model] = []
        }

        const foundChat = allChats.value[model].find(e => e.id === chat.id)
        if (foundChat) {
          foundChat.title = newTitle
          allChats.value[model] = [foundChat, ...allChats.value[model].filter(e => e.id !== chat.id)]
        }
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to rename the chat.')
    }
  }

  const setChatPersona = async (
    model: string,
    chat: { id: string },
    persona: { id: string, name: string } | null,
  ) => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.put(url, { id: chat.id, persona_id: persona?.id ?? null })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          foundChat.persona = persona
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to set the chat persona.')
    }
  }

  const setChatActiveCollections = async (
    model: string,
    chat: { id: string },
    collections: { id: number, name: string }[],
  ): Promise<boolean> => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.put(url, { id: chat.id, collection_ids: collections.map(c => c.id) })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          foundChat.active_collections = collections

        return true
      }

      return false
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to update active document collections.')

      return false
    }
  }

  const setChatToolsEnabled = async (
    model: string,
    chat: { id: string },
    toolsEnabled: boolean,
  ): Promise<boolean> => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.put(url, { id: chat.id, tools_enabled: toolsEnabled })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          foundChat.tools_enabled = toolsEnabled

        return true
      }

      return false
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to update tool calling.')

      return false
    }
  }

  const setChatGenerationParams = async (
    model: string,
    chat: { id: string },
    params: Partial<IGenerationParams>,
  ) => {
    const url = `all-chats/${model}`

    try {
      const response = await api.value.put(url, { id: chat.id, ...params })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          Object.assign(foundChat, params)
      }

      return true
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to set generation parameters.')

      return false
    }
  }

  const fetchChatContext = async (model: string, chatId: string, parameters: string) => {
    try {
      const response = await api.value.get(`chat-context/${model}/${chatId}`, { params: { parameters } })

      if (response?.status === 200)
        chatContext.value = response.data
    }
    catch (error: any) {
      // The meter just stays hidden — it's informational, and a failure
      // here (e.g. model container still starting) isn't worth a toast.
      console.error(error)
      chatContext.value = null
    }
  }

  const applyReplyContext = (report: ContextReport) => {
    chatContext.value = chatContext.value
      ? { ...chatContext.value, ...report }
      : null
  }

  const setChatMemory = async (
    model: string,
    chat: { id: string },
    memory: { num_ctx?: number | null, memory_mode?: MemoryMode },
  ): Promise<boolean> => {
    try {
      const response = await api.value.put(`all-chats/${model}`, { id: chat.id, ...memory })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          Object.assign(foundChat, memory)

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to change memory settings.')
    }

    return false
  }

  const setChatThinkLevel = async (model: string, chat: { id: string }, thinkLevel: string): Promise<boolean> => {
    try {
      const response = await api.value.put(`all-chats/${model}`, { id: chat.id, think_level: thinkLevel })

      if (response.status === 200) {
        const foundChat = allChats.value[model]?.find(e => e.id === chat.id)
        if (foundChat)
          foundChat.think_level = thinkLevel

        return true
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to change the thinking setting.')
    }

    return false
  }

  const continueChat = async (model: string, chatId: string): Promise<string | null> => {
    try {
      const response = await api.value.post(`chat-continue/${model}`, { chat_id: chatId })

      if (response.status === 201) {
        await fetchAllChats(model)

        return String(response.data.id)
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Failed to start a new chat from this one.')
    }

    return null
  }

  const searchChats = async (model: string, query: string): Promise<ISearchResult[]> => {
    const url = `search/${model}`

    try {
      const response = await api.value.get(url, { params: { q: query } })

      if (response?.status === 200) {
        return response.data
      }
    }
    catch (error: any) {
      console.error(error)
      snackbarStore.showSnackbarError(error.response?.data?.error || 'Search failed.')
    }

    return []
  }

  return {
    aiModels,
    chatHistoryPerModel,
    allChats,
    chatContext,
    loading,
    resetState,
    fetchAIModels,
    pullAIModels,
    fetchAllChats,
    fetchChatHistory,
    switchBranch,
    createChat,
    deleteChat,
    changeChatTitle,
    setChatPersona,
    setChatActiveCollections,
    setChatToolsEnabled,
    setChatGenerationParams,
    fetchChatContext,
    applyReplyContext,
    setChatMemory,
    setChatThinkLevel,
    continueChat,
    searchChats,
  }
})
