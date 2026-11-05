import type { Ref } from 'vue'
import type { IContainer } from '~/models/container'
import type { IChatSummary, IGenerationParams } from '~/stores/chatStore'

// The open chat's summary and the settings read from it. Every chat
// component needs the same "find this chat in this model's list" lookup;
// doing it once here keeps them agreeing on what "no chat open" looks like.
export function useCurrentChat(model: Ref<IContainer | null>, chatId: Ref<string>) {
  const chatStore = useChatStore()
  const { allChats } = storeToRefs(chatStore)

  const chat = computed<IChatSummary | null>(() => {
    if (!model.value || !chatId.value)
      return null

    return allChats.value[model.value.model]?.find(c => c.id === chatId.value) ?? null
  })

  const title = computed(() => chat.value?.title ?? '')
  const persona = computed(() => chat.value?.persona ?? null)
  const activeCollections = computed(() => chat.value?.active_collections ?? [])
  const toolsEnabled = computed(() => chat.value?.tools_enabled ?? false)
  const thinkLevel = computed(() => chat.value?.think_level ?? '')
  const memoryMode = computed(() => chat.value?.memory_mode ?? 'trim')
  const continuedFrom = computed(() => chat.value?.continued_from ?? null)

  const generationParams = computed<IGenerationParams>(() => ({
    temperature: chat.value?.temperature ?? null,
    num_ctx: chat.value?.num_ctx ?? null,
    top_p: chat.value?.top_p ?? null,
    seed: chat.value?.seed ?? null,
  }))

  function refreshContext() {
    if (model.value && chatId.value)
      chatStore.fetchChatContext(model.value.model, chatId.value, model.value.parameters)
  }

  return {
    chat,
    title,
    persona,
    activeCollections,
    toolsEnabled,
    thinkLevel,
    memoryMode,
    continuedFrom,
    generationParams,
    refreshContext,
  }
}
