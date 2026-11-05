import type { Ref } from 'vue'
import type { useCurrentChat } from '~/composables/useCurrentChat'
import type { IContainer } from '~/models/container'
import type { IGenerationParams, MemoryMode } from '~/stores/chatStore'

// Everything that changes a setting of the open chat. Each does nothing
// without a model and chat open — the controls that call these aren't shown
// then, but a stale click mid-switch must not write to the wrong chat.
export function useChatSettings(
  model: Ref<IContainer | null>,
  chatId: Ref<string>,
  current: ReturnType<typeof useCurrentChat>,
) {
  const chatStore = useChatStore()
  const snackbarStore = useSnackbarStore()

  function target() {
    return model.value && chatId.value
      ? { model: model.value.model, chat: { id: chatId.value } }
      : null
  }

  async function selectPersona(persona: { id: string, name: string } | null) {
    const t = target()
    if (t)
      await chatStore.setChatPersona(t.model, t.chat, persona)
  }

  async function saveGenerationParams(params: Partial<IGenerationParams>) {
    const t = target()
    if (t)
      await chatStore.setChatGenerationParams(t.model, t.chat, params)
  }

  async function saveMemory(memory: { num_ctx?: number | null, memory_mode?: MemoryMode }) {
    const t = target()
    if (t && await chatStore.setChatMemory(t.model, t.chat, memory)) {
      current.refreshContext()
      snackbarStore.showSnackbarSuccess('Memory settings saved — they apply from the next message.')
    }
  }

  function setMemorySize(tokens: number) {
    return saveMemory({ num_ctx: tokens })
  }

  function switchToSummarize() {
    return saveMemory({ memory_mode: 'summarize' })
  }

  async function setThinkLevel(thinkLevel: string) {
    const t = target()
    if (t)
      await chatStore.setChatThinkLevel(t.model, t.chat, thinkLevel)
  }

  async function toggleTools() {
    const t = target()
    if (t)
      await chatStore.setChatToolsEnabled(t.model, t.chat, !current.toolsEnabled.value)
  }

  async function updateActiveCollections(collections: { id: number, name: string }[]) {
    const t = target()
    if (t)
      await chatStore.setChatActiveCollections(t.model, t.chat, collections)
  }

  // Resolves to the new chat's id so the caller can open it, or null when
  // nothing was started.
  async function continueInFreshChat(): Promise<string | null> {
    const t = target()
    if (!t)
      return null

    const newChatId = await chatStore.continueChat(t.model, t.chat.id)
    if (newChatId)
      snackbarStore.showSnackbarInfo('New chat started — your first message there will include a summary of the old one.')

    return newChatId
  }

  return {
    selectPersona,
    saveGenerationParams,
    saveMemory,
    setMemorySize,
    switchToSummarize,
    setThinkLevel,
    toggleTools,
    updateActiveCollections,
    continueInFreshChat,
  }
}
