<script setup lang="ts">
import type { IContainer } from '~/models/container'

type DialogTarget = 'persona' | 'parameters' | 'files' | 'templates' | 'json' | 'tools'

const props = defineProps<{
  model: IContainer | null
  chatId: string
  hasHistory: boolean
}>()

const emit = defineEmits<{
  (e: 'send', payload: { message: string, image: string }): void
  (e: 'stop'): void
  (e: 'imageAttached'): void
  (e: 'export', format: 'md' | 'json'): void
  (e: 'continueFresh'): void
}>()

const message = defineModel<string>('message', { default: '' })

const model = toRef(props, 'model')
const chatId = toRef(props, 'chatId')

const chatStore = useChatStore()
const { aiModels, chatContext } = storeToRefs(chatStore)

const { hardware } = storeToRefs(useHardwareStore())

const { waitingForResponse, websocket, isReconnecting, reconnectFailed } = storeToRefs(useChatSocketStore())

const current = useCurrentChat(model, chatId)
const { persona, generationParams, activeCollections, toolsEnabled, thinkLevel, memoryMode } = current
const {
  selectPersona,
  saveGenerationParams,
  saveMemory,
  setThinkLevel,
  toggleTools,
  updateActiveCollections,
} = useChatSettings(model, chatId, current)

const composerRef = ref<{ focus: () => void, clear: () => void } | null>(null)

const personaDialogOpen = ref(false)
const parameterPanelOpen = ref(false)
const memoryDialogOpen = ref(false)
const documentsDialogOpen = ref(false)
const promptTemplateDialogOpen = ref(false)
const jsonDialogOpen = ref(false)
const plusMenuOpen = ref(false)

const structuredOutputFormat = ref<{ field: string, type: string, description?: string, arrayType?: string }[]>([])
const isFormValid = ref(false)
const jsonEnforced = ref(false)

// What a reply must be shaped like, or null when the user hasn't switched
// JSON output on (or the form describing it is incomplete).
const structuredOutput = computed(() => (isFormValid.value && structuredOutputFormat.value.length > 0 && jsonEnforced.value
  ? structuredOutputFormat.value
  : null))

const LOW_RAM_BYTES = 16 * 1024 ** 3

const lowResources = computed(() => hardware.value.gpu_available === false
  || (hardware.value.ram_bytes !== null && hardware.value.ram_bytes < LOW_RAM_BYTES))

const embeddingModels = computed(() => aiModels.value.filter(m => m.is_embedding))

const modelRunning = computed(() => model.value?.status === 'running')

// The installed model's own report beats the ollama.com catalog, which has
// listed capable models without their "tools" chip.
function modelHas(capability: string, catalogFlag: boolean | undefined): boolean {
  const reported = chatContext.value?.capabilities
  if (reported?.length)
    return reported.includes(capability)

  return !!catalogFlag
}

const toolsAvailable = computed(() => modelHas('tools', model.value?.canUseTools))
const visionAvailable = computed(() => modelHas('vision', model.value?.canProcessImages))

const sendGateOk = computed(() => !!chatId.value && !!websocket.value && !isReconnecting.value && !reconnectFailed.value)

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

function insertTemplate(text: string) {
  message.value = text
  composerRef.value?.focus()
}

defineExpose({
  structuredOutput,
  focus: () => composerRef.value?.focus(),
  clear: () => composerRef.value?.clear(),
})
</script>

<template>
  <div class="composer-dock">
    <div class="composer-dock-inner">
      <div
        v-if="chatId"
        class="chip-row"
      >
        <MemoryMeter
          v-if="chatContext"
          :context="chatContext"
          @open="memoryDialogOpen = true"
        />

        <ChatSettingsChips
          :persona="persona"
          :params="generationParams"
          :active-collections-count="activeCollections.length"
          :tools-enabled="toolsEnabled"
          :json-enabled="jsonEnforced"
          :think-level="chatContext?.thinking
            ? thinkLevel
            : ''"
          @open="openDialog"
        />

        <ChatPlusMenu
          v-model="plusMenuOpen"
          :persona="persona"
          :params="generationParams"
          :active-collections-count="activeCollections.length"
          :tools-enabled="toolsEnabled"
          :tools-available="toolsAvailable"
          :thinking-controls="chatContext?.thinking ?? null"
          :think-level="thinkLevel"
          :json-enabled="jsonEnforced"
          :json-field-count="structuredOutputFormat.length"
          :has-history="hasHistory"
          @open="openDialog"
          @toggle-tools="toggleTools"
          @set-think-level="setThinkLevel"
          @export="emit('export', $event)"
        />
      </div>

      <ChatComposer
        ref="composerRef"
        v-model:message="message"
        :model-running="modelRunning"
        :waiting-for-response="waitingForResponse"
        :send-gate-ok="sendGateOk"
        :show-image-button="visionAvailable"
        @send="emit('send', $event)"
        @stop="emit('stop')"
        @image-attached="emit('imageAttached')"
      />
    </div>
  </div>

  <PersonaDialog
    v-model="personaDialogOpen"
    :current-persona-id="persona?.id ?? null"
    @select="selectPersona"
  />

  <ParameterPanelDialog
    v-model="parameterPanelOpen"
    :current-params="generationParams"
    @save="saveGenerationParams"
    @open-memory="memoryDialogOpen = true"
  />

  <MemoryDialog
    v-model="memoryDialogOpen"
    :context="chatContext"
    :current-num-ctx="generationParams.num_ctx"
    :memory-mode="memoryMode"
    :low-resources="lowResources"
    :has-history="hasHistory"
    @save="saveMemory"
    @continue-fresh="emit('continueFresh')"
  />

  <DocumentsDialog
    v-if="chatId"
    v-model="documentsDialogOpen"
    :chat-id="chatId"
    :active-collections="activeCollections"
    :embedding-models="embeddingModels"
    @update:active-collections="updateActiveCollections"
  />

  <PromptTemplateDialog
    v-model="promptTemplateDialogOpen"
    @insert="insertTemplate"
  />

  <StructuredOutputSelector
    v-model:is-open="jsonDialogOpen"
    v-model:format="structuredOutputFormat"
    v-model:is-form-valid="isFormValid"
    v-model:enforced="jsonEnforced"
  />
</template>

<style scoped>
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

@media (max-width: 600px) {
  .composer-dock {
    padding: 0 12px 14px;
  }
}
</style>
