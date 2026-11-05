<script setup lang="ts">
import type { IChatContext, MemoryMode } from '~/stores/chatStore'
import { DEFAULT_PRESET_TOKENS, formatPages, MEMORY_PRESETS, memoryPercent } from '~/utils/memory'

const props = defineProps<{
  context: IChatContext | null
  // The chat's saved choice — null means the default ("Standard").
  currentNumCtx: number | null
  memoryMode: MemoryMode
  // No usable GPU, or little RAM: bigger presets get a gentle warning.
  lowResources: boolean
  // Nothing to summarize in an empty chat.
  hasHistory: boolean
}>()

const emit = defineEmits<{
  (e: 'save', value: { num_ctx: number | null, memory_mode: MemoryMode }): void
  (e: 'continueFresh'): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const selectedTokens = ref(DEFAULT_PRESET_TOKENS)
const selectedMode = ref<MemoryMode>('trim')
const customTokens = ref<number | null>(null)
const showAdvanced = ref(false)

const modelMax = computed(() => props.context?.model_max ?? null)

watch(isOpen, (open) => {
  if (!open)
    return

  const current = props.currentNumCtx ?? DEFAULT_PRESET_TOKENS
  const isPreset = MEMORY_PRESETS.some(p => p.tokens === current)
  selectedTokens.value = current
  customTokens.value = isPreset
    ? null
    : current
  showAdvanced.value = !isPreset
  selectedMode.value = props.memoryMode
})

function isTooBig(tokens: number): boolean {
  return modelMax.value !== null && tokens > modelMax.value
}

function choosePreset(tokens: number) {
  if (isTooBig(tokens))
    return
  selectedTokens.value = tokens
  customTokens.value = null
}

const effectiveTokens = computed(() => customTokens.value ?? selectedTokens.value)

const customError = computed(() => {
  const value = customTokens.value
  if (value === null)
    return ''
  if (!Number.isInteger(value) || value < 1024)
    return 'Use a whole number of at least 1024.'
  if (isTooBig(value))
    return `This model supports at most ${modelMax.value} tokens.`

  return ''
})

const usageLine = computed(() => {
  if (!props.context)
    return ''

  return `Right now this chat uses ${formatPages(props.context.used)} of ${formatPages(props.context.limit)} `
    + `(${memoryPercent(props.context)}%).`
})

function save() {
  if (customError.value)
    return

  emit('save', {
    num_ctx: effectiveTokens.value === DEFAULT_PRESET_TOKENS && customTokens.value === null
      ? null
      : effectiveTokens.value,
    memory_mode: selectedMode.value,
  })
  isOpen.value = false
}

function continueFresh() {
  isOpen.value = false
  emit('continueFresh')
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="540"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Memory</span>

        <button
          type="button"
          class="reichat-dialog-close"
          title="Close"
          aria-label="Close"
          @click="isOpen = false"
        >
          <v-icon
            size="16"
            icon="$mdiClose"
          />
        </button>
      </div>

      <v-card-text class="reichat-dialog-body">
        <p class="memory-intro">
          How much of this conversation the model can keep in mind at once. Everything older than
          that is invisible to it — unless it's summarized.
        </p>

        <p
          v-if="usageLine"
          class="memory-usage"
        >
          {{ usageLine }}
        </p>

        <div class="memory-section-title">
          How much it remembers
        </div>

        <div class="memory-options">
          <button
            v-for="preset in MEMORY_PRESETS"
            :key="preset.id"
            type="button"
            class="memory-option"
            :class="{
              'memory-option--active': customTokens === null && selectedTokens === preset.tokens,
              'memory-option--disabled': isTooBig(preset.tokens),
            }"
            :disabled="isTooBig(preset.tokens)"
            @click="choosePreset(preset.tokens)"
          >
            <span class="memory-option-head">
              <span class="memory-option-label">{{ preset.label }}</span>

              <span class="memory-option-size">{{ formatPages(preset.tokens) }}</span>
            </span>

            <span class="memory-option-desc">
              <template v-if="isTooBig(preset.tokens)">
                This model can't remember that much.
              </template>

              <template v-else>
                {{ preset.description }}
                <template v-if="lowResources && preset.tokens > DEFAULT_PRESET_TOKENS">
                  May be slow on this computer.
                </template>
              </template>
            </span>
          </button>
        </div>

        <div
          v-if="modelMax"
          class="memory-hint"
        >
          This model can remember up to {{ formatPages(modelMax) }}.
        </div>

        <div class="memory-section-title">
          When the conversation gets longer than that
        </div>

        <div class="memory-options memory-options--two">
          <button
            type="button"
            class="memory-option"
            :class="{'memory-option--active': selectedMode === 'trim'}"
            @click="selectedMode = 'trim'"
          >
            <span class="memory-option-label">Forget the oldest messages</span>

            <span class="memory-option-desc">Fastest. The model only sees the most recent part of the chat.</span>
          </button>

          <button
            type="button"
            class="memory-option"
            :class="{'memory-option--active': selectedMode === 'summarize'}"
            @click="selectedMode = 'summarize'"
          >
            <span class="memory-option-label">Summarize older messages</span>

            <span class="memory-option-desc">
              Keeps the gist of everything. Now and then a reply takes a few extra seconds while it summarizes.
            </span>
          </button>
        </div>

        <button
          v-if="hasHistory"
          type="button"
          class="memory-fresh"
          @click="continueFresh"
        >
          <v-icon
            size="16"
            icon="$mdiFileDocumentRefreshOutline"
          />

          <span>
            <strong>Start a fresh chat with a summary</strong>

            <span class="memory-option-desc">
              A new chat with the same settings that begins knowing what this one was about.
            </span>
          </span>
        </button>

        <button
          type="button"
          class="memory-advanced-toggle"
          @click="showAdvanced = !showAdvanced"
        >
          {{ showAdvanced
            ? 'Hide advanced'
            : 'Advanced: exact size' }}
        </button>

        <v-text-field
          v-if="showAdvanced"
          v-model.number="customTokens"
          type="number"
          label="Context size in tokens (num_ctx)"
          min="1024"
          step="1024"
          density="compact"
          :error-messages="customError"
          hint="Leave empty to use the option selected above."
          persistent-hint
          clearable
        />
      </v-card-text>

      <v-card-actions class="reichat-dialog-actions">
        <v-spacer />

        <v-btn
          variant="text"
          @click="isOpen = false"
        >
          Cancel
        </v-btn>

        <v-btn
          color="mint-btn"
          variant="flat"
          :disabled="!!customError"
          @click="save"
        >
          Save
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.memory-intro {
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
  margin: 0 0 6px;
}

.memory-usage {
  font-size: 12.5px;
  color: var(--color-mint-deep);
  margin: 0 0 14px;
}

.memory-section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-ink);
  margin: 14px 0 8px;
}

.memory-options {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.memory-option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  text-align: left;
  font-family: var(--font-sans);
  cursor: pointer;
}

.memory-option--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
}

.memory-option--disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.memory-option-head {
  display: flex;
  justify-content: space-between;
  width: 100%;
  gap: 6px;
}

.memory-option-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.memory-option-size {
  font-size: 11px;
  color: var(--color-mint-deep);
}

.memory-option-desc {
  display: block;
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--color-ink-2);
}

.memory-hint {
  margin-top: 6px;
  font-size: 11.5px;
  color: var(--color-ink-3);
}

.memory-fresh {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  width: 100%;
  margin-top: 16px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px dashed var(--color-line-dash);
  background: transparent;
  text-align: left;
  font-family: var(--font-sans);
  font-size: 13px;
  color: var(--color-ink);
  cursor: pointer;
}

.memory-fresh:hover {
  border-color: var(--color-mint-border);
}

.memory-advanced-toggle {
  margin: 14px 0 8px;
  border: none;
  background: transparent;
  padding: 0;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--color-ink-3);
  cursor: pointer;
}

@media (max-width: 480px) {
  .memory-options {
    grid-template-columns: 1fr;
  }
}
</style>
