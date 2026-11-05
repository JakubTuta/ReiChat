<script setup lang="ts">
import type { IGenerationParams } from '~/stores/chatStore'

const props = defineProps<{
  currentParams: IGenerationParams
}>()

const emit = defineEmits<{
  // num_ctx isn't edited here — it's the "Memory" setting (MemoryDialog).
  (e: 'save', params: Omit<IGenerationParams, 'num_ctx'>): void
  (e: 'openMemory'): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const DEFAULT_TEMPERATURE = 0.8
const DEFAULT_TOP_P = 0.9

const temperature = ref<number | null>(null)
const topP = ref<number | null>(null)
const seed = ref<number | null>(null)

watch(isOpen, (open) => {
  if (!open)
    return

  temperature.value = props.currentParams.temperature
  topP.value = props.currentParams.top_p
  seed.value = props.currentParams.seed
})

// Each field is either null ("use Ollama's own default") or a concrete
// override — the switch is what the user actually toggles; flipping it on
// seeds a sane starting value rather than leaving the slider at 0.
const temperatureEnabled = computed({
  get: () => temperature.value !== null,
  set: (enabled: boolean) => {
    temperature.value = enabled
      ? DEFAULT_TEMPERATURE
      : null
  },
})
const topPEnabled = computed({
  get: () => topP.value !== null,
  set: (enabled: boolean) => {
    topP.value = enabled
      ? DEFAULT_TOP_P
      : null
  },
})
const seedEnabled = computed({
  get: () => seed.value !== null,
  set: (enabled: boolean) => {
    seed.value = enabled
      ? 0
      : null
  },
})

// v-slider can't bind to null (it's disabled when not enabled anyway), so
// these fall back to the default just for display purposes.
const temperatureSliderValue = computed({
  get: () => temperature.value ?? DEFAULT_TEMPERATURE,
  set: (value: number) => { temperature.value = value },
})
const topPSliderValue = computed({
  get: () => topP.value ?? DEFAULT_TOP_P,
  set: (value: number) => { topP.value = value },
})

function save() {
  emit('save', {
    temperature: temperature.value,
    top_p: topP.value,
    seed: seed.value,
  })

  isOpen.value = false
}

function resetAll() {
  temperature.value = null
  topP.value = null
  seed.value = null
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="470"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Parameters</span>

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
        <div class="param-hint">
          Overrides for this chat only. Off means Ollama's own default for the model.
        </div>

        <div class="param-row">
          <div class="param-row-header">
            <div>
              <span class="param-row-label">Temperature</span>

              <span class="param-row-value font-mono">
                {{ temperatureEnabled
                  ? temperatureSliderValue.toFixed(2)
                  : 'default' }}
              </span>
            </div>

            <v-switch
              v-model="temperatureEnabled"
              density="compact"
              hide-details
              color="mint-btn"
            />
          </div>

          <v-slider
            v-model="temperatureSliderValue"
            :disabled="!temperatureEnabled"
            :min="0"
            :max="2"
            :step="0.05"
            color="mint"
            thumb-label
            hide-details
          />
        </div>

        <div class="param-row">
          <div class="param-row-header">
            <div>
              <span class="param-row-label">Top P</span>

              <span class="param-row-value font-mono">
                {{ topPEnabled
                  ? topPSliderValue.toFixed(2)
                  : 'default' }}
              </span>
            </div>

            <v-switch
              v-model="topPEnabled"
              density="compact"
              hide-details
              color="mint-btn"
            />
          </div>

          <v-slider
            v-model="topPSliderValue"
            :disabled="!topPEnabled"
            :min="0"
            :max="1"
            :step="0.05"
            color="mint"
            thumb-label
            hide-details
          />
        </div>

        <button
          type="button"
          class="param-memory-link"
          @click="isOpen = false; emit('openMemory')"
        >
          Looking for context length? It's under <strong>Memory</strong> now.
        </button>

        <div class="param-row">
          <div class="param-row-header">
            <div>
              <span class="param-row-label">Seed</span>
            </div>

            <v-switch
              v-model="seedEnabled"
              density="compact"
              hide-details
              color="mint-btn"
            />
          </div>

          <v-text-field
            v-model.number="seed"
            :disabled="!seedEnabled"
            type="number"
            step="1"
            density="compact"
            hide-details
            hint="Same seed + same prompt = reproducible output"
            persistent-hint
          />
        </div>
      </v-card-text>

      <v-card-actions class="reichat-dialog-actions">
        <v-btn
          variant="text"
          @click="resetAll"
        >
          Reset all to default
        </v-btn>

        <v-spacer />

        <v-btn
          color="mint-btn"
          variant="flat"
          @click="save"
        >
          Save
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.param-hint {
  font-size: 12.5px;
  color: var(--color-ink-2);
  margin-bottom: 14px;
}

.param-row {
  margin-bottom: 16px;
}

.param-row-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.param-row-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.param-memory-link {
  display: block;
  margin-bottom: 16px;
  padding: 0;
  border: none;
  background: transparent;
  font-family: var(--font-sans);
  font-size: 12.5px;
  color: var(--color-mint-deep);
  cursor: pointer;
  text-align: left;
}

.param-row-value {
  margin-left: 8px;
  font-size: 11px;
  color: var(--color-mint-deep);
}
</style>
