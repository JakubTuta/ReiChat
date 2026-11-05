<script setup lang="ts">
const isShow = defineModel<boolean>('isShow', { default: false })

const chatStore = useChatStore()
const { loading } = storeToRefs(chatStore)

const containerStore = useContainerStore()
const { scrapeProgress } = storeToRefs(containerStore)

const minPullCount = ref(200000)

// A raw "minimum popularity threshold" number was the first thing a new
// user had to figure out. Presets cover the common cases; the field below
// is still there for anyone who wants an exact number.
const presets = [
  { label: 'Popular', value: 1_000_000 },
  { label: 'Recommended', value: 200_000 },
  { label: 'All models', value: 0 },
] as const

const activePreset = computed(() => presets.find(p => p.value === minPullCount.value)?.label ?? null)

const scrapePercent = computed(() => {
  if (!scrapeProgress.value.total)
    return 0

  return Math.round((scrapeProgress.value.completed / scrapeProgress.value.total) * 100)
})

const scrapeLabel = computed(() => {
  if (!scrapeProgress.value.running)
    return ''
  if (!scrapeProgress.value.total)
    return 'Starting scrape…'

  return `Scraped ${scrapeProgress.value.completed} of ${scrapeProgress.value.total} models${scrapeProgress.value.current
    ? ` — ${scrapeProgress.value.current}`
    : ''}`
})

async function pullModels() {
  await chatStore.pullAIModels(minPullCount.value)
}

// The refresh runs in a background thread regardless of whether this dialog
// is open — the "runs in background" copy below used to be contradicted by
// a `persistent` dialog the user couldn't leave until it finished, and
// nothing closed it automatically once it did.
watch(
  () => scrapeProgress.value.running,
  (running, wasRunning) => {
    if (wasRunning && !running && !scrapeProgress.value.error)
      isShow.value = false
  },
)
</script>

<template>
  <v-dialog
    v-model="isShow"
    max-width="520"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Refresh model list</span>

        <button
          type="button"
          class="reichat-dialog-close"
          title="Close"
          aria-label="Close"
          @click="isShow = false"
        >
          <v-icon
            size="16"
            icon="$mdiClose"
          />
        </button>
      </div>

      <v-card-text class="reichat-dialog-body">
        <div class="preset-row">
          <button
            v-for="preset in presets"
            :key="preset.label"
            type="button"
            class="preset-chip"
            :class="{'preset-chip--active': activePreset === preset.label}"
            :disabled="loading || scrapeProgress.running"
            @click="minPullCount = preset.value"
          >
            {{ preset.label }}
          </button>
        </div>

        <v-text-field
          v-model.number="minPullCount"
          label="Minimum popularity threshold (advanced)"
          hint="Only models with at least this many downloads on ollama.com will be imported."
          persistent-hint
          density="compact"
          :disabled="loading || scrapeProgress.running"
        />

        <div class="info-note mt-4">
          Fetching the catalogue runs in the background — you can close this dialog and keep
          browsing.
        </div>

        <template v-if="scrapeProgress.running">
          <div class="scrape-label mt-4">
            {{ scrapeLabel }}
          </div>

          <v-progress-linear
            class="mt-2"
            :model-value="scrapePercent"
            :indeterminate="!scrapeProgress.total"
            color="mint"
            rounded
            height="4"
          />
        </template>

        <div
          v-if="scrapeProgress.error"
          class="error-note mt-4"
        >
          {{ scrapeProgress.error }}
        </div>
      </v-card-text>

      <v-card-actions class="reichat-dialog-actions">
        <v-btn
          variant="text"
          @click="isShow = false"
        >
          {{ scrapeProgress.running
            ? 'Close (keeps running)'
            : 'Close' }}
        </v-btn>

        <v-spacer />

        <v-btn
          color="mint-btn"
          variant="flat"
          :loading="loading || scrapeProgress.running"
          :disabled="loading || scrapeProgress.running"
          @click="pullModels"
        >
          {{ scrapeProgress.running
            ? 'Fetching…'
            : 'Fetch model list' }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.preset-row {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.preset-chip {
  padding: 6px 13px;
  border-radius: 20px;
  font-family: var(--font-sans);
  font-size: 12.5px;
  font-weight: 500;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}

.preset-chip:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.preset-chip--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.info-note {
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-soft);
  color: var(--color-ink-2);
  font-size: 12.5px;
}

.scrape-label {
  font-size: 12.5px;
  color: var(--color-ink-2);
}

.error-note {
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-banner-bg);
  border: 1px solid var(--color-banner-border);
  color: var(--color-red);
  font-size: 12.5px;
}
</style>
