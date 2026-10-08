<script setup lang="ts">
const isOpen = defineModel<boolean>({ default: false, required: true })

const containerStore = useContainerStore()
const { connection } = storeToRefs(containerStore)

const DEFAULT_URL = 'http://host.docker.internal:11434'

const mode = ref<'managed' | 'external'>('managed')
const url = ref(DEFAULT_URL)
const saving = ref(false)
const error = ref('')

watch(isOpen, (open) => {
  if (!open)
    return

  mode.value = connection.value.mode
  url.value = connection.value.url || DEFAULT_URL
  error.value = ''
})

async function save() {
  saving.value = true
  error.value = await containerStore.saveConnection(mode.value, url.value.trim())
  saving.value = false

  if (!error.value)
    isOpen.value = false
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
        <span class="reichat-dialog-title">Where models run</span>

        <button
          type="button"
          class="reichat-dialog-close"
          title="Close"
          aria-label="Close"
          @click="isOpen = false"
        >
          <v-icon size="16">
            mdi-close
          </v-icon>
        </button>
      </div>

      <v-card-text class="reichat-dialog-body">
        <div class="connection-options">
          <button
            type="button"
            class="connection-option"
            :class="{'connection-option--active': mode === 'managed'}"
            @click="mode = 'managed'"
          >
            <span class="connection-option-label">Let ReiChat manage it (recommended)</span>

            <span class="connection-option-desc">
              Each model runs in its own Docker container. Nothing else to install.
            </span>
          </button>

          <button
            type="button"
            class="connection-option"
            :class="{'connection-option--active': mode === 'external'}"
            @click="mode = 'external'"
          >
            <span class="connection-option-label">Use my own Ollama</span>

            <span class="connection-option-desc">
              Use the Ollama app you already have. On a Mac this is how models get to use the graphics chip,
              and anything you've already downloaded shows up here.
            </span>
          </button>
        </div>

        <template v-if="mode === 'external'">
          <v-text-field
            v-model="url"
            label="Ollama address"
            density="compact"
            class="mt-4"
            hint="Ollama on this computer: http://host.docker.internal:11434. On Linux, start Ollama with OLLAMA_HOST=0.0.0.0 so ReiChat can reach it."
            persistent-hint
          />
        </template>

        <p
          v-if="mode === 'managed' && connection.mode === 'external'"
          class="connection-note"
        >
          Models on your own Ollama stay there; switching back shows the app-managed models instead.
        </p>

        <p
          v-if="error"
          class="connection-error"
        >
          {{ error }}
        </p>
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
          :loading="saving"
          @click="save"
        >
          {{ mode === 'external'
            ? 'Connect'
            : 'Save' }}
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.connection-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.connection-option {
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

.connection-option--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
}

.connection-option-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.connection-option-desc {
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-ink-2);
}

.connection-note {
  margin: 12px 0 0;
  font-size: 12px;
  color: var(--color-ink-2);
}

.connection-error {
  margin: 12px 0 0;
  font-size: 12.5px;
  color: var(--color-red);
}
</style>
