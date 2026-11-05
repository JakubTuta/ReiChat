<script setup lang="ts">
const emit = defineEmits<{
  (e: 'added', value: { model: string, parameters: string }): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const chatStore = useChatStore()

const reference = ref('')
const saving = ref(false)

const EXAMPLES = [
  'llama3.2:1b',
  'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M',
]

watch(isOpen, (open) => {
  if (open)
    reference.value = ''
})

async function add() {
  const value = reference.value.trim()
  if (!value)
    return

  saving.value = true
  const added = await chatStore.addCustomModel(value)
  saving.value = false

  if (added) {
    isOpen.value = false
    emit('added', added)
  }
}
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="520"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">Add a model by name</span>

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
        <p class="add-model-intro">
          Install any model Ollama can download, even if it isn't in the list. That includes thousands
          from Hugging Face: open a model page with "GGUF" in its name, copy its name and put
          <code>hf.co/</code> in front of it.
        </p>

        <v-text-field
          v-model="reference"
          label="Model name"
          placeholder="hf.co/user/model-GGUF:Q4_K_M"
          density="compact"
          autofocus
          hint="After the colon: a size (8b) or, for Hugging Face, a quantization like Q4_K_M — smaller numbers download less but answer a bit worse."
          persistent-hint
          @keydown.enter="add"
        />

        <div class="add-model-examples">
          Examples:
          <button
            v-for="example in EXAMPLES"
            :key="example"
            type="button"
            class="add-model-example font-mono"
            @click="reference = example"
          >
            {{ example }}
          </button>
        </div>
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
          :disabled="!reference.trim()"
          @click="add"
        >
          Add and install
        </v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.add-model-intro {
  margin: 0 0 14px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
}

.add-model-examples {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 14px;
  font-size: 12px;
  color: var(--color-ink-3);
}

.add-model-example {
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  color: var(--color-ink-2);
  font-size: 11px;
  cursor: pointer;
}
</style>
