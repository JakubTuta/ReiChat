<script setup lang="ts">
import type { Citation } from '~/constants/websocket'

const props = defineProps<{
  citation: Citation | null
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const documentStore = useDocumentStore()

const loading = ref(false)
const passage = ref<{ filename: string, page: number | null, content: string } | null>(null)
const error = ref('')

watch(isOpen, async (open) => {
  if (!open || !props.citation)
    return

  loading.value = true
  passage.value = null
  error.value = ''

  const result = await documentStore.fetchChunk(props.citation.document_id, props.citation.chunk_index)
  if ('error' in result)
    error.value = result.error
  else
    passage.value = result

  loading.value = false
})
</script>

<template>
  <v-dialog
    v-model="isOpen"
    max-width="560"
    transition="dialog-rise-transition"
  >
    <v-card class="reichat-dialog">
      <div class="reichat-dialog-header">
        <span class="reichat-dialog-title">
          {{ citation?.filename }}<template v-if="citation?.page">, page {{ citation.page }}</template>
        </span>

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
        <p class="citation-hint">
          This is the passage from your file that the model was shown when writing its answer.
        </p>

        <v-progress-linear
          v-if="loading"
          indeterminate
          color="mint"
        />

        <p
          v-else-if="error"
          class="citation-error"
        >
          {{ error }}
        </p>

        <blockquote
          v-else-if="passage"
          class="citation-passage"
        >
          {{ passage.content }}
        </blockquote>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.citation-hint {
  margin: 0 0 10px;
  font-size: 12.5px;
  color: var(--color-ink-2);
}

.citation-error {
  font-size: 13px;
  color: var(--color-red);
}

.citation-passage {
  margin: 0;
  padding: 12px 14px;
  max-height: 50vh;
  overflow-y: auto;
  border-left: 3px solid var(--color-mint);
  border-radius: 0 8px 8px 0;
  background: var(--color-soft-2);
  white-space: pre-wrap;
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--color-ink);
}
</style>
