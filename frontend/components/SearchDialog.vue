<script setup lang="ts">
import type { ISearchResult } from '~/stores/chatStore'

const props = defineProps<{
  model: string
}>()

const emit = defineEmits<{
  (e: 'select', chatId: string): void
}>()

const isOpen = defineModel<boolean>({ default: false, required: true })

const chatStore = useChatStore()

const query = ref('')
const results = ref<ISearchResult[]>([])
const loading = ref(false)
const searched = ref(false)

let debounceTimer: ReturnType<typeof setTimeout> | null = null

watch(query, (value) => {
  if (debounceTimer)
    clearTimeout(debounceTimer)

  if (value.trim().length < 2) {
    results.value = []
    searched.value = false

    return
  }

  debounceTimer = setTimeout(async () => {
    loading.value = true
    results.value = await chatStore.searchChats(props.model, value.trim())
    searched.value = true
    loading.value = false
  }, 300)
})

watch(isOpen, (open) => {
  if (!open) {
    query.value = ''
    results.value = []
    searched.value = false
  }
})

function selectResult(result: ISearchResult) {
  emit('select', result.chat_id)
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
        <span class="reichat-dialog-title">Search chats</span>

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
        <v-text-field
          v-model="query"
          autofocus
          density="compact"
          placeholder="Search message content…"
          prepend-inner-icon="$mdiMagnify"
          clearable
          hide-details
        />

        <v-progress-linear
          v-if="loading"
          indeterminate
          color="mint"
          class="mt-2"
        />

        <div
          v-if="results.length"
          class="search-results"
        >
          <div
            v-for="result in results"
            :key="`${result.chat_id}-${result.snippet}`"
            class="search-result-row"
            @click="selectResult(result)"
          >
            <span class="search-result-title">{{ result.chat_title }}</span>

            <span class="search-result-snippet">
              <span class="mono-kicker search-result-marker">{{ result.role === 'user'
                ? 'USER'
                : 'ASSISTANT' }}</span>
              {{ result.snippet }}
            </span>
          </div>
        </div>

        <div
          v-else-if="searched && !loading"
          class="search-empty"
        >
          No matches found.
        </div>
      </v-card-text>
    </v-card>
  </v-dialog>
</template>

<style scoped>
.search-results {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.search-result-row {
  padding: 9px 11px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.search-result-row:hover {
  background: var(--color-soft);
}

.search-result-title {
  display: block;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink);
  margin-bottom: 2px;
}

.search-result-snippet {
  display: flex;
  align-items: baseline;
  gap: 6px;
  font-size: 11.5px;
  color: var(--color-ink-2);
  white-space: pre-wrap;
}

.search-result-marker {
  flex-shrink: 0;
  color: var(--color-mint-deep);
}

.search-empty {
  padding: 16px 4px;
  text-align: center;
  font-size: 12.5px;
  color: var(--color-ink-2);
}
</style>
