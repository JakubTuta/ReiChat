<script setup lang="ts">
import type { Citation } from '~/constants/websocket'
import type { IContainer } from '~/models/container'
import type { IChatMessage } from '~/stores/chatStore'

const props = defineProps<{
  message: IChatMessage
  index: number
  isLast: boolean
  editing: boolean
  // Gates regenerate/edit/branch-switch: mirrors the composer's
  // "model running and not mid-stream" condition — never true while a
  // response is in flight, since the underlying data could shift under it.
  actionsEnabled: boolean
  model: IContainer | null
}>()

const emit = defineEmits<{
  (e: 'regenerate'): void
  (e: 'startEdit', index: number): void
  (e: 'requestSaveEdit', payload: { index: number, content: string }): void
  (e: 'cancelEdit'): void
  (e: 'switchBranch', payload: { index: number, siblingIndex: number }): void
  (e: 'moreMemory', tokens: number): void
  (e: 'lessMemory', tokens: number): void
  (e: 'summarizeMemory'): void
  (e: 'continueFresh'): void
}>()

const splitContent = computed(() => splitMessageRaw(props.message.content))

const { copiedKey, copy } = useCopyFeedback()

const editingContent = ref('')

watch(() => props.editing, (isEditing) => {
  if (isEditing)
    editingContent.value = props.message.content
})

function requestSaveEdit() {
  const trimmed = editingContent.value.trim()
  if (!trimmed)
    return

  emit('requestSaveEdit', { index: props.index, content: trimmed })
}

function switchToPrevBranch() {
  emit('switchBranch', { index: props.index, siblingIndex: (props.message.sibling_index ?? 0) - 1 })
}

function switchToNextBranch() {
  emit('switchBranch', { index: props.index, siblingIndex: (props.message.sibling_index ?? 0) + 1 })
}

const hasBranches = computed(() => (props.message.sibling_count ?? 0) > 1)

// One chip per file+page, not per chunk — several neighbouring chunks from
// the same page would otherwise read as duplicate sources.
const sources = computed(() => {
  const seen = new Set<string>()

  return (props.message.citations ?? []).filter((citation) => {
    const key = `${citation.document_id}:${citation.page ?? ''}`
    if (seen.has(key))
      return false
    seen.add(key)

    return true
  })
})

const citationDialogOpen = ref(false)
const openCitation = ref<Citation | null>(null)

function showCitation(citation: Citation) {
  openCitation.value = citation
  citationDialogOpen.value = true
}
</script>

<template>
  <div
    v-if="message.role === 'assistant'"
    class="assistant-message"
  >
    <div class="assistant-header">
      <span class="assistant-dot" />

      <span
        v-if="model"
        class="assistant-model font-mono"
      >{{ model.model }}:{{ model.parameters }}</span>

      <MessageTrace
        :thoughts="message.thinking || splitContent.thoughts"
        :tool-calls="message.tool_calls ?? []"
      />
    </div>

    <MessageNotices :notices="message.notices ?? []" />

    <MemoryNotice
      v-if="message.context"
      :context="message.context"
      :actionable="isLast && actionsEnabled"
      @more-memory="(tokens: number) => emit('moreMemory', tokens)"
      @less-memory="(tokens: number) => emit('lessMemory', tokens)"
      @summarize="emit('summarizeMemory')"
      @continue-fresh="emit('continueFresh')"
    />

    <div class="assistant-body">
      <MessageParts :parts="splitContent.parts" />
    </div>

    <p
      v-if="message.image"
      class="assistant-image"
    >
      <img
        :src="message.image"
        alt="Uploaded image"
        style="max-width: 100%; max-height: 160px; border-radius: 10px"
      >
    </p>

    <div class="assistant-footer font-mono">
      <span
        v-if="sources.length"
        class="footer-citations"
      >
        Sources:
        <button
          v-for="citation in sources"
          :key="`${citation.document_id}-${citation.chunk_index}`"
          type="button"
          class="citation-chip"
          title="See the passage the model used"
          @click="showCitation(citation)"
        >
          {{ citation.filename }}<template v-if="citation.page"> p.{{ citation.page }}</template>
        </button>
      </span>

      <CitationDialog
        v-model="citationDialogOpen"
        :citation="openCitation"
      />

      <span
        v-if="message.stats"
        class="footer-stats"
      >{{ formatStats(message.stats) }}</span>

      <span
        v-if="hasBranches"
        class="branch-switcher"
      >
        <button
          type="button"
          class="branch-btn"
          :disabled="!actionsEnabled || (message.sibling_index ?? 0) === 0"
          title="Previous branch"
          aria-label="Previous branch"
          @click="switchToPrevBranch"
        >
          ‹
        </button>
        {{ (message.sibling_index ?? 0) + 1 }}/{{ message.sibling_count }}
        <button
          type="button"
          class="branch-btn"
          :disabled="!actionsEnabled || (message.sibling_index ?? 0) >= (message.sibling_count ?? 1) - 1"
          title="Next branch"
          aria-label="Next branch"
          @click="switchToNextBranch"
        >
          ›
        </button>
      </span>

      <span class="footer-spacer" />

      <button
        type="button"
        class="footer-action"
        :title="copiedKey === 'message'
          ? 'Copied'
          : 'Copy message'"
        @click="copy(message.content, 'message')"
      >
        {{ copiedKey === 'message'
          ? 'copied'
          : 'copy' }}
      </button>

      <button
        v-if="isLast && actionsEnabled"
        type="button"
        class="footer-action"
        title="Regenerate response"
        @click="emit('regenerate')"
      >
        regenerate
      </button>
    </div>
  </div>

  <div
    v-else
    class="user-message"
  >
    <div class="user-bubble">
      <template v-if="editing">
        <v-textarea
          v-model="editingContent"
          density="compact"
          auto-grow
          rows="1"
          hide-details
          autofocus
          class="edit-message-textarea"
          @keydown.enter.exact.prevent="requestSaveEdit"
          @keydown.esc="emit('cancelEdit')"
        />

        <div class="edit-message-actions">
          <v-btn
            size="small"
            variant="text"
            @click="emit('cancelEdit')"
          >
            Cancel
          </v-btn>

          <v-btn
            size="small"
            variant="flat"
            color="mint-btn"
            :disabled="!editingContent.trim()"
            @click="requestSaveEdit"
          >
            Save & resend
          </v-btn>
        </div>
      </template>

      <template v-else>
        {{ message.content }}
      </template>

      <p
        v-if="message.image"
        class="user-image"
      >
        <img
          :src="message.image"
          alt="Uploaded image"
          style="max-width: 100%; max-height: 140px; border-radius: 8px"
        >
      </p>
    </div>

    <div class="user-footer font-mono">
      <span
        v-if="hasBranches"
        class="branch-switcher"
      >
        <button
          type="button"
          class="branch-btn"
          :disabled="!actionsEnabled || (message.sibling_index ?? 0) === 0"
          title="Previous branch"
          aria-label="Previous branch"
          @click="switchToPrevBranch"
        >
          ‹
        </button>
        {{ (message.sibling_index ?? 0) + 1 }}/{{ message.sibling_count }}
        <button
          type="button"
          class="branch-btn"
          :disabled="!actionsEnabled || (message.sibling_index ?? 0) >= (message.sibling_count ?? 1) - 1"
          title="Next branch"
          aria-label="Next branch"
          @click="switchToNextBranch"
        >
          ›
        </button>
      </span>

      <span class="footer-spacer" />

      <button
        type="button"
        class="footer-action"
        :title="copiedKey === 'message'
          ? 'Copied'
          : 'Copy message'"
        @click="copy(message.content, 'message')"
      >
        {{ copiedKey === 'message'
          ? 'copied'
          : 'copy' }}
      </button>

      <button
        v-if="!editing && actionsEnabled"
        type="button"
        class="footer-action"
        title="Edit and resend"
        @click="emit('startEdit', index)"
      >
        edit & resend
      </button>
    </div>
  </div>
</template>

<style scoped>
.assistant-message {
  width: 100%;
}

.assistant-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.assistant-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--color-mint);
  flex-shrink: 0;
}

.assistant-model {
  font-size: 11px;
  color: var(--color-mint-deep);
}

.assistant-body {
  font-size: 15.5px;
  line-height: 1.75;
  color: var(--color-ink);
}

.assistant-footer {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 8px;
  font-size: 10.5px;
  color: var(--color-ink-3);
  flex-wrap: wrap;
}

.footer-citations {
  color: var(--color-mint-deep);
}

.citation-chip {
  margin-left: 4px;
  padding: 1px 7px;
  border-radius: 10px;
  border: 1px solid var(--color-mint-border);
  background: var(--color-mint-tint);
  color: var(--color-mint-ink);
  font-family: var(--font-mono);
  font-size: 10.5px;
  cursor: pointer;
}

.footer-spacer {
  flex: 1;
}

.footer-action {
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  font-family: var(--font-mono);
  font-size: 10.5px;
  cursor: pointer;
  padding: 0;
}

.footer-action:hover {
  color: var(--color-mint-deep);
}

.branch-switcher {
  display: flex;
  align-items: center;
  gap: 4px;
  font-variant-numeric: tabular-nums;
}

.branch-btn {
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  cursor: pointer;
  padding: 0 2px;
  font-size: 13px;
  line-height: 1;
}

.branch-btn:hover:not(:disabled) {
  color: var(--color-mint-deep);
}

.branch-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.user-message {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-left: 64px;
}

.user-bubble {
  background: var(--color-card);
  border: 1px solid var(--color-line-2);
  border-radius: 14px 14px 4px 14px;
  padding: 13px 16px;
  font-size: 14.5px;
  line-height: 1.6;
  color: var(--color-ink);
  word-break: break-word;
  max-width: 100%;
}

.user-image {
  margin: 8px 0 0;
  text-align: right;
}

.user-footer {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 4px;
  font-size: 10.5px;
}

.edit-message-textarea {
  min-width: 240px;
}

.edit-message-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 8px;
}
</style>
