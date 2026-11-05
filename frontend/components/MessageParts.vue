<script setup lang="ts">
import type { MessagePart } from '~/utils/splitMessage'
import { marked } from 'marked'

defineProps<{
  parts: MessagePart[]
}>()

marked.use({ gfm: true, breaks: true })

function renderMarkdown(text: string): string {
  return marked.parse(text) as string
}

const { copiedKey, copy } = useCopyFeedback()
</script>

<template>
  <div
    v-for="(part, partIndex) in parts"
    :key="partIndex"
  >
    <div
      v-if="part.title === 'text'"
      v-sanitize-html:reply="renderMarkdown(part.content)"
      class="markdown-body"
    />

    <div
      v-else-if="part.title === 'code'"
      class="code-block"
    >
      <div class="code-title">
        <span class="font-mono">{{ part.language || 'code' }}</span>

        <button
          type="button"
          class="code-copy font-mono"
          :title="copiedKey === `code-${partIndex}`
            ? 'Copied'
            : 'Copy code'"
          @click="copy(part.content, `code-${partIndex}`)"
        >
          {{ copiedKey === `code-${partIndex}`
            ? 'copied'
            : 'copy' }}
        </button>
      </div>

      <div class="code-text font-mono">
        {{ part.content }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.code-block {
  margin: 12px 0;
  border-radius: 11px;
  background: var(--color-soft);
  border: 1px solid var(--color-line-2);
  overflow: hidden;
}

.code-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-line-2);
  font-size: 11px;
  color: var(--color-ink-2);
}

.code-copy {
  border: none;
  background: transparent;
  color: var(--color-mint-deep);
  font-size: 11px;
  cursor: pointer;
  padding: 0;
}

.code-text {
  white-space: pre-wrap;
  padding: 12px;
  font-size: 12.5px;
  line-height: 1.8;
  color: var(--color-ink);
}

.markdown-body :deep(p) {
  margin: 0.3em 0;
}

.markdown-body :deep(p:first-child) {
  margin-top: 0;
}

.markdown-body :deep(p:last-child) {
  margin-bottom: 0;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  font-weight: 600;
  line-height: 1.3;
  margin: 0.75em 0 0.3em;
}

.markdown-body :deep(h1:first-child),
.markdown-body :deep(h2:first-child),
.markdown-body :deep(h3:first-child) {
  margin-top: 0;
}

.markdown-body :deep(h1) { font-size: 1.35em; }
.markdown-body :deep(h2) { font-size: 1.2em; }
.markdown-body :deep(h3) { font-size: 1.08em; }
.markdown-body :deep(h4) { font-size: 1em; }

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: 1.5em;
  margin: 0.4em 0;
}

.markdown-body :deep(li) {
  margin: 0.2em 0;
}

.markdown-body :deep(strong) {
  font-weight: 600;
}

.markdown-body :deep(em) {
  font-style: italic;
}

.markdown-body :deep(code) {
  font-family: var(--font-mono);
  font-size: 0.85em;
  background: var(--color-soft);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.markdown-body :deep(blockquote) {
  border-left: 3px solid var(--color-line-2);
  padding-left: 12px;
  margin: 0.5em 0;
  opacity: 0.85;
}

.markdown-body :deep(hr) {
  border: none;
  border-top: 1px solid rgba(127, 127, 127, 0.3);
  margin: 0.75em 0;
}

.markdown-body :deep(a) {
  color: inherit;
  text-decoration: underline;
  opacity: 0.9;
}
</style>
