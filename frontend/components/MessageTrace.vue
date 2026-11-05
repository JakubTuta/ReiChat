<script setup lang="ts">
import type { ToolCall } from '~/composables/websocket'

const props = defineProps<{
  thoughts: string
  toolCalls: ToolCall[]
}>()

const visible = ref(false)

const hasTrace = computed(() => !!props.thoughts || props.toolCalls.length > 0)

function formatArgs(args: Record<string, unknown>): string {
  return Object.values(args).map(v => JSON.stringify(v)).join(', ')
}

const summary = computed(() => {
  const parts: string[] = []
  if (props.thoughts)
    parts.push('thoughts')
  if (props.toolCalls.length) {
    parts.push(`${props.toolCalls.length} tool${props.toolCalls.length === 1
      ? ''
      : 's'}`)
  }

  return parts.join(' · ')
})
</script>

<template>
  <button
    v-if="hasTrace"
    type="button"
    class="trace-toggle font-mono"
    @click="visible = !visible"
  >
    {{ visible
      ? 'hide trace'
      : summary }}
  </button>

  <div
    v-if="hasTrace && visible"
    class="trace-panel font-mono"
  >
    <div
      v-for="(call, i) in toolCalls"
      :key="`tool-${i}`"
      class="trace-tool-row"
    >
      <span class="trace-tool-name">{{ call.name }}</span>({{ formatArgs(call.args) }}) → {{ call.result }}
    </div>

    <div
      v-if="thoughts"
      class="trace-thoughts"
    >
      {{ thoughts }}
    </div>
  </div>
</template>

<style scoped>
.trace-toggle {
  display: inline-block;
  margin-bottom: 8px;
  padding: 0;
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  font-size: 11px;
  cursor: pointer;
  transition: color 0.15s ease;
}

.trace-toggle:hover {
  color: var(--color-mint-deep);
}

.trace-panel {
  margin-bottom: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-soft-2);
  border: 1px solid var(--color-line-2);
  font-size: 11px;
  line-height: 1.85;
  color: var(--color-ink-2);
}

.trace-tool-row {
  white-space: pre-wrap;
  word-break: break-word;
}

.trace-tool-name {
  color: var(--color-mint-deep);
}

.trace-thoughts {
  white-space: pre-wrap;
  font-style: italic;
  opacity: 0.85;
}

.trace-thoughts:not(:first-child) {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid var(--color-line-2);
}
</style>
