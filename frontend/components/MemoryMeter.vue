<script setup lang="ts">
import type { IChatContext } from '~/stores/chatStore'
import { formatPages, memoryLevel, memoryPercent } from '~/utils/memory'

const props = defineProps<{
  context: IChatContext
}>()

const emit = defineEmits<{
  (e: 'open'): void
}>()

const percent = computed(() => memoryPercent(props.context))
const level = computed(() => memoryLevel(props.context))

const label = computed(() => {
  if (props.context.summarized)
    return 'Memory · summarizing'
  if (props.context.dropped > 0)
    return 'Memory full'

  return `Memory ${percent.value}%`
})

const tooltip = computed(() => {
  const capacity = formatPages(props.context.limit)
  const used = formatPages(props.context.used)

  if (props.context.dropped > 0) {
    return `This chat no longer fits in the model's memory (${capacity}), so it can't see the oldest `
      + `${props.context.dropped} messages. Click to give it more memory or summarize older messages.`
  }
  if (props.context.summarized)
    return `Older messages are kept as a summary so the model remembers the gist. Memory: ${capacity}.`

  return `The model can keep ${capacity} of this chat in mind; it's using ${used} so far. Click to change.`
})
</script>

<template>
  <button
    type="button"
    class="memory-meter"
    :class="`memory-meter--${level}`"
    :aria-label="tooltip"
    @click="emit('open')"
  >
    <span class="memory-meter-track">
      <span
        class="memory-meter-fill"
        :style="{'width': `${percent}%`}"
      />
    </span>

    <span class="memory-meter-label">{{ label }}</span>

    <v-tooltip
      activator="parent"
      location="top"
      max-width="320"
    >
      {{ tooltip }}
    </v-tooltip>
  </button>
</template>

<style scoped>
.memory-meter {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 11px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-family: var(--font-sans);
  font-size: 11.5px;
  cursor: pointer;
  white-space: nowrap;
}

.memory-meter:hover {
  border-color: var(--color-mint-border);
}

.memory-meter-track {
  position: relative;
  width: 42px;
  height: 5px;
  border-radius: 3px;
  background: var(--color-soft-2);
  overflow: hidden;
}

.memory-meter-fill {
  position: absolute;
  inset: 0 auto 0 0;
  border-radius: 3px;
  background: var(--color-mint);
  transition: width 0.3s ease;
}

.memory-meter--filling .memory-meter-fill {
  background: var(--color-amber);
}

.memory-meter--full .memory-meter-fill {
  background: var(--color-red);
}

.memory-meter--full {
  color: var(--color-red);
}
</style>
