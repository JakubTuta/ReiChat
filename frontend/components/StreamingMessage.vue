<script setup lang="ts">
import type { ToolCall } from '~/constants/websocket'
import type { IContainer } from '~/models/container'

const props = defineProps<{
  content: string
  toolCalls?: ToolCall[]
  notices?: string[]
  model: IContainer | null
}>()

const parts = computed(() => splitMessageRaw(props.content))
</script>

<template>
  <div
    class="assistant-message"
    aria-live="polite"
    aria-atomic="false"
  >
    <div class="assistant-header">
      <span class="assistant-dot assistant-dot--live" />

      <span
        v-if="model"
        class="assistant-model font-mono"
      >{{ model.model }}:{{ model.parameters }}</span>

      <MessageTrace
        :thoughts="parts.thoughts"
        :tool-calls="toolCalls ?? []"
      />
    </div>

    <MessageNotices :notices="notices ?? []" />

    <div class="assistant-body">
      <MessageParts :parts="parts.parts" />

      <span
        v-if="content"
        class="stream-caret"
      />
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
}

.assistant-dot--live {
  animation: pulse 1.6s ease-in-out infinite;
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

.stream-caret {
  display: inline-block;
  width: 2px;
  height: 1em;
  background-color: currentColor;
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 1.1s step-end infinite;
}
</style>
