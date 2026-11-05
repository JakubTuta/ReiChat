<script setup lang="ts">
import type { ToolCall } from '~/composables/websocket'
import type { IContainer } from '~/models/container'

const props = defineProps<{
  content: string
  toolCalls?: ToolCall[]
  notices?: string[]
  thinking?: string
  model: IContainer | null
}>()

const parts = computed(() => splitMessageRaw(props.content))
const thoughts = computed(() => props.thinking || parts.value.thoughts)
// The reasoning is shown live, open, until the answer itself starts — up to
// then it's the only sign of progress, and can run for minutes on a big model.
const thinkingLive = computed(() => !!props.thinking && !props.content)

const thinkingBox = ref<HTMLDivElement | null>(null)
watch(() => props.thinking, () => {
  nextTick(() => {
    if (thinkingBox.value)
      thinkingBox.value.scrollTop = thinkingBox.value.scrollHeight
  })
})
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
        v-if="!thinkingLive"
        :thoughts="thoughts"
        :tool-calls="toolCalls ?? []"
      />
    </div>

    <MessageNotices :notices="notices ?? []" />

    <div
      v-if="thinkingLive"
      class="thinking-live"
    >
      <div class="thinking-live-header">
        <v-icon
          size="14"
          icon="$mdiLightbulbOnOutline"
        />

        <span>Thinking it through before answering…</span>
      </div>

      <div
        ref="thinkingBox"
        class="thinking-live-text"
      >
        {{ thinking }}
      </div>
    </div>

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

.thinking-live {
  margin-bottom: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--color-soft-2);
  border: 1px solid var(--color-line-2);
}

.thinking-live-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--color-mint-deep);
}

.thinking-live-text {
  max-height: 160px;
  overflow-y: auto;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.6;
  font-style: italic;
  color: var(--color-ink-2);
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
