<script setup lang="ts">
import type { ContextReport } from '~/constants/websocket'
import { nextPreset, previousPreset, spillsOutOfGpu } from '~/utils/memory'

const props = defineProps<{
  context: ContextReport
  // Only the newest reply offers buttons — acting on an old one would change
  // settings based on a state the chat has moved past.
  actionable: boolean
}>()

const emit = defineEmits<{
  (e: 'moreMemory', tokens: number): void
  (e: 'lessMemory', tokens: number): void
  (e: 'summarize'): void
  (e: 'continueFresh'): void
}>()

const bigger = computed(() => nextPreset(props.context.limit, props.context.model_max))
const smaller = computed(() => previousPreset(props.context.limit))
const spilling = computed(() => spillsOutOfGpu(props.context))
const gpuPercent = computed(() => Math.round((props.context.gpu_share ?? 0) * 100))
</script>

<template>
  <div
    v-if="context.dropped > 0 && !context.summarized"
    class="memory-notice"
    role="status"
  >
    <div class="memory-notice-text">
      <v-icon
        size="14"
        class="memory-notice-icon"
      >
        mdi-brain
      </v-icon>

      <span>
        This chat is longer than the model can keep in mind, so it didn't see the first
        {{ context.dropped }} message{{ context.dropped === 1
          ? ''
          : 's' }} when writing this.
      </span>
    </div>

    <div
      v-if="actionable"
      class="memory-notice-actions"
    >
      <button
        v-if="bigger"
        type="button"
        class="memory-notice-btn memory-notice-btn--primary"
        @click="emit('moreMemory', bigger.tokens)"
      >
        Give it more memory ({{ bigger.label }})
      </button>

      <button
        type="button"
        class="memory-notice-btn"
        @click="emit('summarize')"
      >
        Summarize old messages instead
      </button>

      <button
        type="button"
        class="memory-notice-btn"
        @click="emit('continueFresh')"
      >
        Start a fresh chat with a summary
      </button>
    </div>
  </div>

  <div
    v-if="spilling"
    class="memory-notice"
    role="status"
  >
    <div class="memory-notice-text">
      <v-icon
        size="14"
        class="memory-notice-icon"
      >
        mdi-speedometer-slow
      </v-icon>

      <span>
        Only {{ gpuPercent }}% of the model fits in your graphics card's memory at this memory size, so
        the rest runs on the much slower processor.
        <template v-if="smaller">
          A shorter memory usually makes replies a lot faster.
        </template>
      </span>
    </div>

    <div
      v-if="actionable && smaller"
      class="memory-notice-actions"
    >
      <button
        type="button"
        class="memory-notice-btn memory-notice-btn--primary"
        @click="emit('lessMemory', smaller.tokens)"
      >
        Use shorter memory ({{ smaller.label }})
      </button>
    </div>
  </div>
</template>

<style scoped>
.memory-notice {
  margin-bottom: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 1px solid var(--color-banner-border);
  background: var(--color-banner-bg);
  color: var(--color-banner-text);
  font-size: 12.5px;
  line-height: 1.5;
}

.memory-notice-text {
  display: flex;
  align-items: flex-start;
  gap: 6px;
}

.memory-notice-icon {
  margin-top: 2px;
}

.memory-notice-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
  padding-left: 20px;
}

.memory-notice-btn {
  padding: 4px 10px;
  border-radius: 14px;
  border: 1px solid var(--color-banner-border);
  background: var(--color-card);
  color: var(--color-banner-text);
  font-family: var(--font-sans);
  font-size: 11.5px;
  cursor: pointer;
}

.memory-notice-btn--primary {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: var(--color-mint-ink);
}
</style>
