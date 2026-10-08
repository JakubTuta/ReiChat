<script setup lang="ts">
defineProps<{
  variant: 'no-model' | 'no-messages'
  suggestions?: string[]
}>()

const emit = defineEmits<{
  (e: 'useSuggestion', text: string): void
}>()
</script>

<template>
  <div class="empty-state">
    <div class="empty-state-icon">
      <v-icon
        size="18"
        :icon="variant === 'no-model'
          ? 'mdi-robot-outline'
          : 'mdi-chat-outline'"
        style="color: var(--color-mint-ink)"
      />
    </div>

    <template v-if="variant === 'no-model'">
      <div class="empty-state-headline">
        No model is running yet
      </div>

      <GetStartedCard class="empty-state-get-started" />

      <div class="empty-state-or">
        Or set one up yourself:
      </div>

      <ol class="empty-state-steps">
        <li>Make sure <strong>Docker Desktop</strong> is running.</li>

        <li>
          Go to the
          <NuxtLink
            to="/models"
            class="empty-state-link"
          >
            Models page
          </NuxtLink>

          , pick a model, select a version, and click <strong>Create container</strong>.
        </li>

        <li>Once the status shows <strong>Running</strong>, come back here and start chatting.</li>
      </ol>
    </template>

    <template v-else>
      <div class="empty-state-headline">
        What are we working on?
      </div>

      <div class="empty-state-suggestions">
        <button
          v-for="s in suggestions"
          :key="s"
          type="button"
          class="suggestion-pill"
          @click="emit('useSuggestion', s)"
        >
          {{ s }}
        </button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.empty-state-get-started {
  margin: 14px 0 18px;
}

.empty-state-or {
  font-size: 12px;
  color: var(--color-ink-3);
  margin-bottom: 6px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 16px;
  text-align: center;
}

.empty-state-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--color-mint-btn);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 18px;
}

.empty-state-headline {
  font-size: 25px;
  line-height: 1.3;
  letter-spacing: -0.02em;
  color: var(--color-ink);
  margin-bottom: 18px;
}

.empty-state-steps {
  text-align: left;
  max-width: 380px;
  margin: 0 auto;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--color-ink-2);
}

.empty-state-link {
  color: var(--color-mint-deep);
}

.empty-state-suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  max-width: 480px;
}

.suggestion-pill {
  padding: 8px 14px;
  border-radius: 20px;
  border: none;
  background: var(--color-mint-tint);
  color: oklch(0.4 0.07 168);
  font-family: var(--font-sans);
  font-size: 13px;
  cursor: pointer;
  transition: transform 0.15s ease;
}

.suggestion-pill:hover {
  transform: translateY(-2px);
}

@media (prefers-reduced-motion: reduce) {
  .suggestion-pill:hover {
    transform: none;
  }
}
</style>
