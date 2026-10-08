<script setup lang="ts">
defineProps<{
  notices: string[]
}>()
</script>

<template>
  <div
    v-if="notices.length"
    class="message-notices"
  >
    <div
      v-for="(notice, idx) in notices"
      :key="idx"
      class="message-notice"
      role="status"
    >
      <v-icon
        size="14"
        class="message-notice-icon"
      >
        mdi-alert-outline
      </v-icon>

      <span>{{ notice }}</span>
    </div>
  </div>
</template>

<style scoped>
.message-notices {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 8px;
}

.message-notice {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-size: 12.5px;
  line-height: 1.5;
}

.message-notice-icon {
  color: var(--color-amber);
  margin-top: 2px;
}
</style>
