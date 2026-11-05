<script setup lang="ts">
import type { NuxtError } from '#app'

const props = defineProps<{ error: NuxtError }>()

function goHome() {
  clearError({ redirect: '/' })
}
</script>

<template>
  <!--
    error.vue replaces app.vue entirely rather than rendering inside it
    (confirmed via Nuxt docs — migrating a legacy error layout into
    error.vue needs its own explicit <NuxtLayout> wrap for the same
    reason), so this needs its own <v-app> — there's no ancestor one to
    rely on here.
  -->
  <v-app>
    <AppTopBar />

    <v-main>
      <v-container class="error-page">
        <v-icon
          size="72"
          icon="$mdiRobotConfusedOutline"
          class="text-medium-emphasis mb-4"
        />

        <div class="text-h3 font-weight-bold mb-2">
          {{ props.error.status || 404 }}
        </div>

        <div class="text-h6 font-weight-medium mb-2">
          {{ props.error.status === 404
            ? 'Page not found'
            : 'Something went wrong' }}
        </div>

        <div class="text-body-1 text-medium-emphasis mb-6">
          {{ props.error.statusText || props.error.message || 'An unexpected error occurred.' }}
        </div>

        <v-btn
          color="primary"
          @click="goHome"
        >
          Back to home
        </v-btn>
      </v-container>
    </v-main>
  </v-app>
</template>

<style scoped>
.error-page {
  min-height: calc(100dvh - 64px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}
</style>
