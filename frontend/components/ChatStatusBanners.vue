<script setup lang="ts">
import type { IContainer } from '~/models/container'

defineProps<{
  model: IContainer | null
  reconnecting: boolean
  reconnectFailed: boolean
}>()

const emit = defineEmits<{
  (e: 'reconnect'): void
}>()
</script>

<template>
  <div
    v-if="model && model.status !== 'running'"
    class="status-banner"
  >
    <span class="status-banner-label">Model</span>

    <span class="status-banner-text">
      <template v-if="model.status === 'pulling_model'">
        Model is still being pulled — sending will be enabled once it's ready.
      </template>

      <template v-else-if="model.status === 'starting'">
        Model is starting up — this takes a few seconds.
      </template>

      <template v-else-if="model.status === 'incomplete'">
        This model's download didn't finish. Resume it on the Models page to start chatting.
      </template>

      <template v-else>
        Container is not running.
      </template>
    </span>

    <NuxtLink
      v-if="model.status !== 'pulling_model'"
      to="/models"
      class="status-banner-action"
      style="text-decoration: none"
    >
      Manage on Models
    </NuxtLink>
  </div>

  <div
    v-if="reconnecting"
    class="status-banner"
  >
    <span class="status-banner-label">Connection</span>

    <span class="status-banner-text">Connection lost — reconnecting…</span>
  </div>

  <div
    v-if="reconnectFailed"
    class="status-banner"
  >
    <span class="status-banner-label">Connection</span>

    <span class="status-banner-text">Couldn't reconnect. Sending is paused until you reconnect.</span>

    <button
      type="button"
      class="status-banner-action"
      @click="emit('reconnect')"
    >
      Reconnect
    </button>
  </div>
</template>
