<script setup lang="ts">
import type { ToolApproval } from '~/constants/websocket'

const props = defineProps<{
  request: ToolApproval
}>()

const emit = defineEmits<{
  (e: 'decide', payload: { approved: boolean, always: boolean }): void
}>()

const prettyArgs = computed(() => JSON.stringify(props.request.args, null, 2))

function decide(approved: boolean, always: boolean) {
  emit('decide', { approved, always })
}
</script>

<template>
  <div
    class="tool-approval"
    role="alertdialog"
    aria-label="Allow a tool to run?"
  >
    <div class="tool-approval-title">
      <v-icon
        size="16"
        icon="$mdiShieldAlertOutline"
      />

      <span>
        The model wants to use <strong>{{ request.name }}</strong>

        from <strong>{{ request.server }}</strong>
      </span>
    </div>

    <p class="tool-approval-body">
      Tools from connected servers can act on your behalf (read or change files, send messages, call websites).
      Only allow it if this is something you asked for.
    </p>

    <pre class="tool-approval-args font-mono">{{ prettyArgs }}</pre>

    <div class="tool-approval-actions">
      <v-btn
        size="small"
        variant="flat"
        color="mint-btn"
        @click="decide(true, false)"
      >
        Allow once
      </v-btn>

      <v-btn
        size="small"
        variant="outlined"
        @click="decide(true, true)"
      >
        Always allow {{ request.server }}
      </v-btn>

      <v-btn
        size="small"
        variant="text"
        color="red"
        @click="decide(false, false)"
      >
        Deny
      </v-btn>
    </div>
  </div>
</template>

<style scoped>
.tool-approval {
  margin: 4px 0 12px;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid var(--color-banner-border);
  background: var(--color-banner-bg);
  color: var(--color-banner-text);
}

.tool-approval-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13.5px;
}

.tool-approval-body {
  margin: 6px 0 8px;
  font-size: 12.5px;
  line-height: 1.5;
}

.tool-approval-args {
  max-height: 180px;
  overflow: auto;
  margin: 0 0 10px;
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--color-card);
  color: var(--color-ink);
  font-size: 11.5px;
  white-space: pre-wrap;
  word-break: break-word;
}

.tool-approval-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
</style>
