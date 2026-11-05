<script setup lang="ts">
const snackbarStore = useSnackbarStore()
const { toasts } = storeToRefs(snackbarStore)

const DOT_COLOR: Record<string, string> = {
  success: 'var(--color-mint)',
  warning: 'var(--color-amber)',
  error: 'var(--color-red)',
  info: 'oklch(0.72 0.05 168)',
}
</script>

<template>
  <div
    class="toast-stack"
    role="status"
    aria-live="polite"
  >
    <TransitionGroup name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        class="toast"
      >
        <span
          class="toast-dot"
          :style="{'backgroundColor': DOT_COLOR[toast.color] ?? DOT_COLOR.info}"
        />

        <span class="toast-text">{{ toast.text }}</span>

        <button
          type="button"
          class="toast-dismiss"
          title="Dismiss"
          aria-label="Dismiss notification"
          @click="snackbarStore.dismiss(toast.id)"
        >
          <v-icon
            size="14"
            icon="$mdiClose"
          />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-stack {
  position: fixed;
  right: 20px;
  bottom: 20px;
  z-index: 3000;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: min(340px, calc(100vw - 40px));
}

.toast {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 11px 15px;
  border-radius: 12px;
  background: var(--color-card);
  border: 1px solid var(--color-line);
  box-shadow: var(--shadow-toast);
}

.toast-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.toast-text {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  color: var(--color-ink);
}

.toast-dismiss {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  cursor: pointer;
}

.toast-dismiss:hover {
  background: var(--color-soft);
  color: var(--color-ink);
}

.toast-enter-active,
.toast-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(6px);
}

.toast-leave-active {
  position: absolute;
}
</style>
