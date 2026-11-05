<script setup lang="ts">
import type { ShellType } from '~/utils/dockerCommands'
import { SHELL_OPTIONS } from '~/utils/dockerCommands'

defineProps<{
  commands: string[]
}>()

const shell = defineModel<ShellType>('shell', { required: true })

const { copiedKey, copy } = useCopyFeedback()
</script>

<template>
  <v-expansion-panels
    class="advanced-panel"
    variant="accordion"
  >
    <v-expansion-panel>
      <v-expansion-panel-title class="advanced-toggle font-mono">
        advanced: run manually
      </v-expansion-panel-title>

      <v-expansion-panel-text>
        <div class="shell-pills">
          <button
            v-for="option in SHELL_OPTIONS"
            :key="option.value"
            type="button"
            class="shell-pill font-mono"
            :class="{'shell-pill--active': shell === option.value}"
            @click="shell = option.value"
          >
            {{ option.title }}
          </button>
        </div>

        <div
          v-for="(cmd, idx) in commands"
          :key="idx"
          class="cli-command"
        >
          <pre class="font-mono">{{ cmd }}</pre>

          <button
            type="button"
            class="cli-copy"
            :title="copiedKey === `cmd-${idx}`
              ? 'Copied'
              : 'Copy'"
            :aria-label="`Copy command ${idx + 1}`"
            @click="copy(cmd, `cmd-${idx}`)"
          >
            <v-icon
              size="14"
              icon="$mdiContentCopy"
            />
          </button>
        </div>
      </v-expansion-panel-text>
    </v-expansion-panel>
  </v-expansion-panels>
</template>

<style scoped>
.advanced-panel {
  width: auto;
  flex-basis: 100%;
}

.advanced-toggle {
  font-size: 10.5px;
  color: var(--color-ink-3);
  min-height: 0 !important;
  padding: 8px !important;
}

.shell-pills {
  display: flex;
  gap: 6px;
  margin-bottom: 10px;
}

.shell-pill {
  padding: 4px 9px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  color: var(--color-ink-2);
  font-size: 10.5px;
  cursor: pointer;
}

.shell-pill--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.cli-command {
  position: relative;
  background: var(--color-soft);
  border: 1px solid var(--color-line-2);
  border-radius: 10px;
  padding: 10px 34px 10px 10px;
  margin-bottom: 8px;
}

.cli-command pre {
  margin: 0;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--color-ink);
}

.cli-copy {
  position: absolute;
  top: 8px;
  right: 8px;
  border: none;
  background: transparent;
  color: var(--color-ink-3);
  cursor: pointer;
}

.cli-copy:hover {
  color: var(--color-mint-deep);
}
</style>
