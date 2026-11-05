<script setup lang="ts">
import type { PullProgressInfo } from '~/composables/containerSocket'
import type { IAIModel, IAIModelVersion } from '~/models/aiModel'
import type { IHardwareInfo } from '~/models/hardware'
import type { LoadedModel } from '~/stores/containerStore'
import type { ShellType } from '~/utils/dockerCommands'
import { computeFitLabel } from '~/models/hardware'
import { formatCount } from '~/utils/modelCatalog'
import { FIT_LABEL, gpuShareLabel, statusLabel, statusMeta } from '~/utils/modelStatus'

const props = defineProps<{
  model: IAIModel
  installedVersionCount: number
  // Container status of the selected size; 'not_found' when nothing exists
  // for it (or no size is selected).
  status: string
  pullProgress: PullProgressInfo | null
  // A start/stop/remove for this size is in flight.
  operationLoading: boolean
  loaded: LoadedModel | null
  hardware: IHardwareInfo
  isExternal: boolean
  manualCommands: string[]
}>()

const emit = defineEmits<{
  (e: 'create' | 'start' | 'resume' | 'stop' | 'remove' | 'update' | 'unload'): void
}>()

const selectedVersion = defineModel<IAIModelVersion | null>('selectedVersion', { default: null })
const shell = defineModel<ShellType>('shell', { required: true })

function selectVersion(version: IAIModelVersion) {
  selectedVersion.value = selectedVersion.value?.parameters === version.parameters
    ? null
    : version
}

const fit = computed(() => (selectedVersion.value
  ? computeFitLabel(selectedVersion.value.size_bytes, props.hardware)
  : null))

interface PrimaryAction {
  label: string
  primary: boolean
  disabled: boolean
  loading: boolean
  event: 'create' | 'start' | 'resume' | 'stop' | null
}

function inert(label: string, loading = false): PrimaryAction {
  return { label, primary: false, disabled: true, loading, event: null }
}

const primaryAction = computed<PrimaryAction>(() => {
  if (!selectedVersion.value)
    return inert('Select a version first')

  if (props.pullProgress)
    return inert('Downloading…', true)

  if (props.operationLoading || props.status === 'restarting')
    return inert('Please wait…', true)

  switch (props.status) {
    case 'not_found':
      return { label: 'Create container', primary: true, disabled: false, loading: false, event: 'create' }
    case 'exited':
    case 'paused':
      return { label: 'Start container', primary: false, disabled: false, loading: false, event: 'start' }
    case 'incomplete':
      return { label: 'Resume download', primary: true, disabled: false, loading: false, event: 'resume' }
    case 'starting':
      return inert('Starting…', true)
    case 'running':
      return props.isExternal
        ? inert('Installed')
        : { label: 'Stop container', primary: false, disabled: false, loading: false, event: 'stop' }
    case 'pulling_model':
      return inert('Pulling model…', true)
    default:
      return inert('N/A')
  }
})

const canRemove = computed(() => ['running', 'exited', 'paused', 'restarting', 'pulling_model', 'starting', 'incomplete'].includes(props.status))

function runPrimaryAction() {
  if (primaryAction.value.event)
    emit(primaryAction.value.event)
}
</script>

<template>
  <div class="model-card">
    <div class="model-card-title-row">
      <span class="model-card-name">{{ model.name }}</span>

      <span
        v-if="model.can_process_image"
        class="badge badge--mint"
      >VISION</span>

      <span
        v-if="installedVersionCount"
        class="badge badge--grey"
      >{{ installedVersionCount }} INSTALLED</span>

      <span
        v-if="model.is_custom"
        class="badge badge--grey"
      >ADDED BY YOU</span>

      <span class="model-card-popularity font-mono">★ {{ formatCount(model.popularity) }}</span>
    </div>

    <p class="model-card-description">
      {{ model.description }}
    </p>

    <div class="version-pills">
      <button
        v-for="version in model.versions"
        :key="version.parameters"
        type="button"
        class="version-pill font-mono"
        :class="{'version-pill--active': selectedVersion?.parameters === version.parameters}"
        @click="selectVersion(version)"
      >
        {{ version.parameters }} · {{ version.size }}
      </button>
    </div>

    <template v-if="selectedVersion">
      <div class="status-line">
        <span
          class="status-dot"
          :style="{'backgroundColor': `var(--color-${statusMeta(status).dot})`}"
        />

        <span class="status-label">{{ statusLabel(status, model.is_embedding) }}</span>

        <template v-if="fit && fit.label !== 'unknown'">
          <span class="status-sep">·</span>

          <span class="fit-label">
            {{ FIT_LABEL[fit.label] }}

            <v-tooltip
              activator="parent"
              location="top"
            >
              Approximate — based on this model's download size vs.
              {{ fit.source === 'vram'
                ? 'GPU memory'
                : 'system RAM' }}
              ({{ formatBytes(fit.availableBytes) }} available).
            </v-tooltip>
          </span>
        </template>
      </div>

      <div
        v-if="loaded"
        class="loaded-line"
      >
        <span>
          In memory: {{ formatBytes(loaded.size_bytes) }} · {{ gpuShareLabel(loaded, hardware.gpu_available) }}
        </span>

        <button
          type="button"
          class="loaded-free"
          title="Unload it now instead of waiting a few idle minutes"
          @click="emit('unload')"
        >
          Free memory
        </button>
      </div>

      <template v-if="pullProgress">
        <div class="pull-progress-row">
          <span>{{ pullProgress.phase === 'image_pull_progress'
            ? 'Downloading base image'
            : 'Downloading model' }}</span>

          <span class="font-mono">{{ pullProgress.percent }}%</span>
        </div>

        <div class="pull-progress-track">
          <div
            class="pull-progress-fill"
            :style="{'width': `${pullProgress.percent}%`}"
          />
        </div>

        <div class="pull-progress-detail">
          {{ pullProgress.detail }}
        </div>
      </template>

      <div class="action-row">
        <v-btn
          :color="primaryAction.primary
            ? 'mint-btn'
            : undefined"
          :variant="primaryAction.primary
            ? 'flat'
            : 'outlined'"
          :disabled="primaryAction.disabled"
          :loading="primaryAction.loading"
          size="small"
          rounded="lg"
          style="min-width: 150px"
          @click="runPrimaryAction"
        >
          {{ primaryAction.label }}
        </v-btn>

        <v-btn
          v-if="status === 'running' && !pullProgress"
          variant="text"
          size="small"
          title="Download a newer version of this model if there is one"
          @click="emit('update')"
        >
          Check for update
        </v-btn>

        <v-btn
          v-if="canRemove"
          variant="text"
          color="red"
          size="small"
          :disabled="operationLoading"
          @click="emit('remove')"
        >
          Remove
        </v-btn>

        <span class="action-row-spacer" />

        <ModelCliPanel
          v-if="!isExternal && manualCommands.length"
          v-model:shell="shell"
          :commands="manualCommands"
        />
      </div>
    </template>

    <div
      v-else
      class="no-version-hint"
    >
      Select a version to manage this model.
    </div>
  </div>
</template>

<style scoped>
.model-card {
  background: var(--color-card);
  border: 1px solid var(--color-line-2);
  border-radius: 18px;
  padding: 24px 24px 20px;
  display: flex;
  flex-direction: column;
}

.model-card-title-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}

.model-card-name {
  font-size: 14.5px;
  font-weight: 600;
  color: var(--color-ink);
}

.badge {
  font-family: var(--font-mono);
  font-size: 9.5px;
  padding: 2px 6px;
  border-radius: 5px;
}

.badge--mint {
  background: var(--color-mint-tint);
  color: var(--color-mint-deep);
}

.badge--grey {
  background: var(--color-soft-2);
  color: var(--color-ink-2);
}

.model-card-popularity {
  margin-left: auto;
  font-size: 10.5px;
  color: var(--color-ink-3);
}

.model-card-description {
  font-size: 13px;
  line-height: 1.6;
  color: var(--color-ink-2);
  margin: 0 0 14px;
}

.version-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.version-pill {
  padding: 5px 10px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  color: var(--color-ink-2);
  font-size: 11px;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.version-pill--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
}

.status-line {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 12px;
  color: var(--color-ink-2);
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-sep {
  color: var(--color-ink-3);
}

.fit-label {
  text-decoration: underline dotted var(--color-ink-3);
  text-underline-offset: 3px;
  cursor: help;
}

.loaded-line {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -4px 0 10px;
  font-size: 12px;
  color: var(--color-mint-deep);
}

.loaded-free {
  border: none;
  background: transparent;
  padding: 0;
  font-size: 12px;
  color: var(--color-ink-3);
  text-decoration: underline;
  cursor: pointer;
}

.pull-progress-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--color-ink-2);
  margin-bottom: 4px;
}

.pull-progress-track {
  height: 4px;
  border-radius: 4px;
  background: var(--color-soft-2);
  overflow: hidden;
  margin-bottom: 4px;
}

.pull-progress-fill {
  height: 100%;
  background: var(--color-mint);
  transition: width 0.2s ease;
}

.pull-progress-detail {
  font-size: 11px;
  color: var(--color-ink-3);
  margin-bottom: 12px;
}

.action-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.action-row-spacer {
  flex: 1;
}

.no-version-hint {
  font-size: 12.5px;
  color: var(--color-ink-3);
}
</style>
