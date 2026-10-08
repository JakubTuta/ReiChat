<script setup lang="ts">
import type { IGenerationParams } from '~/stores/chatStore'

const props = defineProps<{
  persona: { id: string, name: string } | null
  params: IGenerationParams
  activeCollectionsCount: number
  toolsEnabled: boolean
  jsonEnabled: boolean
}>()

const emit = defineEmits<{
  (e: 'open', target: 'persona' | 'parameters' | 'files' | 'json' | 'tools'): void
}>()

const activeParamLabels = computed(() => {
  const entries: string[] = []
  if (props.params.temperature !== null)
    entries.push(`temp ${props.params.temperature.toFixed(2)}`)
  if (props.params.top_p !== null)
    entries.push(`top_p ${props.params.top_p.toFixed(2)}`)
  if (props.params.seed !== null)
    entries.push(`seed ${props.params.seed}`)

  return entries
})

const paramsChipLabel = computed(() => {
  if (activeParamLabels.value.length === 0)
    return null
  if (activeParamLabels.value.length === 1)
    return activeParamLabels.value[0]

  return `Parameters · ${activeParamLabels.value.length}`
})

const hasAny = computed(() => !!props.persona
  || !!paramsChipLabel.value
  || props.activeCollectionsCount > 0
  || props.toolsEnabled
  || props.jsonEnabled)
</script>

<template>
  <template v-if="hasAny">
    <button
      v-if="persona"
      type="button"
      class="setting-chip"
      @click="emit('open', 'persona')"
    >
      <span class="setting-chip-dot" />
      {{ persona.name }}
    </button>

    <button
      v-if="paramsChipLabel"
      type="button"
      class="setting-chip font-mono"
      @click="emit('open', 'parameters')"
    >
      <span class="setting-chip-dot" />
      {{ paramsChipLabel }}
    </button>

    <button
      v-if="activeCollectionsCount"
      type="button"
      class="setting-chip"
      @click="emit('open', 'files')"
    >
      <span class="setting-chip-dot" />
      Files · {{ activeCollectionsCount }}
    </button>

    <button
      v-if="toolsEnabled"
      type="button"
      class="setting-chip"
      @click="emit('open', 'tools')"
    >
      <span class="setting-chip-dot" />
      Tools
    </button>

    <button
      v-if="jsonEnabled"
      type="button"
      class="setting-chip"
      @click="emit('open', 'json')"
    >
      <span class="setting-chip-dot" />
      JSON schema
    </button>
  </template>
</template>

<style scoped>
.setting-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 11px;
  border-radius: 20px;
  border: none;
  background: var(--color-mint-tint);
  color: oklch(0.4 0.07 168);
  font-family: var(--font-sans);
  font-size: 11.5px;
  cursor: pointer;
  white-space: nowrap;
  transition: filter 0.15s ease;
}

.setting-chip:hover {
  filter: brightness(0.97);
}

.setting-chip-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-mint);
  flex-shrink: 0;
}
</style>
