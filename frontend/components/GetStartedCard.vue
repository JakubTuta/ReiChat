<script setup lang="ts">
import type { Recommendation } from '~/utils/recommend'
import { containerNameFor } from '~/utils/containerName'
import { recommendModels } from '~/utils/recommend'

const chatStore = useChatStore()
const { aiModels } = storeToRefs(chatStore)

const hardwareStore = useHardwareStore()
const { hardware } = storeToRefs(hardwareStore)

const containerStore = useContainerStore()
const { pullProgress } = storeToRefs(containerStore)

const EMBEDDER = { model: 'nomic-embed-text', parameters: 'latest' }

const withDocuments = ref(false)
const chosenKey = ref<string | null>(null)
const installingKey = ref<string | null>(null)

onMounted(() => {
  if (!aiModels.value.length)
    chatStore.fetchAIModels()
  if (hardware.value.ram_bytes === null)
    hardwareStore.fetchHardware()
  containerStore.connectSocket()
})

const recommendations = computed(() => recommendModels(aiModels.value, hardware.value))

const options = computed(() => [recommendations.value.best, recommendations.value.lighter]
  .filter((r): r is Recommendation => r !== null)
  .filter((r, i, all) => all.findIndex(o => keyOf(o) === keyOf(r)) === i))

watch(options, (list) => {
  if (!chosenKey.value && list.length)
    chosenKey.value = keyOf(list[0])
}, { immediate: true })

function keyOf(r: Recommendation): string {
  return containerNameFor(r.model.model, r.version.parameters)
}

const progress = computed(() => (installingKey.value
  ? pullProgress.value[installingKey.value] ?? null
  : null))

const embedderAvailable = computed(() => aiModels.value.some(m => m.model === EMBEDDER.model))

const hardwareLine = computed(() => {
  const { ram_bytes, vram_bytes, gpu_available } = hardware.value
  if (vram_bytes && gpu_available !== false)
    return `Picked for your graphics card (${formatBytes(vram_bytes)} of video memory).`
  if (ram_bytes)
    return `Picked for your computer (${formatBytes(ram_bytes)} of memory available to Docker).`

  return 'Your hardware couldn\'t be detected, so these are small models that run almost anywhere.'
})

async function install() {
  const choice = options.value.find(o => keyOf(o) === chosenKey.value)
  if (!choice)
    return

  installingKey.value = keyOf(choice)
  await containerStore.runContainer({ model: choice.model.model, parameters: choice.version.parameters })
  if (withDocuments.value && embedderAvailable.value)
    await containerStore.runContainer(EMBEDDER)
}
</script>

<template>
  <div class="get-started">
    <div class="get-started-title">
      Get started in one click
    </div>

    <p class="get-started-lede">
      {{ hardwareLine }} You can install more models any time on the Models page.
    </p>

    <div
      v-if="!options.length"
      class="get-started-loading"
    >
      Loading the model list…
    </div>

    <template v-else-if="!installingKey">
      <div class="get-started-options">
        <button
          v-for="(option, index) in options"
          :key="keyOf(option)"
          type="button"
          class="get-started-option"
          :class="{'get-started-option--active': chosenKey === keyOf(option)}"
          @click="chosenKey = keyOf(option)"
        >
          <span class="get-started-option-head">
            <span class="get-started-option-name">{{ option.model.name }} {{ option.version.parameters }}</span>

            <span class="get-started-option-tag">{{ index === 0
              ? 'Recommended'
              : 'Faster' }}</span>
          </span>

          <span class="get-started-option-desc">{{ option.blurb }}</span>

          <span class="get-started-option-size">{{ option.version.size }} download</span>
        </button>
      </div>

      <label
        v-if="embedderAvailable"
        class="get-started-docs"
      >
        <input
          v-model="withDocuments"
          type="checkbox"
        >
        Also set up asking questions about your own files (+ about 270 MB)
      </label>

      <v-btn
        color="mint-btn"
        variant="flat"
        :disabled="!chosenKey"
        @click="install"
      >
        Install and get chatting
      </v-btn>
    </template>

    <div
      v-else
      class="get-started-progress"
    >
      <template v-if="progress">
        <div class="get-started-progress-label">
          Downloading… {{ progress.percent }}%
        </div>

        <div class="get-started-track">
          <div
            class="get-started-fill"
            :style="{'width': `${progress.percent}%`}"
          />
        </div>

        <div class="get-started-progress-detail">
          {{ progress.detail }}
        </div>
      </template>

      <template v-else>
        Setting things up — this page updates when the model is ready.
      </template>
    </div>
  </div>
</template>

<style scoped>
.get-started {
  width: 100%;
  max-width: 560px;
  margin: 0 auto;
  padding: 18px 20px;
  border-radius: 14px;
  border: 1px solid var(--color-line);
  background: var(--color-card);
  text-align: left;
}

.get-started-title {
  font-size: 16px;
  font-weight: 500;
  color: var(--color-ink);
}

.get-started-lede {
  margin: 4px 0 14px;
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--color-ink-2);
}

.get-started-loading {
  font-size: 12.5px;
  color: var(--color-ink-3);
}

.get-started-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}

.get-started-option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 3px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  text-align: left;
  font-family: var(--font-sans);
  cursor: pointer;
}

.get-started-option--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
}

.get-started-option-head {
  display: flex;
  justify-content: space-between;
  width: 100%;
  gap: 6px;
}

.get-started-option-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.get-started-option-tag {
  font-size: 10.5px;
  color: var(--color-mint-deep);
}

.get-started-option-desc {
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--color-ink-2);
}

.get-started-option-size {
  font-size: 11px;
  color: var(--color-ink-3);
}

.get-started-docs {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 12px;
  font-size: 12.5px;
  color: var(--color-ink-2);
  cursor: pointer;
}

.get-started-progress {
  font-size: 12.5px;
  color: var(--color-ink-2);
}

.get-started-progress-label {
  margin-bottom: 6px;
  color: var(--color-ink);
}

.get-started-track {
  height: 6px;
  border-radius: 3px;
  background: var(--color-soft-2);
  overflow: hidden;
}

.get-started-fill {
  height: 100%;
  background: var(--color-mint);
  transition: width 0.3s ease;
}

.get-started-progress-detail {
  margin-top: 4px;
  font-size: 11px;
  color: var(--color-ink-3);
}
</style>
