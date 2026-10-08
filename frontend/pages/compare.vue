<script setup lang="ts">
import type { CompareSocketWrapper } from '~/constants/compareSocket'
import { getCompareSocket } from '~/constants/compareSocket'

interface ResultState {
  content: string
  done: boolean
  error: string | null
  stats: { prompt_tokens?: number, completion_tokens?: number, tokens_per_second?: number } | null
}

const MAX_TARGETS = 4

const containerStore = useContainerStore()
const { containers } = storeToRefs(containerStore)

const chatStore = useChatStore()
const { aiModels } = storeToRefs(chatStore)

const snackbarStore = useSnackbarStore()
const wsBaseURL = useRuntimeConfig().public.serverUrl

const prompt = ref('')
// The prompt the current results answer — the box may be edited since.
const lastPrompt = ref('')
const selectedKeys = ref<string[]>([])
const results = ref<Record<string, ResultState>>({})
const running = ref(false)
const socket = ref<CompareSocketWrapper | null>(null)
const BLIND_STORAGE_KEY = 'reichat.compare.blind'

function readBlindPreference(): boolean {
  try {
    return localStorage.getItem(BLIND_STORAGE_KEY) === 'true'
  }
  catch {
    return false
  }
}

const blind = ref(readBlindPreference())

function setBlind(value: boolean) {
  blind.value = value
  try {
    localStorage.setItem(BLIND_STORAGE_KEY, String(value))
  }
  catch {
    // Private windows can refuse storage; the choice then lasts for this visit only.
  }
}
const displayOrder = ref<string[]>([])
const pickedKey = ref<string | null>(null)

onMounted(async () => {
  if (!aiModels.value.length)
    await chatStore.fetchAIModels()
  if (!containers.value.length)
    await containerStore.getUserContainers()
  containerStore.startPolling()

  // Opened as soon as the page loads, not lazily on the first "Compare"
  // click — a WebSocket takes a moment to finish its handshake, and
  // creating it right when send() needs it raced that handshake: the very
  // first comparison after a page load would silently drop the prompt
  // (readyState was still CONNECTING) and spin forever with nothing sent.
  // Connecting early gives it the whole time the user spends picking
  // models and typing a prompt to finish, the same way the chat page's
  // socket opens as soon as a chat is selected rather than on Send.
  ensureSocket()
  fetchScoreboard()
})

onUnmounted(() => {
  containerStore.stopPolling()
  socket.value?.closeConnection()
})

// Every installed, non-embedding model is selectable — running ones can
// answer immediately, stopped ones start on the same click that selects
// them (only "pulling" and "not installed" are excluded: there's nothing
// to compare against yet).
const selectableModels = computed(() => {
  return containers.value
    .filter(c => c.status !== 'pulling_model')
    .map((container) => {
      const foundAIModel = aiModels.value.find(m => m.model === container.environment.model)
      if (!foundAIModel || foundAIModel.is_embedding)
        return null

      return {
        name: foundAIModel.name as string,
        model: container.environment.model as string,
        parameters: container.environment.parameters as string,
        key: `${container.environment.model}:${container.environment.parameters}`,
        status: container.status as string,
      }
    })
    .filter((m): m is { name: string, model: string, parameters: string, key: string, status: string } => m !== null)
    .sort((a, b) => a.name.localeCompare(b.name))
})

function toggleTarget(target: { key: string, model: string, parameters: string, status: string }) {
  if (selectedKeys.value.includes(target.key)) {
    selectedKeys.value = selectedKeys.value.filter(k => k !== target.key)

    return
  }

  if (selectedKeys.value.length >= MAX_TARGETS) {
    snackbarStore.showSnackbarError(`Compare up to ${MAX_TARGETS} models at once.`)

    return
  }

  if (target.status !== 'running')
    containerStore.runContainer({ model: target.model, parameters: target.parameters })

  selectedKeys.value = [...selectedKeys.value, target.key]
}

const canRun = computed(() => !!prompt.value.trim() && selectedKeys.value.length > 0 && !running.value)

function ensureSocket() {
  if (socket.value)
    return socket.value

  socket.value = getCompareSocket({
    onConnect: () => {},
    onDisconnect: () => {
      running.value = false
    },
    onMessage: (data) => {
      const key = data.target
      if (!key || !(key in results.value))
        return

      if (data.error) {
        results.value[key] = { ...results.value[key], done: true, error: data.error }
      }
      else if (data.stopped) {
        results.value[key] = {
          ...results.value[key],
          done: true,
          error: results.value[key].content
            ? null
            : 'Stopped.',
        }
      }
      else if (data.done) {
        results.value[key] = { ...results.value[key], done: true, stats: data.stats ?? null }
      }
      else {
        results.value[key] = { ...results.value[key], content: results.value[key].content + (data.message ?? '') }
      }

      if (Object.values(results.value).every(r => r.done))
        running.value = false
    },
    onMaxReconnectFailed: () => {
      snackbarStore.showSnackbarError('Lost connection to the server.')
      running.value = false
    },
  }, wsBaseURL)

  return socket.value
}

function runCompare() {
  if (!canRun.value)
    return

  const targets = selectableModels.value.filter(m => selectedKeys.value.includes(m.key))
  lastPrompt.value = prompt.value.trim()
  pickedKey.value = null
  displayOrder.value = blind.value
    ? shuffled(targets.map(t => t.key))
    : targets.map(t => t.key)
  results.value = Object.fromEntries(
    targets.map(t => [t.key, { content: '', done: false, error: null, stats: null } as ResultState]),
  )
  running.value = true

  ensureSocket().send(prompt.value.trim(), targets.map(t => ({ model: t.model, parameters: t.parameters })))
}

function stopCompare() {
  socket.value?.stop()
}

// Blind test: answers are shown as "Answer A/B/…" in a shuffled order
// until the user picks the one they like best, so the model's reputation
// can't sway the judgement.
const scoreboard = ref<{ model: string, wins: number, matches: number, win_rate: number }[]>([])
const clearScoresDialog = ref(false)
const { api } = storeToRefs(useApiStore())

const revealed = computed(() => !blind.value || pickedKey.value !== null)
const canPick = computed(() => blind.value && !running.value && pickedKey.value === null
  && displayOrder.value.filter(k => results.value[k]?.content && !results.value[k]?.error).length >= 2)

function shuffled<T>(items: T[]): T[] {
  const copy = [...items]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]]
  }

  return copy
}

async function fetchScoreboard() {
  try {
    scoreboard.value = (await api.value.get('compare-votes/')).data
  }
  catch (error: any) {
    console.error(error)
  }
}

async function pickBest(key: string) {
  pickedKey.value = key
  const contenders = displayOrder.value.filter(k => results.value[k]?.content && !results.value[k]?.error)
  try {
    await api.value.post('compare-votes/', { prompt: lastPrompt.value, winner: key, contenders })
    await fetchScoreboard()
  }
  catch (error: any) {
    console.error(error)
    snackbarStore.showSnackbarError(error.response?.data?.error || 'Could not save your pick.')
  }
}

async function clearScoreboard() {
  clearScoresDialog.value = false
  try {
    await api.value.delete('compare-votes/')
    scoreboard.value = []
  }
  catch (error: any) {
    console.error(error)
    snackbarStore.showSnackbarError('Could not clear the scoreboard.')
  }
}

function panelLabel(key: string): string {
  if (revealed.value)
    return labelFor(key)

  return `Answer ${String.fromCharCode(65 + displayOrder.value.indexOf(key))}`
}

const continuingKey = ref<string | null>(null)

async function continueInChat(key: string) {
  const target = selectableModels.value.find(m => m.key === key)
  const result = results.value[key]
  if (!target || !result?.content)
    return

  continuingKey.value = key
  const chatId = await chatStore.chatFromCompare(target.model, target.parameters, lastPrompt.value, result.content)
  continuingKey.value = null

  if (chatId)
    await navigateTo({ path: '/chat', query: { model: target.model, parameters: target.parameters, chat: chatId } })
}

function labelFor(key: string): string {
  return selectableModels.value.find(m => m.key === key)?.name ?? key
}
</script>

<template>
  <AppTopBar />

  <div class="compare-page">
    <h1 class="page-h1">
      Compare
    </h1>

    <p class="page-lede">
      Run one prompt across up to {{ MAX_TARGETS }} models at once and read their answers side by side.
    </p>

    <div class="prompt-card">
      <div
        v-if="!selectableModels.length"
        class="empty-hint"
      >
        No models installed yet.
        <NuxtLink to="/models">
          Go to Models →
        </NuxtLink>
      </div>

      <template v-else>
        <div class="mono-kicker mb-2">
          Pick up to four
        </div>

        <div class="model-chip-row">
          <button
            v-for="m in selectableModels"
            :key="m.key"
            type="button"
            class="model-chip"
            :class="{
              'model-chip--active': selectedKeys.includes(m.key),
              'model-chip--stopped': m.status !== 'running',
            }"
            @click="toggleTarget(m)"
          >
            {{ m.name }} ({{ m.parameters }}){{ m.status !== 'running'
              ? ' · stopped'
              : '' }}
          </button>
        </div>
      </template>

      <v-textarea
        v-model="prompt"
        placeholder="Prompt"
        rows="3"
        hide-details
        density="compact"
        class="prompt-textarea"
        :disabled="running"
      />

      <div class="prompt-footer">
        <span class="mono-kicker">{{ selectedKeys.length }} of {{ MAX_TARGETS }} models selected</span>

        <v-spacer />

        <div
          class="mode-pills"
          role="radiogroup"
          aria-label="Comparison mode"
        >
          <button
            type="button"
            class="mode-pill"
            :class="{'mode-pill--active': !blind}"
            :disabled="running"
            title="See which model wrote each answer"
            @click="setBlind(false)"
          >
            Show models
          </button>

          <button
            type="button"
            class="mode-pill"
            :class="{'mode-pill--active': blind}"
            :disabled="running"
            title="Hide the model names until you pick the best answer"
            @click="setBlind(true)"
          >
            Blind test
          </button>
        </div>

        <v-btn
          v-if="!running"
          color="mint-btn"
          variant="flat"
          rounded="lg"
          :disabled="!canRun"
          @click="runCompare"
        >
          Compare
        </v-btn>

        <v-btn
          v-else
          variant="outlined"
          color="amber"
          rounded="lg"
          prepend-icon="mdi-stop"
          @click="stopCompare"
        >
          Stop
        </v-btn>
      </div>
    </div>

    <p
      v-if="canPick"
      class="pick-hint"
    >
      Read the answers, then pick the one you like best — the model names are revealed after you choose.
    </p>

    <div
      v-if="Object.keys(results).length"
      class="results-grid"
    >
      <div
        v-for="key in displayOrder.filter(k => k in results)"
        :key="key"
        class="result-panel"
        :class="{'result-panel--picked': pickedKey === key}"
      >
        <div class="result-header">
          <span class="result-dot" />

          <span class="result-name">{{ panelLabel(key) }}</span>

          <span
            v-if="pickedKey === key"
            class="result-picked"
          >Your pick</span>
        </div>

        <div class="result-body">
          <div
            v-if="!results[key].content && !results[key].done"
            class="result-loading"
          >
            <span />

            <span />

            <span />
          </div>

          <div
            v-if="results[key].error"
            class="result-error"
          >
            {{ results[key].error }}
          </div>

          <div
            v-else
            class="result-content"
          >
            <MessageParts :parts="splitMessageRaw(results[key].content).parts" />
          </div>
        </div>

        <div
          v-if="results[key].stats && revealed"
          class="result-stats font-mono"
        >
          {{ formatStats(results[key].stats!) }}
        </div>

        <v-btn
          v-if="canPick && results[key].content && !results[key].error"
          size="small"
          color="mint-btn"
          variant="flat"
          class="result-continue"
          @click="pickBest(key)"
        >
          This one's best
        </v-btn>

        <v-btn
          v-if="revealed && results[key].done && !results[key].error && results[key].content"
          size="small"
          variant="text"
          class="result-continue"
          :loading="continuingKey === key"
          @click="continueInChat(key)"
        >
          Continue in chat →
        </v-btn>
      </div>
    </div>

    <section
      v-if="scoreboard.length"
      class="scoreboard"
    >
      <div class="scoreboard-head">
        <span class="mono-kicker">Your blind-test scoreboard</span>

        <button
          type="button"
          class="scoreboard-clear"
          @click="clearScoresDialog = true"
        >
          Clear
        </button>
      </div>

      <div
        v-for="row in scoreboard"
        :key="row.model"
        class="scoreboard-row"
      >
        <span class="scoreboard-model font-mono">{{ row.model }}</span>

        <span class="scoreboard-bar">
          <span
            class="scoreboard-fill"
            :style="{'width': `${Math.round(row.win_rate * 100)}%`}"
          />
        </span>

        <span class="scoreboard-score">won {{ row.wins }} of {{ row.matches }}</span>
      </div>
    </section>
  </div>

  <ConfirmDialog
    v-model="clearScoresDialog"
    title="Clear scoreboard"
    message="This deletes every blind-test pick you've made. This can't be undone."
    confirm-label="Clear"
    confirm-color="red"
    @confirm="clearScoreboard"
    @cancel="clearScoresDialog = false"
  />
</template>

<style scoped>
.pick-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--color-mint-deep);
}

.result-panel--picked {
  border-color: var(--color-mint-border) !important;
  box-shadow: 0 0 0 2px var(--color-mint-tint);
}

.result-picked {
  margin-left: auto;
  font-size: 11px;
  color: var(--color-mint-deep);
}

.mode-pills {
  display: inline-flex;
  padding: 2px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  margin-right: 8px;
}

.mode-pill {
  padding: 4px 12px;
  border: none;
  border-radius: 18px;
  background: transparent;
  font-family: var(--font-sans);
  font-size: 12px;
  color: var(--color-ink-2);
  cursor: pointer;
}

.mode-pill--active {
  background: var(--color-mint-tint);
  color: var(--color-mint-ink);
}

.mode-pill:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.scoreboard {
  margin-top: 28px;
  padding: 16px 18px;
  border-radius: 13px;
  border: 1px solid var(--color-line-2);
  background: var(--color-card);
}

.scoreboard-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.scoreboard-clear {
  border: none;
  background: transparent;
  font-size: 11.5px;
  color: var(--color-ink-3);
  cursor: pointer;
}

.scoreboard-row {
  display: grid;
  grid-template-columns: minmax(120px, 220px) 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 4px 0;
  font-size: 12.5px;
}

.scoreboard-model {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--color-ink);
}

.scoreboard-bar {
  height: 6px;
  border-radius: 3px;
  background: var(--color-soft-2);
  overflow: hidden;
}

.scoreboard-fill {
  display: block;
  height: 100%;
  background: var(--color-mint);
}

.scoreboard-score {
  color: var(--color-ink-3);
}
.compare-page {
  max-width: 1080px;
  margin: 0 auto;
  padding: 28px 20px 60px;
}

.page-h1 {
  font-size: 30px;
  font-weight: 400;
  letter-spacing: -0.025em;
  color: var(--color-ink);
  margin: 0 0 6px;
}

.page-lede {
  font-size: 14px;
  color: var(--color-ink-2);
  margin: 0 0 22px;
}

.prompt-card {
  background: var(--color-card);
  border: 1px solid var(--color-line-2);
  border-radius: 17px;
  padding: 20px;
  margin-bottom: 22px;
}

.empty-hint {
  font-size: 13px;
  color: var(--color-ink-2);
}

.empty-hint a {
  color: var(--color-mint-deep);
}

.model-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.model-chip {
  padding: 6px 13px;
  border-radius: 20px;
  border: 1px solid var(--color-line);
  background: var(--color-soft);
  color: var(--color-ink-2);
  font-size: 12.5px;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.model-chip--stopped {
  opacity: 0.6;
}

.model-chip--active {
  background: var(--color-mint-tint);
  border-color: var(--color-mint-border);
  color: oklch(0.4 0.07 168);
  opacity: 1;
}

.prompt-textarea {
  margin-bottom: 10px;
}

.prompt-footer {
  display: flex;
  align-items: center;
  gap: 12px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}

.result-panel {
  background: var(--color-soft);
  border: 1px solid var(--color-line-2);
  border-radius: 14px;
  padding: 16px;
  min-height: 180px;
  display: flex;
  flex-direction: column;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 10px;
}

.result-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-mint);
  flex-shrink: 0;
}

.result-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-ink);
}

.result-body {
  flex: 1;
}

.result-loading {
  display: flex;
  gap: 5px;
}

.result-loading span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-ink-3);
  animation: dots 1.2s infinite;
}

.result-loading span:nth-child(1) { animation-delay: 0s; }
.result-loading span:nth-child(2) { animation-delay: 0.2s; }
.result-loading span:nth-child(3) { animation-delay: 0.4s; }

.result-error {
  color: var(--color-red);
  font-size: 13px;
}

.result-content {
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--color-ink);
}

.result-stats {
  margin-top: 10px;
  font-size: 10.5px;
  color: var(--color-ink-3);
}
</style>
