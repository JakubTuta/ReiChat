<script setup lang="ts">
const props = defineProps<{
  modelRunning: boolean
  waitingForResponse: boolean
  // Everything canSend needs beyond modelRunning/waitingForResponse/message:
  // an active chat, a live socket, and no reconnect in progress.
  sendGateOk: boolean
  showImageButton: boolean
}>()

const emit = defineEmits<{
  (e: 'send', payload: { message: string, image: string }): void
  (e: 'stop'): void
  (e: 'imageAttached'): void
}>()

const message = defineModel<string>('message', { default: '' })

const image = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const composerTextarea = ref<HTMLTextAreaElement | null>(null)

const inputsDisabled = computed(() => !props.modelRunning || props.waitingForResponse)
const canSend = computed(() => props.modelRunning && !props.waitingForResponse && props.sendGateOk && !!message.value.trim(),
)

function autoGrow(el: HTMLTextAreaElement) {
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 128)}px`
}

function onTextareaInput(event: Event) {
  autoGrow(event.target as HTMLTextAreaElement)
}

// Covers programmatic changes (e.g. a suggestion chip filling the composer)
// — direct typing is already handled synchronously by onTextareaInput.
watch(message, () => {
  nextTick(() => {
    if (composerTextarea.value)
      autoGrow(composerTextarea.value)
  })
})

function send() {
  if (!canSend.value)
    return

  emit('send', { message: message.value, image: image.value })
}

function stop() {
  emit('stop')
}

function clearImage() {
  image.value = ''
}

function toBase64(file: Blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => resolve(reader.result)
    reader.onerror = error => reject(error)
  })
}

async function handleImageUpload(event: Event) {
  const target = event.target as HTMLInputElement
  if (!target.files?.length)
    return

  const file = target.files[0]
  const fileInBase64 = await toBase64(file) as string
  image.value = fileInBase64

  // A file input doesn't fire `change` again if the same file is picked
  // twice in a row (its value string is unchanged) — reset it so
  // re-attaching the same image isn't a silent no-op.
  target.value = ''

  emit('imageAttached')
}

function focus() {
  composerTextarea.value?.focus()
}

// Only called once the caller (ChatCard) has confirmed the send actually
// went out over an OPEN socket — clearing eagerly would wipe what the user
// typed even when "Connection lost" bounced the send.
function clear() {
  message.value = ''
  image.value = ''

  if (composerTextarea.value)
    composerTextarea.value.style.height = 'auto'
}

defineExpose({ focus, clear })
</script>

<template>
  <div
    class="composer-bar"
    :class="{'composer-disabled': !modelRunning}"
  >
    <div
      v-if="image"
      class="composer-image-preview"
    >
      <!--
        A v-badge with a click handler isn't a focusable, operable
        control on its own — keyboard users had no way to remove an
        attached image. A real button overlaid on the preview is.
      -->
      <div class="image-preview-wrap">
        <img
          :src="image"
          alt="Uploaded image"
          style="max-height: 80px; border-radius: 8px"
        >

        <v-btn
          icon="$mdiClose"
          color="error"
          size="x-small"
          class="image-remove-btn"
          title="Remove attached image"
          aria-label="Remove attached image"
          @click="clearImage"
        />
      </div>
    </div>

    <div
      class="composer-input-wrap"
      @click="focus"
    >
      <textarea
        ref="composerTextarea"
        v-model="message"
        placeholder="Message…"
        aria-label="Message"
        rows="1"
        class="composer-native-textarea"
        :disabled="inputsDisabled"
        @input="onTextareaInput"
        @keydown.enter.exact.prevent="send"
        @keydown.shift.enter.exact.stop
      />
    </div>

    <div class="composer-footer">
      <button
        v-if="showImageButton"
        type="button"
        class="composer-icon-btn"
        title="Attach image"
        aria-label="Attach image"
        :disabled="inputsDisabled"
        @click="fileInput?.click()"
      >
        <v-icon
          size="18"
          icon="$mdiImagePlus"
        />

        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          style="display: none"
          @change="handleImageUpload"
        >
      </button>

      <span class="composer-footer-spacer" />

      <span
        v-if="waitingForResponse"
        class="composer-hint font-mono"
      >
        esc to stop
      </span>

      <span
        v-else-if="canSend"
        class="composer-hint font-mono"
      >
        ⏎ send
      </span>

      <button
        v-if="waitingForResponse"
        type="button"
        class="send-btn send-btn--active"
        title="Stop generating"
        aria-label="Stop generating"
        @click="stop"
      >
        <v-icon
          size="16"
          icon="$mdiStop"
        />
      </button>

      <button
        v-else
        type="button"
        class="send-btn"
        :class="{'send-btn--active': canSend}"
        title="Send message"
        aria-label="Send message"
        :disabled="!canSend"
        @click="send"
      >
        <v-icon
          size="18"
          icon="$mdiArrowUp"
        />
      </button>
    </div>
  </div>
</template>

<style scoped>
.composer-bar {
  border: 1px solid var(--color-line);
  border-radius: 17px;
  padding: 13px 12px 10px 16px;
  background: var(--color-card);
  box-shadow: var(--shadow-composer);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  flex-shrink: 0;
}

.composer-bar:focus-within {
  border-color: var(--color-mint-border);
}

.composer-disabled {
  opacity: 0.65;
}

.composer-input-wrap {
  cursor: text;
}

.composer-native-textarea {
  width: 100%;
  resize: none;
  border: none;
  outline: none;
  background: transparent;
  font: inherit;
  font-size: 14.5px;
  line-height: 1.5;
  color: var(--color-ink);
  min-height: 22px;
  max-height: 120px;
  overflow-y: auto;
  display: block;
}

.composer-native-textarea::placeholder {
  color: var(--color-ink-3);
}

.composer-native-textarea:disabled {
  cursor: not-allowed;
}

.composer-image-preview {
  padding: 0 0 8px 0;
  display: flex;
  justify-content: flex-end;
}

.image-preview-wrap {
  position: relative;
  display: inline-block;
}

.image-remove-btn {
  position: absolute;
  top: -8px;
  right: -8px;
}

.composer-footer {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 6px;
}

.composer-footer-spacer {
  flex: 1;
}

.composer-hint {
  font-size: 10.5px;
  color: var(--color-ink-3);
}

.composer-icon-btn {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: var(--color-ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.composer-icon-btn:hover:not(:disabled) {
  background: var(--color-soft);
}

.composer-icon-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.send-btn {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: none;
  background: oklch(0.9 0.03 168);
  color: var(--color-ink-3);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.15s ease, background-color 0.15s ease, color 0.15s ease;
  flex-shrink: 0;
}

.send-btn--active {
  background: var(--color-mint-btn);
  color: var(--color-mint-ink);
}

.send-btn--active:hover {
  background: var(--color-mint-btn-hover);
}

.send-btn:disabled {
  cursor: not-allowed;
}

.send-btn--active:active {
  transform: scale(0.88);
}
</style>
