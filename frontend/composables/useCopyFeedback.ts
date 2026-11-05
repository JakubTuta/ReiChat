export function useCopyFeedback(timeoutMs = 1500) {
  const copiedKey = ref<string | null>(null)
  const snackbarStore = useSnackbarStore()
  let timer: ReturnType<typeof setTimeout> | null = null

  // "Copied" is only shown once the clipboard write actually succeeded —
  // the browser refuses it when the page isn't focused or clipboard access
  // is blocked, and the user would otherwise paste something stale.
  async function copy(text: string, key = 'default') {
    try {
      await navigator.clipboard.writeText(text)
    }
    catch (error) {
      console.error(error)
      snackbarStore.showSnackbarError('Couldn\'t copy — your browser blocked clipboard access. Select the text and copy it by hand.')

      return
    }

    copiedKey.value = key

    if (timer)
      clearTimeout(timer)

    timer = setTimeout(() => {
      copiedKey.value = null
    }, timeoutMs)
  }

  return { copiedKey, copy }
}
