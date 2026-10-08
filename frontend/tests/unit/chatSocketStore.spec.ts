import { createPinia, defineStore, setActivePinia, storeToRefs } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

// stores/chatSocketStore.ts is written for Nuxt's auto-import environment
// (`defineStore`/`ref`/`storeToRefs`/`useChatStore`/`useSnackbarStore`
// resolved globally rather than imported) — see snackbarStore.spec.ts for
// the same pattern.
vi.stubGlobal('defineStore', defineStore)
vi.stubGlobal('ref', ref)
vi.stubGlobal('storeToRefs', storeToRefs)

const chatHistoryPerModel = ref<Record<string, any[]>>({})
const fetchChatHistory = vi.fn(async () => {})
vi.stubGlobal('useChatStore', () => ({ chatHistoryPerModel, fetchChatHistory }))

const showSnackbarError = vi.fn()
vi.stubGlobal('useSnackbarStore', () => ({ showSnackbarError }))

let instances: FakeWebSocket[]

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = FakeWebSocket.CONNECTING
  sent: string[] = []
  private listeners: Record<string, Array<(event?: any) => void>> = {}

  constructor(public url: string) {
    instances.push(this)
  }

  addEventListener(type: string, cb: (event?: any) => void) {
    (this.listeners[type] ||= []).push(cb)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
    this.dispatch('close')
  }

  dispatch(type: string, event: any = {}) {
    for (const cb of this.listeners[type] ?? []) cb(event)
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.dispatch('open')
  }

  receive(data: unknown) {
    this.dispatch('message', { data: JSON.stringify(data) })
  }
}

vi.stubGlobal('WebSocket', FakeWebSocket)

const { useChatSocketStore } = await import('~/stores/chatSocketStore')

beforeEach(() => {
  instances = []
  chatHistoryPerModel.value = {}
  showSnackbarError.mockClear()
  fetchChatHistory.mockClear()
  setActivePinia(createPinia())
})

describe('chatSocketStore', () => {
  it('reuses the live socket when reconnecting to the same room instead of tearing it down', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    instances[0].open()

    // Simulates ChatCard remounting after the user navigated away and back
    // — the whole point of moving the socket into a store is that this must
    // NOT close and reopen the connection, or a response streaming while
    // the component was unmounted would be lost.
    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')

    expect(instances).toHaveLength(1)
  })

  it('closes the old socket and opens a fresh one when switching to a different room', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    instances[0].open()

    store.connectToChat('room2', 'llama3.1', 'http://localhost:8000/')

    expect(instances).toHaveLength(2)
    expect(instances[0].readyState).toBe(FakeWebSocket.CLOSED)
  })

  it('a response that finishes streaming lands in chat history even with no component reading store state', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    instances[0].open()

    const sent = store.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })
    expect(sent).toBe(true)

    // Stream two token chunks, then the final frame — as if this happened
    // while the user had already navigated to a different page.
    instances[0].receive({ message: 'Hel', done: false })
    instances[0].receive({ message: 'lo', done: false })
    instances[0].receive({ message: 'Hello', done: true })

    expect(chatHistoryPerModel.value['llama3.1']).toEqual([
      { role: 'user', content: 'hi', image: '' },
      { role: 'assistant', content: 'Hello', image: '' },
    ])
    expect(store.waitingForResponse).toBe(false)
    expect(store.botResponse).toBe('')
  })

  it('sendMessage keys the optimistic user message off the message itself, not an external selection', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'mistral', 'http://localhost:8000/')
    instances[0].open()

    store.sendMessage({ message: 'hi', ai_model: 'mistral', ai_model_parameters: '' })

    expect(chatHistoryPerModel.value.mistral).toEqual([{ role: 'user', content: 'hi', image: '' }])
  })

  it('sendMessage refuses to send while the socket is not open', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    // never opened

    const sent = store.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '' })

    expect(sent).toBe(false)
    expect(chatHistoryPerModel.value.llama3_1).toBeUndefined()
  })

  it('a failed prompt reloads history from the server and hands the text back', () => {
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    instances[0].open()
    store.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })

    instances[0].receive({ error: 'Model container is not running.', done: true })

    // The optimistic user message was never stored server-side — the list
    // has to come back from the server, not keep the phantom message.
    expect(fetchChatHistory).toHaveBeenCalledWith('llama3.1', 'room1')
    expect(store.failedPrompt).toBe('hi')
    expect(store.waitingForResponse).toBe(false)
  })

  it('a reconnect drops the half-streamed reply and reloads history', () => {
    vi.useFakeTimers()
    const store = useChatSocketStore()

    store.connectToChat('room1', 'llama3.1', 'http://localhost:8000/')
    instances[0].open()
    expect(fetchChatHistory).not.toHaveBeenCalled()

    store.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })
    instances[0].receive({ message: 'partial', done: false })
    instances[0].readyState = FakeWebSocket.CLOSED
    instances[0].dispatch('close')
    vi.advanceTimersByTime(1000)
    vi.useRealTimers()

    instances[1].open()

    expect(store.botResponse).toBe('')
    expect(fetchChatHistory).toHaveBeenCalledWith('llama3.1', 'room1')
  })
})
