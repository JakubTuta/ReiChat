import type { WebsocketHandlers } from '~/constants/websocket'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getWebsocket } from '~/constants/websocket'

let instances: FakeWebSocket[]
let handlers: { [K in keyof WebsocketHandlers]-?: WebsocketHandlers[K] }

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
}

beforeEach(() => {
  instances = []
  vi.stubGlobal('WebSocket', FakeWebSocket)
  handlers = {
    onConnect: vi.fn(),
    onDisconnect: vi.fn(),
    onSendMessage: vi.fn(),
    onReceiveMessage: vi.fn(),
    onError: vi.fn(),
    onNotice: vi.fn(),
    onChatTitle: vi.fn(),
    onReconnecting: vi.fn(),
    onMaxReconnectFailed: vi.fn(),
  }
})

describe('getWebsocket reconnect wrapper', () => {
  it('readyState always reflects the currently-live socket, even after a reconnect', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    const first = instances[0]
    first.readyState = FakeWebSocket.OPEN
    expect(wrapper.readyState).toBe(FakeWebSocket.OPEN)

    vi.useFakeTimers()
    first.readyState = FakeWebSocket.CLOSED
    first.dispatch('close') // not intentional -> schedules a reconnect
    vi.advanceTimersByTime(1000) // base backoff delay
    vi.useRealTimers()

    expect(instances).toHaveLength(2)
    const second = instances[1]
    second.readyState = FakeWebSocket.OPEN

    // Before the fix, `Object.assign` missed `readyState` (a prototype
    // getter on the old socket), so this kept reporting the dead socket's
    // state forever after any reconnect.
    expect(wrapper.readyState).toBe(FakeWebSocket.OPEN)

    first.readyState = FakeWebSocket.CLOSED
    expect(wrapper.readyState).toBe(FakeWebSocket.OPEN) // still reading `second`, not `first`
  })

  it('routes chat_title frames to onChatTitle', () => {
    getWebsocket(handlers, 'room1', 'http://localhost:8000/')

    instances[0].dispatch('message', { data: JSON.stringify({ chat_title: 'Trip to Rome' }) })

    expect(handlers.onChatTitle).toHaveBeenCalledWith('Trip to Rome')
    expect(handlers.onReceiveMessage).not.toHaveBeenCalled()
  })

  it('routes notice frames to onNotice, not onReceiveMessage', () => {
    getWebsocket(handlers, 'room1', 'http://localhost:8000/')

    instances[0].dispatch('message', { data: JSON.stringify({ notice: 'Docs not searched.', done: false }) })

    expect(handlers.onNotice).toHaveBeenCalledWith('Docs not searched.')
    expect(handlers.onReceiveMessage).not.toHaveBeenCalled()
  })

  it('sends on the current socket after a reconnect, not the dead one', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    const first = instances[0]
    first.readyState = FakeWebSocket.OPEN

    vi.useFakeTimers()
    first.readyState = FakeWebSocket.CLOSED
    first.dispatch('close')
    vi.advanceTimersByTime(1000)
    vi.useRealTimers()

    const second = instances[1]
    second.readyState = FakeWebSocket.OPEN

    wrapper.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })

    expect(second.sent).toHaveLength(1)
    expect(first.sent).toHaveLength(0)
    expect(handlers.onSendMessage).toHaveBeenCalledTimes(1)
  })

  it('tags a sent prompt with type "prompt" for the consumer dispatcher', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.OPEN

    wrapper.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })

    expect(JSON.parse(instances[0].sent[0])).toEqual({
      type: 'prompt',
      message: 'hi',
      ai_model: 'llama3.1',
      ai_model_parameters: '8b',
    })
  })

  it('stopGeneration sends a stop frame without going through onSendMessage', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.OPEN

    wrapper.stopGeneration()

    expect(JSON.parse(instances[0].sent[0])).toEqual({ type: 'stop' })
    expect(handlers.onSendMessage).not.toHaveBeenCalled()
  })

  it('stopGeneration warns instead of throwing while the socket is not open', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.CONNECTING

    wrapper.stopGeneration()

    expect(instances[0].sent).toHaveLength(0)
    expect(warnSpy).toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it('regenerate sends a regenerate frame without going through onSendMessage', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.OPEN

    wrapper.regenerate({ ai_model: 'llama3.1', ai_model_parameters: '8b' })

    expect(JSON.parse(instances[0].sent[0])).toEqual({
      type: 'regenerate',
      ai_model: 'llama3.1',
      ai_model_parameters: '8b',
    })
    expect(handlers.onSendMessage).not.toHaveBeenCalled()
  })

  it('editResend sends an edit_resend frame carrying the target index', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.OPEN

    wrapper.editResend({ index: 2, message: 'edited', ai_model: 'llama3.1', ai_model_parameters: '8b' })

    expect(JSON.parse(instances[0].sent[0])).toEqual({
      type: 'edit_resend',
      index: 2,
      message: 'edited',
      ai_model: 'llama3.1',
      ai_model_parameters: '8b',
    })
    expect(handlers.onSendMessage).not.toHaveBeenCalled()
  })

  it('does not send, and warns, while the socket is not open', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')
    instances[0].readyState = FakeWebSocket.CONNECTING

    wrapper.sendMessage({ message: 'hi', ai_model: 'llama3.1', ai_model_parameters: '8b' })

    expect(instances[0].sent).toHaveLength(0)
    expect(handlers.onSendMessage).not.toHaveBeenCalled()
    expect(warnSpy).toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it('an intentional close does not reconnect or fire onDisconnect', () => {
    const wrapper = getWebsocket(handlers, 'room1', 'http://localhost:8000/')

    wrapper.closeConnection()

    expect(handlers.onDisconnect).not.toHaveBeenCalled()
    expect(instances).toHaveLength(1)
  })

  it('gives up after the max reconnect attempts and calls onMaxReconnectFailed', () => {
    getWebsocket(handlers, 'room1', 'http://localhost:8000/')

    vi.useFakeTimers()
    for (let i = 0; i < 8; i++) {
      const socket = instances[instances.length - 1]
      socket.readyState = FakeWebSocket.CLOSED
      socket.dispatch('close')
      vi.runOnlyPendingTimers()
    }
    // The 9th failure: attempts are already at the max, so no further
    // reconnect gets scheduled — this used to leave the UI stuck on
    // "reconnecting..." forever with no way to recover.
    const last = instances[instances.length - 1]
    last.readyState = FakeWebSocket.CLOSED
    last.dispatch('close')
    vi.useRealTimers()

    expect(handlers.onReconnecting).toHaveBeenCalledTimes(8)
    expect(handlers.onReconnecting).toHaveBeenNthCalledWith(1, 1)
    expect(handlers.onReconnecting).toHaveBeenNthCalledWith(8, 8)
    expect(handlers.onMaxReconnectFailed).toHaveBeenCalledTimes(1)
    expect(instances).toHaveLength(9) // initial socket + 8 reconnects
  })

  it('resets the backoff counter after a successful reconnect', () => {
    getWebsocket(handlers, 'room1', 'http://localhost:8000/')

    vi.useFakeTimers()
    instances[0].readyState = FakeWebSocket.CLOSED
    instances[0].dispatch('close')
    vi.runOnlyPendingTimers()

    instances[1].dispatch('open') // reconnect succeeded
    instances[1].readyState = FakeWebSocket.CLOSED
    instances[1].dispatch('close')
    vi.runOnlyPendingTimers()
    vi.useRealTimers()

    // Second failure after a successful reconnect is attempt #1 again, not #2.
    expect(handlers.onReconnecting).toHaveBeenNthCalledWith(1, 1)
    expect(handlers.onReconnecting).toHaveBeenNthCalledWith(2, 1)
  })
})
