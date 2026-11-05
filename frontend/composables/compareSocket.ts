export interface CompareTarget {
  model: string
  parameters: string
}

export interface CompareStats {
  prompt_tokens?: number
  completion_tokens?: number
  tokens_per_second?: number
}

export interface CompareResponse {
  target: string | null
  message?: string
  done: boolean
  error?: string
  stats?: CompareStats
  stopped?: boolean
}

export interface CompareSocketHandlers {
  onConnect: () => void
  onDisconnect: () => void
  onMessage: (data: CompareResponse) => void
  onReconnecting?: (attempt: number) => void
  onMaxReconnectFailed?: () => void
}

export interface CompareSocketWrapper {
  readonly readyState: number
  send: (prompt: string, targets: CompareTarget[]) => void
  stop: () => void
  closeConnection: () => void
}

const RECONNECT_BASE_DELAY_MS = 1000
const RECONNECT_MAX_DELAY_MS = 30_000
const RECONNECT_MAX_ATTEMPTS = 8

export function getCompareSocket(handlers: CompareSocketHandlers, baseURL: string): CompareSocketWrapper {
  const url = new URL(baseURL)
  const socketHost = url.host
  const scheme = url.protocol === 'https:'
    ? 'wss'
    : 'ws'

  let reconnectAttempts = 0
  let intentionallyClosed = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let currentSocket: WebSocket

  function buildSocket(): WebSocket {
    const ws = new WebSocket(`${scheme}://${socketHost}/ws/compare/`)
    currentSocket = ws

    ws.addEventListener('open', () => {
      reconnectAttempts = 0
      handlers.onConnect()
    })

    ws.addEventListener('close', () => {
      if (intentionallyClosed)
        return

      handlers.onDisconnect()

      if (reconnectAttempts < RECONNECT_MAX_ATTEMPTS) {
        const delay = Math.min(
          RECONNECT_BASE_DELAY_MS * 2 ** reconnectAttempts,
          RECONNECT_MAX_DELAY_MS,
        )
        reconnectAttempts++

        handlers.onReconnecting?.(reconnectAttempts)

        reconnectTimer = setTimeout(() => {
          buildSocket()
        }, delay)
      }
      else {
        handlers.onMaxReconnectFailed?.()
      }
    })

    ws.addEventListener('message', (event) => {
      try {
        const data: CompareResponse = JSON.parse(event.data)
        handlers.onMessage(data)
      }
      catch (error) {
        console.error('Failed to parse compare WebSocket message:', error)
      }
    })

    return ws
  }

  currentSocket = buildSocket()

  return {
    get readyState() {
      return currentSocket.readyState
    },
    send: (prompt: string, targets: CompareTarget[]) => {
      if (currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ prompt, targets }))
      }
      else {
        console.warn('Compare WebSocket is not open. Prompt not sent.')
      }
    },
    stop: () => {
      if (currentSocket.readyState === WebSocket.OPEN)
        currentSocket.send(JSON.stringify({ type: 'stop' }))
    },
    closeConnection: () => {
      intentionallyClosed = true
      if (reconnectTimer)
        clearTimeout(reconnectTimer)
      if (currentSocket.readyState !== WebSocket.CLOSED)
        currentSocket.close()
    },
  }
}
