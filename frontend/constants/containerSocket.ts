export interface PullProgressInfo {
  phase: string
  percent: number
  detail: string
}

export interface ScrapeProgressInfo {
  running: boolean
  total: number | null
  completed: number
  current: string | null
  error: string | null
}

export interface ContainerInfo {
  name: string
  status: string
  port: string | null
  environment: { model: string | null, parameters: string | null }
}

export type ContainerSocketMessageType
  = | 'snapshot'
    | 'image_pull_progress'
    | 'model_pull_progress'
    | 'scrape_progress'
    | 'container_update'
    | 'ollama_update'

export interface ContainerSocketMessage {
  type: ContainerSocketMessageType
  containers?: ContainerInfo[]
  progress?: Record<string, PullProgressInfo>
  scrape?: ScrapeProgressInfo
  container?: string
  phase?: string
  percent?: number
  detail?: string
  error?: string
  // container_update: a plain-language success message (e.g. after an
  // Ollama update) rather than an error.
  info?: string
  running?: boolean
  total?: number | null
  completed?: number
  current?: string | null
}

export interface ContainerSocketHandlers {
  onConnect: () => void
  onDisconnect: () => void
  onReconnecting: (attempt: number) => void
  onMessage: (data: ContainerSocketMessage) => void
  onMaxReconnectFailed?: () => void
}

export interface ContainerSocketWrapper {
  closeConnection: () => void
}

const RECONNECT_BASE_DELAY_MS = 1000
const RECONNECT_MAX_DELAY_MS = 30_000
const RECONNECT_MAX_ATTEMPTS = 8

export function getContainerSocket(handlers: ContainerSocketHandlers): ContainerSocketWrapper {
  const runtimeConfig = useRuntimeConfig()
  const baseURL = runtimeConfig.public.serverUrl

  const url = new URL(baseURL)
  const socketHost = url.host
  const scheme = url.protocol === 'https:'
    ? 'wss'
    : 'ws'

  let reconnectAttempts = 0
  let intentionallyClosed = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let currentSocket: WebSocket | null = null

  function buildSocket(): WebSocket {
    const ws = new WebSocket(`${scheme}://${socketHost}/ws/containers/`)
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
        handlers.onReconnecting(reconnectAttempts)

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
        const data: ContainerSocketMessage = JSON.parse(event.data)
        handlers.onMessage(data)
      }
      catch (error) {
        console.error('Failed to parse container socket message:', error)
      }
    })

    return ws
  }

  buildSocket()

  return {
    closeConnection: () => {
      intentionallyClosed = true
      if (reconnectTimer)
        clearTimeout(reconnectTimer)
      if (currentSocket && currentSocket.readyState !== WebSocket.CLOSED)
        currentSocket.close()
    },
  }
}
