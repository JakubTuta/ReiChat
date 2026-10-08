export interface WebsocketMessage {
  message: string
  ai_model: string
  ai_model_parameters: string
  image?: string
  structured_output?: {
    field: string
    type: string
  }[]
}

export interface GenerationStats {
  prompt_tokens?: number
  completion_tokens?: number
  tokens_per_second?: number
}

// How full the model's memory (context window) was for one reply — see
// django_app/context_window.py. `used` and `limit` are in tokens.
export interface ContextReport {
  limit: number
  used: number
  model_max: number | null
  preset: string | null
  // How many of the oldest messages didn't fit and weren't sent.
  dropped: number
  // Older messages were replaced by a running summary.
  summarized: boolean
  // Share (0–1) of the model that fit in GPU memory for this reply; null
  // when no GPU is involved.
  gpu_share?: number | null
}

export interface Citation {
  document_id: number
  filename: string
  chunk_index: number
}

// An MCP tool call waiting for the user's Allow/Deny — see
// ChatConsumer._request_tool_approval.
export interface ToolApproval {
  id: string
  name: string
  server: string
  args: Record<string, unknown>
}

export interface ToolCall {
  name: string
  args: Record<string, unknown>
  result: string
}

export interface WebsocketResponse {
  message: string
  done: boolean
  error?: string
  // Set on the final message of a generation the user cancelled — `message`
  // is empty in that case (the partial text was already sent as earlier
  // `done: false` chunks), rather than a fresh full response.
  stopped?: boolean
  // Only present on the final `done: true` frame, and only when Ollama
  // actually reported it (never on partial/cancelled generations).
  stats?: GenerationStats
  // On every final frame of a reply that got as far as generating.
  context?: ContextReport
  // Only present when the chat has active document collections and at
  // least one chunk was relevant enough to include — see rag/retrieval.py.
  citations?: Citation[]
  // A standalone `done: false` frame — one per tool call the model made
  // before its final answer. Never combined with `message` in the same
  // frame (see CompareConsumer/ChatConsumer — tool_call frames carry no
  // text of their own).
  tool_call?: ToolCall
  // A standalone `done: false` frame sent before the answer starts — a
  // plain-language reason some attached documents weren't searched.
  notice?: string
  // A standalone `done: false` frame: the next piece of a thinking model's
  // reasoning, streamed apart from the answer (`message`).
  thinking?: string
  // A standalone `done: false` frame: generation is paused until the user
  // answers with respondToolApproval.
  tool_approval?: ToolApproval
  // Sent once, after the first reply in an untitled chat: its generated title.
  chat_title?: string
}

export interface WebsocketHandlers {
  onConnect: () => void
  onDisconnect: () => void
  onSendMessage: (message: WebsocketMessage) => void
  onReceiveMessage: (data: WebsocketResponse) => void
  onError?: (message: string) => void
  onNotice?: (notice: string) => void
  onChatTitle?: (title: string) => void
  onReconnecting?: (attempt: number) => void
  onMaxReconnectFailed?: () => void
}

export interface RegeneratePayload {
  ai_model: string
  ai_model_parameters: string
  structured_output?: WebsocketMessage['structured_output']
}

export interface EditResendPayload {
  index: number
  message: string
  ai_model: string
  ai_model_parameters: string
  image?: string
  structured_output?: WebsocketMessage['structured_output']
}

export interface WebSocketWrapper {
  readonly readyState: number
  sendMessage: (message: WebsocketMessage) => void
  stopGeneration: () => void
  // Re-runs the last assistant reply in place — no new user message. The
  // consumer replaces the existing assistant row rather than appending one.
  regenerate: (payload: RegeneratePayload) => void
  // Edits the user message at `index` and discards everything from that
  // point on, replacing it with a freshly generated exchange.
  editResend: (payload: EditResendPayload) => void
  respondToolApproval: (id: string, approved: boolean) => void
  closeConnection: () => void
}

const RECONNECT_BASE_DELAY_MS = 1000
const RECONNECT_MAX_DELAY_MS = 30_000
const RECONNECT_MAX_ATTEMPTS = 8

export function getWebsocket(handlers: WebsocketHandlers, roomId: string, baseURL: string): WebSocketWrapper {
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
    const ws = new WebSocket(`${scheme}://${socketHost}/ws/chat/${roomId}/`)
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
        const data: WebsocketResponse = JSON.parse(event.data)

        if (data.error) {
          handlers.onError?.(data.error)

          return
        }

        if (data.notice) {
          handlers.onNotice?.(data.notice)

          return
        }

        if (data.chat_title) {
          handlers.onChatTitle?.(data.chat_title)

          return
        }

        handlers.onReceiveMessage(data)
      }
      catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    })

    return ws
  }

  currentSocket = buildSocket()

  return {
    // A getter delegating to whichever socket is currently live — reconnects
    // used to replace the underlying WebSocket but callers kept reading
    // `readyState` off the very first (now-dead) instance via `Object.assign`,
    // which only copies own enumerable properties and misses prototype
    // accessors like `readyState`. Every send after any reconnect silently
    // failed as a result.
    get readyState() {
      return currentSocket.readyState
    },
    sendMessage: (message: WebsocketMessage) => {
      if (currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'prompt', ...message }))
        handlers.onSendMessage(message)
      }
      else {
        console.warn('WebSocket is not open. Message not sent:', message)
      }
    },
    stopGeneration: () => {
      if (currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'stop' }))
      }
      else {
        console.warn('WebSocket is not open. Stop not sent.')
      }
    },
    regenerate: (payload: RegeneratePayload) => {
      if (currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'regenerate', ...payload }))
      }
      else {
        console.warn('WebSocket is not open. Regenerate not sent.')
      }
    },
    editResend: (payload: EditResendPayload) => {
      if (currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'edit_resend', ...payload }))
      }
      else {
        console.warn('WebSocket is not open. Edit not sent.')
      }
    },
    respondToolApproval: (id: string, approved: boolean) => {
      if (currentSocket.readyState === WebSocket.OPEN)
        currentSocket.send(JSON.stringify({ type: 'tool_approval', id, approved }))
      else
        console.warn('WebSocket is not open. Tool approval not sent.')
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
