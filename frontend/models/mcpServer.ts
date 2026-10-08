export type MCPTransport = 'stdio' | 'http'

export interface IMCPServer {
  id: number
  name: string
  transport: MCPTransport
  command: string
  url: string
  enabled: boolean
  created_at: string
}

export type MCPTestResult
  = | { ok: true, tools: { name: string, description: string }[] }
    | { ok: false, error: string }
