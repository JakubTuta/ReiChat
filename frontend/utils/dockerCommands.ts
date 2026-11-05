import { containerNameFor } from '~/utils/containerName'

// cmd.exe uses `^` for line continuation, but PowerShell — the default
// shell on Windows 11 — uses a backtick, and `^` just fails there. A plain
// "Windows commands" checkbox defaulting to true assumed cmd unconditionally
// and broke for anyone on the actual Win11 default.
export type ShellType = 'powershell' | 'cmd' | 'bash'

export const SHELL_OPTIONS: { title: string, value: ShellType }[] = [
  { title: 'bash/zsh', value: 'bash' },
  { title: 'PowerShell', value: 'powershell' },
  { title: 'Command Prompt', value: 'cmd' },
]

export function isShellType(value: unknown): value is ShellType {
  return value === 'powershell' || value === 'cmd' || value === 'bash'
}

export function lineContinuation(shell: ShellType): string {
  if (shell === 'cmd')
    return '^'
  if (shell === 'powershell')
    return '`'

  return '\\'
}

export interface ManualCommandInput {
  model: string
  parameters: string
  // 'not_found' when no container exists for this model and size yet.
  status: string
  port: number
  gpuAvailable: boolean | null | undefined
  shell: ShellType
}

function createCommands({ model, parameters, port, gpuAvailable, shell }: ManualCommandInput): string[] {
  const escape = lineContinuation(shell)
  const containerName = containerNameFor(model, parameters)

  const gpuFlag = gpuAvailable === false
    ? ''
    : `  --gpus=all ${escape}\n`
  const createCommand = `docker run -d ${escape}
  --name ${containerName} ${escape}
  --network chatbot-network ${escape}
${gpuFlag}  -p 127.0.0.1:${port}:11434 ${escape}
  -v reichat-ollama-${containerName}:/root/.ollama ${escape}
  --label reichat.managed=true ${escape}
  --label reichat.model=${model} ${escape}
  --label reichat.parameters=${parameters} ${escape}
  --label reichat.port=${port} ${escape}
  ollama/ollama:latest`

  const pullModelCommand = `docker exec -d ${containerName} ollama pull ${model}:${parameters}`

  return [createCommand, pullModelCommand]
}

function startCommands({ model, parameters }: ManualCommandInput): string[] {
  return [`docker start ${containerNameFor(model, parameters)}`]
}

function stopCommands({ model, parameters }: ManualCommandInput): string[] {
  return [`docker stop ${containerNameFor(model, parameters)}`]
}

function removeCommands({ model, parameters }: ManualCommandInput): string[] {
  const containerName = containerNameFor(model, parameters)

  return [`docker rm ${containerName}`, `docker volume rm reichat-ollama-${containerName}`]
}

// The docker commands that do by hand what the buttons on a model card do,
// for the container's current state.
export function manualCommands(input: ManualCommandInput): string[] {
  if (input.status === 'not_found')
    return createCommands(input)

  const cmds: string[] = []
  if (['exited', 'paused'].includes(input.status))
    cmds.push(...startCommands(input))
  if (['running', 'paused', 'restarting', 'pulling_model', 'starting', 'incomplete'].includes(input.status))
    cmds.push(...stopCommands(input))
  cmds.push(...removeCommands(input))

  return cmds
}
