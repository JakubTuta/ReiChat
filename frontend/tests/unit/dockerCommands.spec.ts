import type { ManualCommandInput } from '~/utils/dockerCommands'
import { describe, expect, it } from 'vitest'
import { isShellType, lineContinuation, manualCommands } from '~/utils/dockerCommands'

function input(overrides: Partial<ManualCommandInput> = {}): ManualCommandInput {
  return {
    model: 'llama3.1',
    parameters: '8b',
    status: 'running',
    port: 11436,
    gpuAvailable: true,
    shell: 'bash',
    ...overrides,
  }
}

describe('lineContinuation', () => {
  it('uses each shell\'s own continuation character', () => {
    expect(lineContinuation('bash')).toBe('\\')
    expect(lineContinuation('cmd')).toBe('^')
    expect(lineContinuation('powershell')).toBe('`')
  })
})

describe('isShellType', () => {
  it('accepts only the shells the UI offers', () => {
    expect(isShellType('powershell')).toBe(true)
    expect(isShellType('fish')).toBe(false)
    expect(isShellType(undefined)).toBe(false)
    expect(isShellType(['bash'])).toBe(false)
  })
})

describe('manualCommands', () => {
  it('creates and pulls when no container exists yet', () => {
    const [create, pull] = manualCommands(input({ status: 'not_found' }))

    expect(create).toContain('docker run -d \\')
    expect(create).toContain('--name llama3.1_8b \\')
    expect(create).toContain('-p 127.0.0.1:11436:11434 \\')
    expect(create).toContain('--label reichat.port=11436 \\')
    expect(create).toContain('--gpus=all \\')
    expect(pull).toBe('docker exec -d llama3.1_8b ollama pull llama3.1:8b')
  })

  it('leaves out the GPU flag only when the host is known to have no GPU', () => {
    expect(manualCommands(input({ status: 'not_found', gpuAvailable: false }))[0]).not.toContain('--gpus')
    expect(manualCommands(input({ status: 'not_found', gpuAvailable: null }))[0]).toContain('--gpus=all')
  })

  it('continues each line with the chosen shell\'s character', () => {
    const [create] = manualCommands(input({ status: 'not_found', shell: 'powershell' }))

    expect(create).toContain('docker run -d `')
    expect(create).not.toContain('\\')
  })

  it('offers to start a stopped container, then remove it', () => {
    expect(manualCommands(input({ status: 'exited' }))).toEqual([
      'docker start llama3.1_8b',
      'docker rm llama3.1_8b',
      'docker volume rm reichat-ollama-llama3.1_8b',
    ])
  })

  it.each(['running', 'pulling_model', 'starting', 'incomplete'])(
    'stops a %s container exactly once before removing it',
    (status) => {
      const commands = manualCommands(input({ status }))

      expect(commands).toEqual([
        'docker stop llama3.1_8b',
        'docker rm llama3.1_8b',
        'docker volume rm reichat-ollama-llama3.1_8b',
      ])
    },
  )

  it('stops a restarting container but does not try to start it', () => {
    expect(manualCommands(input({ status: 'restarting' }))).toEqual([
      'docker stop llama3.1_8b',
      'docker rm llama3.1_8b',
      'docker volume rm reichat-ollama-llama3.1_8b',
    ])
  })

  it('names the container the same way the backend does for odd model names', () => {
    const commands = manualCommands(input({ model: 'hf.co/acme/model', parameters: 'Q4_K_M', status: 'exited' }))

    expect(commands[0]).toBe('docker start hf.co-acme-model_Q4_K_M')
  })
})
