import { describe, expect, it } from 'vitest'
import { containerNameFor } from '~/utils/containerName'

describe('containerNameFor', () => {
  it('matches the backend for plain and namespaced models', () => {
    expect(containerNameFor('llama3.2', '3b')).toBe('llama3.2_3b')
    expect(containerNameFor('hf.co/user/Repo-GGUF', 'Q4_K_M')).toBe('hf.co-user-Repo-GGUF_Q4_K_M')
  })
})
