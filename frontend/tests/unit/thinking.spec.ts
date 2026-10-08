import { describe, expect, it } from 'vitest'
import { thinkingOptions, thinkLevelLabel } from '~/utils/thinking'

describe('thinking options', () => {
  it('offers Auto plus on/off for boolean-controlled models', () => {
    expect(thinkingOptions({ values: [false, true], default: true }).map(o => o.label)).toEqual(['Auto', 'On', 'Off'])
  })

  it('offers plain-language level names for level-controlled models', () => {
    expect(thinkingOptions({ values: ['low', 'medium', 'high'], default: 'medium' })).toEqual([
      { value: '', label: 'Auto' },
      { value: 'low', label: 'Light' },
      { value: 'medium', label: 'Normal' },
      { value: 'high', label: 'Deep' },
    ])
  })

  it('labels a chat setting for the chip, nothing for Auto', () => {
    expect(thinkLevelLabel('')).toBeNull()
    expect(thinkLevelLabel('off')).toBe('Thinking off')
    expect(thinkLevelLabel('high')).toBe('Thinking: Deep')
  })
})
