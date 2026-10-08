// A model's thinking controls as /api/show reports them (see
// django_app/ollama_api.thinking_controls).
export interface ThinkingControls {
  values: (boolean | string)[]
  default: boolean | string
}

export interface ThinkingOption {
  // What gets saved as ChatHistory.think_level.
  value: string
  label: string
}

const LEVEL_LABELS: Record<string, string> = {
  low: 'Light',
  medium: 'Normal',
  high: 'Deep',
}

export function levelLabel(level: string): string {
  return LEVEL_LABELS[level] ?? level.charAt(0).toUpperCase() + level.slice(1)
}

// "Auto" (the model's own default) first, then plain on/off or the model's
// named levels in the order it lists them.
export function thinkingOptions(controls: ThinkingControls): ThinkingOption[] {
  const options: ThinkingOption[] = [{ value: '', label: 'Auto' }]
  const levels = controls.values.filter((v): v is string => typeof v === 'string')

  if (levels.length) {
    options.push(...levels.map(level => ({ value: level, label: levelLabel(level) })))
  }
  else {
    if (controls.values.includes(true))
      options.push({ value: 'on', label: 'On' })
    if (controls.values.includes(false))
      options.push({ value: 'off', label: 'Off' })
  }

  return options
}

export function thinkLevelLabel(thinkLevel: string): string | null {
  if (!thinkLevel)
    return null
  if (thinkLevel === 'on')
    return 'Thinking on'
  if (thinkLevel === 'off')
    return 'Thinking off'

  return `Thinking: ${levelLabel(thinkLevel)}`
}
