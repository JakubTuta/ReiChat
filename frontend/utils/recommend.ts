import type { IAIModel, IAIModelVersion } from '~/models/aiModel'
import type { IHardwareInfo } from '~/models/hardware'
import { computeFitLabel } from '~/models/hardware'

// Good all-round chat models, best first. The first one that's in the
// catalog and "runs well" on this machine is the recommendation, so a
// bigger computer gets a smarter model and a small one still gets
// something that answers quickly.
export const RECOMMENDATION_LADDER: { model: string, parameters: string, blurb: string }[] = [
  { model: 'qwen3', parameters: '14b', blurb: 'Strong at reasoning, writing and code.' },
  { model: 'gemma3', parameters: '12b', blurb: 'Strong all-rounder that can also look at images.' },
  { model: 'qwen3', parameters: '8b', blurb: 'Smart and quick for everyday questions, writing and code.' },
  { model: 'gemma3', parameters: '4b', blurb: 'A capable everyday model that can also look at images.' },
  { model: 'qwen3', parameters: '4b', blurb: 'Fast, and good at step-by-step reasoning.' },
  { model: 'llama3.2', parameters: '3b', blurb: 'Fast and friendly for everyday questions.' },
  { model: 'gemma3', parameters: '1b', blurb: 'Very light — answers quickly even on modest computers.' },
  { model: 'llama3.2', parameters: '1b', blurb: 'Very light — answers quickly even on modest computers.' },
  { model: 'qwen3', parameters: '0.6b', blurb: 'The smallest option; fine for trying things out.' },
]

export interface Recommendation {
  model: IAIModel
  version: IAIModelVersion
  blurb: string
}

function candidates(catalog: IAIModel[]): Recommendation[] {
  return RECOMMENDATION_LADDER.flatMap((step) => {
    const model = catalog.find(m => m.model === step.model)
    const version = model?.versions.find(v => v.parameters === step.parameters)

    return model && version
      ? [{ model, version, blurb: step.blurb }]
      : []
  })
}

// The best model that runs well here, plus a lighter alternative for when
// speed matters more. Hardware unknown → the lightest options, which work
// everywhere.
export function recommendModels(catalog: IAIModel[], hardware: IHardwareInfo): { best: Recommendation | null, lighter: Recommendation | null } {
  const all = candidates(catalog)
  const fitting = all.filter(c => computeFitLabel(c.version.size_bytes, hardware).label === 'runs_well')
  const pool = fitting.length
    ? fitting
    : all.slice(-2)

  return { best: pool[0] ?? null, lighter: pool[2] ?? pool[1] ?? null }
}
