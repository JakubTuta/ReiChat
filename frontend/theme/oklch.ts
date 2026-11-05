// Vuetify parses theme colours itself (it derives hover and state-layer
// overlays from the RGB channels) and only understands hex/rgb/hsl, so the
// oklch tokens in assets/css/main.css have to be converted for it.
// Matrices: Björn Ottosson's OKLab, https://bottosson.github.io/posts/oklab/

const OKLCH_PATTERN = /^oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*[\d.]+)?\s*\)$/

function toGamma(linear: number): number {
  const clamped = Math.min(1, Math.max(0, linear))

  return clamped <= 0.0031308
    ? 12.92 * clamped
    : 1.055 * clamped ** (1 / 2.4) - 0.055
}

function toByte(linear: number): string {
  return Math.round(255 * toGamma(linear)).toString(16).padStart(2, '0')
}

export function oklchToHex(value: string): string {
  const match = OKLCH_PATTERN.exec(value.trim())
  if (!match)
    throw new Error(`Not an oklch() colour: "${value}"`)

  const lightness = Number(match[1])
  const chroma = Number(match[2])
  const hue = Number(match[3]) * Math.PI / 180

  const a = chroma * Math.cos(hue)
  const b = chroma * Math.sin(hue)

  const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3

  const red = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
  const green = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
  const blue = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

  return `#${toByte(red)}${toByte(green)}${toByte(blue)}`
}
