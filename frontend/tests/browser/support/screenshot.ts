import type { Locator, Page } from '@playwright/test'
import { expect } from '@playwright/test'

// Every weight the app ships, loaded up front. A face is only fetched when
// text first uses it, so a dialog title in a weight nothing else on the page
// uses would otherwise be screenshotted in the fallback font whenever the
// machine is busy.
const FONT_FACES = [
  '400 14px "Instrument Sans"',
  '500 14px "Instrument Sans"',
  '600 14px "Instrument Sans"',
  '400 12px "IBM Plex Mono"',
  '500 12px "IBM Plex Mono"',
]

// Waits for fonts and for every finite animation and Vue transition to end.
// A dialog slides in over ~300ms and lands on a fractional pixel; text snaps
// differently while it still carries a transform, so a screenshot taken near
// the end of the slide differs by a pixel from one taken after it.
export async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle')
  await page.evaluate(
    async (faces) => {
      await Promise.all(faces.map(face => document.fonts.load(face)))
      await document.fonts.ready
    },
    FONT_FACES,
  )
  await page.waitForFunction(() => {
    // SVG targets are the progress spinners: they animate forever, and a
    // global `scrollbar-color` transition on their circle never finishes.
    const running = document.getAnimations().filter((a) => {
      const target = (a.effect as KeyframeEffect | null)?.target
      const finite = a.effect?.getComputedTiming().iterations !== Infinity

      return a.playState === 'running' && finite && !(target instanceof SVGElement)
    })

    return running.length === 0
      && !document.querySelector('[class*="-enter-active"], [class*="-leave-active"]')
  })
}

// The only way specs take a screenshot, so none can forget to settle first.
export async function expectPageScreenshot(page: Page, name: string, options: { fullPage?: boolean } = {}): Promise<void> {
  await settle(page)
  await expect(page).toHaveScreenshot(name, options)
}

export async function expectLocatorScreenshot(locator: Locator, name: string): Promise<void> {
  await settle(locator.page())
  await expect(locator).toHaveScreenshot(name)
}
