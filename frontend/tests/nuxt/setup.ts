import { vi } from 'vitest'

// happy-dom doesn't implement visualViewport, which Vuetify's overlays
// (dialogs, menus, tooltips) subscribe to for positioning.
vi.stubGlobal('visualViewport', {
  width: 1280,
  height: 800,
  offsetLeft: 0,
  offsetTop: 0,
  pageLeft: 0,
  pageTop: 0,
  scale: 1,
  addEventListener: () => {},
  removeEventListener: () => {},
})
