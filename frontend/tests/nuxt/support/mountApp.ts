import type { Component } from 'vue'
import { mountSuspended } from '@nuxt/test-utils/runtime'

// Mounts inside the real Nuxt app (so its plugins, Pinia and auto-imports are
// live) and attaches to <body>, where Vuetify's dialogs and menus render.
export function mountApp(component: Component, options: { props?: Record<string, unknown> } = {}) {
  return mountSuspended(component, { attachTo: document.body, props: options.props })
}
