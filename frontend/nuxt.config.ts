/* eslint-disable node/prefer-global/process */
import vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'

const serverUrl = new URL(process.env.SERVER_URL || 'http://localhost:8000/')
const serverSocketOrigin = `${serverUrl.protocol === 'https:'
  ? 'wss'
  : 'ws'}://${serverUrl.host}`

// What the page may load or talk to. Model output is untrusted and is shown as
// HTML, so beyond sanitising it the page itself must be unable to send data
// anywhere: `img-src` is the one that matters most, because an image URL is
// how a prompt-injected reply leaks a chat without a single click. Inline
// script and style stay allowed because Nuxt's bootstrap script and Vuetify's
// style bindings are inline.
const CONTENT_SECURITY_POLICY = [
  'default-src \'self\'',
  'script-src \'self\' \'unsafe-inline\'',
  'style-src \'self\' \'unsafe-inline\'',
  'img-src \'self\' data:',
  'font-src \'self\' data:',
  `connect-src 'self' ${serverUrl.origin} ${serverSocketOrigin}`,
  'object-src \'none\'',
  'frame-src \'none\'',
  'frame-ancestors \'none\'',
  'base-uri \'self\'',
  'form-action \'self\'',
].join('; ')

export default defineNuxtConfig({
  app: {
    head: {
      title: 'ReiChat',
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/icon-16.png' },
        { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/icon-32.png' },
        { rel: 'icon', type: 'image/png', sizes: '48x48', href: '/icon-48.png' },
        { rel: 'apple-touch-icon', sizes: '256x256', href: '/icon-256.png' },
      ],
    },
  },

  css: [
    '@fontsource/instrument-sans/400.css',
    '@fontsource/instrument-sans/500.css',
    '@fontsource/instrument-sans/600.css',
    '@fontsource/ibm-plex-mono/400.css',
    '@fontsource/ibm-plex-mono/500.css',
    '~/assets/css/main.css',
  ],

  build: {
    transpile: ['vuetify'],
  },

  modules: [
    '@vueuse/nuxt',
    '@pinia/nuxt',
    '@radya/nuxt-dompurify',
    (_options, nuxt) => {
      nuxt.hooks.hook('vite:extendConfig', (config) => {
        // eslint-disable-next-line ts/ban-ts-comment
        // @ts-expect-error
        config.plugins.push(vuetify({ autoImport: true }))
      })
    },
  ],

  imports: {
    autoImport: true,
    dirs: [
      'stores/**',
      'utils/**',
      'components/**',
      'models/**',
    ],
  },

  vite: {
    vue: {
      template: {
        transformAssetUrls,
      },
    },
  },

  runtimeConfig: {
    public: {
      // Runs in the browser, so it reaches the backend through the port
      // docker-compose publishes on the host. `host.docker.internal` only
      // resolves inside a container and would never work here.
      serverUrl: process.env.SERVER_URL || 'http://localhost:8000/',
    },
  },

  ssr: false,

  nitro: {
    preset: 'node-server',
  },

  // Production only: the dev server's hot reload and Nuxt DevTools need
  // sockets and frames this policy forbids.
  $production: {
    nitro: {
      routeRules: {
        '/**': { headers: { 'Content-Security-Policy': CONTENT_SECURITY_POLICY } },
      },
    },
  },

  dompurify: {
    profiles: {
      // Used by MessageParts as `v-sanitize-html:reply`. DOMPurify's defaults
      // already drop scripts, event handlers and javascript: links, but still
      // let a reply inject <style> (restyle or hide the whole app), `style`
      // attributes (a fake full-screen overlay) and form controls (a
      // credentials form). A reply is prose, so none of that is needed.
      reply: {
        FORBID_TAGS: ['style', 'form', 'button', 'textarea', 'select', 'option', 'optgroup', 'fieldset', 'dialog'],
        FORBID_ATTR: ['style', 'autofocus', 'formaction', 'form', 'name'],
      },
    },
  },

  typescript: {
    strict: true,
  },

  compatibilityDate: '2024-07-18',
})
