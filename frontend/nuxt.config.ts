/* eslint-disable node/prefer-global/process */
import vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'

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

  typescript: {
    strict: true,
  },

  compatibilityDate: '2024-07-18',
})
