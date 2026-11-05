import { createVuetify } from 'vuetify'
import { aliases, mdi } from 'vuetify/iconsets/mdi-svg'
import mainCss from '~/assets/css/main.css?raw'
import { appIcons } from '~/theme/icons'
import { buildReichatTheme } from '~/theme/reichat'

import 'vuetify/styles'

export default defineNuxtPlugin((app) => {
  const vuetify = createVuetify({
    icons: {
      defaultSet: 'mdi',
      aliases: { ...aliases, ...appIcons },
      sets: { mdi },
    },
    theme: {
      defaultTheme: 'reichat',
      themes: {
        reichat: buildReichatTheme(mainCss),
      },
    },
    defaults: {
      VTextField: {
        variant: 'outlined',
        color: 'mint-deep',
      },
      VAutocomplete: {
        variant: 'outlined',
        color: 'mint-deep',
      },
      VSelect: {
        variant: 'outlined',
        color: 'mint-deep',
      },
      VTextarea: {
        variant: 'outlined',
        color: 'mint-deep',
      },
      VSwitch: {
        color: 'mint-btn',
      },
      VCheckbox: {
        color: 'mint-btn',
      },
      VBtn: {
        variant: 'outlined',
        rounded: 'lg',
        class: 'text-none',
      },
      VContainer: {
        style: 'max-width: 1200px',
      },
      VCard: {
        rounded: 'lg',
        width: '100%',
      },
      VTab: {
        rounded: 'lg',
      },
      VListItem: {
        rounded: 'lg',
      },
    },
    display: {
      mobileBreakpoint: 'sm',
    },
  })
  app.vueApp.use(vuetify)
})
