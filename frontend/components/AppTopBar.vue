<script setup lang="ts">
const route = useRoute()

const navItems = [
  { label: 'Home', to: '/' },
  { label: 'Chat', to: '/chat' },
  { label: 'Models', to: '/models' },
  { label: 'Compare', to: '/compare' },
  { label: 'Integrations', to: '/integrations' },
  { label: 'Your data', to: '/data' },
]

function isActive(to: string) {
  return to === '/'
    ? route.path === '/'
    : route.path.startsWith(to)
}
</script>

<template>
  <v-app-bar
    flat
    elevation="0"
    height="56"
    color="background"
  >
    <NuxtLink
      to="/"
      class="app-logo-link"
    >
      <img
        src="/icon-64.png"
        alt="ReiChat"
        class="app-logo-mark"
        width="30"
        height="30"
      >

      <span class="app-wordmark">ReiChat</span>
    </NuxtLink>

    <nav
      class="app-nav"
      aria-label="Primary"
    >
      <NuxtLink
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        class="app-nav-pill"
        :class="{'app-nav-pill--active': isActive(item.to)}"
      >
        {{ item.label }}
      </NuxtLink>
    </nav>

    <template
      v-if="$slots.append"
      #append
    >
      <slot name="append" />
    </template>
  </v-app-bar>
</template>

<style scoped>
.app-logo-link {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  flex-shrink: 0;
  margin-left: 4px;
}

.app-logo-mark {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  object-fit: cover;
  display: block;
}

.app-wordmark {
  font-weight: 600;
  font-size: 18px;
  letter-spacing: -0.02em;
  color: var(--color-ink);
}

.app-nav {
  display: flex;
  align-items: center;
  gap: 2px;
  margin-left: 14px;
}

.app-nav-pill {
  padding: 6px 11px;
  border-radius: 8px;
  font-size: 12.5px;
  font-weight: 500;
  color: var(--color-ink-2);
  text-decoration: none;
  transition: background-color 0.15s ease, color 0.15s ease;
}

.app-nav-pill:hover {
  background: oklch(0.94 0.014 168);
}

.app-nav-pill--active {
  background: var(--color-mint-tint);
  color: var(--color-ink);
}

@media (max-width: 700px) {
  .app-nav {
    gap: 0;
    margin-left: 6px;
  }

  .app-nav-pill {
    padding: 6px 8px;
  }
}
</style>
