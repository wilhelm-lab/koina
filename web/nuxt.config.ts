import wasm from 'vite-plugin-wasm-esm'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  nitro: {
    prerender: {
      crawlLinks: true,
    },
    experimental: {
      wasm: true,
    },
  },
  modules: ['@nuxtjs/tailwindcss', '@nuxt/eslint', '@vueuse/nuxt'],
  eslint: {
    config: {
      stylistic: true,
    },
  },
  vite: {
    plugins: [wasm(['biowclib-mz'])],
  },
})
