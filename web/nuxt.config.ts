import wasm from "vite-plugin-wasm-esm";


// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  nitro: {
    prerender: {
      crawlLinks: true
    },
    experimental: {
      wasm: true
    }
  },
  modules: [
    '@nuxtjs/eslint-module',
    '@nuxtjs/tailwindcss'
  ],
  eslint: {
    lintOnStart: false
  },
  vite: {
    plugins: [wasm(["biowclib-mz"])],
  },
})
