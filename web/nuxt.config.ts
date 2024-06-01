import wasm from "vite-plugin-wasm-esm";
import topLevelAwait from "vite-plugin-top-level-await";

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
  modules: ["@nuxtjs/tailwindcss", "@nuxt/eslint", "@vueuse/nuxt"],
  eslint: {
    config: {
      stylistic: true,
    },
  },
  vite: {
    plugins: [wasm(["biowclib-mz"]), topLevelAwait()],
    build: {
      target: "esnext",
    },
  },
  vue: {
    compilerOptions: {
      isCustomElement: (tag) => {
        return tag.startsWith("biowc-") || tag == "rapi-doc";
      },
    },
  },
});
