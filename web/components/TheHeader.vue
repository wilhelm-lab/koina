<template>
  <header
    class="py-4 px-2 md:px-4 lg:px-8 flex items-center justify-between font-semibold border-b-2 border-gray-200"
  >
    <nuxt-link
      to="/"
      class="flex justify-center items-center text-3xl font-bold"
    >
      <img src="~/assets/img/koina-logo.svg" class="h-12" />
      <span class="ml-2"> Koina </span>
    </nuxt-link>
    <nav>
      <ul class="flex items-center">
        <li class="ml-4">
          <nuxt-link to="/docs">Documentation</nuxt-link>
        </li>
        <li class="ml-4">
          <a href="mailto:Ludwig.Lautenbacher@tum.de"> Contact </a>
        </li>
        <li class="ml-4">
          <nuxt-link to="https://github.com/wilhelm-lab/koina">
            GitHub
          </nuxt-link>
        </li>
      </ul>
    </nav>
  </header>
</template>

<style scoped>
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  a {
    @apply text-black;
  }
}
</style>
