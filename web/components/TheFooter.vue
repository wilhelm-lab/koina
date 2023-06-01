<template>
  <footer
    class="py-8 px-2 md:px-4 lg:px-8 flex items-center justify-center border-t-2 border-gray-200"
  >
    <div>
      <nuxt-link to="https://www1.ls.tum.de/en/compms/impressum/" class="ml-8">
        Imprint
      </nuxt-link>
    </div>
  </footer>
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
