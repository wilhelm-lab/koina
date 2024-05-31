<script setup lang="ts">
import TheHeader from "~/components/partials/TheHeader.vue";
import TheFooter from "~/components/partials/TheFooter.vue";
</script>

<template>
  <div class="overflow-hidden">
    <TheHeader />
    <slot />
    <TheFooter />
  </div>
</template>
