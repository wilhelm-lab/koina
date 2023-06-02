<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>

<script setup lang="ts">
useHead({
  title: "Koina",
  meta: [
    {
      name: "description",
      content:
        "Community driven endeavour to make AI models for life science research accessible",
    },
  ],
});
</script>
