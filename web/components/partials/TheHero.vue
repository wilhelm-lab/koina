<script setup lang="ts">
import BaseButton from "~/components/ui/BaseButton.vue";
</script>

<template>
  <div
    class="py-16 flex items-center justify-center w-full bg-primary-dark-3 bg-cover bg-center bg-hero-pattern"
  >
    <div
      class="max-w-5xl flex flex-col items-center justify-center text-center"
    >
      <h1 class="text-9xl font-bold mb-4 text-white">Koina</h1>
      <p class="text-3xl text-white">
        Democratizing machine learning for proteomics research
      </p>
      <div class="flex justify-center items-center mt-8">
        <BaseButton to="/docs" color="secondary" text-color="black">
          Get Started
        </BaseButton>
        <BaseButton to="#get-involved" color="secondary" outlined class="ml-8">
          Get involved
        </BaseButton>
      </div>
    </div>
  </div>
</template>
