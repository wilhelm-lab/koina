<script setup lang="ts">
import TheHeader from "~/components/partials/TheHeader.vue";
import TheFooter from "~/components/partials/TheFooter.vue";

definePageMeta({
  layout: false,
});
</script>

<template>
  <div class="overflow-hidden">
    <TheHeader />
    <ClientOnly>
      <div class="openapidoc-container">
        <OpenapiDocs>
          <template #header>
            <TheHeader />
          </template>
          <template #footer>
            <TheFooter />
          </template>
        </OpenapiDocs>
      </div>
    </ClientOnly>
  </div>
</template>

<style>
.openapidoc-container {
  height: calc(100vh - 82px);
}
</style>
