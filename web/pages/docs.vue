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

<script setup lang="ts">
definePageMeta({
  layout: false,
})
</script>

<style>
.openapidoc-container {
  height: calc(100vh - 70px);
}
</style>
