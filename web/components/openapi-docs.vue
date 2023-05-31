<template>
  <div class="rapidoc-container">
    <rapi-doc
      spec-url="/swagger.yml"
      render-style="read"
      show-header="false"
      theme="light"
      primary-color="#001bc8"
      allow-authentication="false"
      allow-server-selection="false"
      nav-bg-color="#f0f0f0"
      font-size="largest"
    />
  </div>
</template>

<script setup lang="ts">
import "rapidoc";
</script>

<style>
.rapidoc-container {
  height: calc(100vh - 70px) !important;
}
</style>
