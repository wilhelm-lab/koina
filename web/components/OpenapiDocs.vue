<template>
  <rapi-doc
    spec-url="/openapi.yml"
    show-header="false"
    render-style="read"
    theme="light"
    primary-color="#001bc8"
    allow-authentication="false"
    allow-server-selection="false"
    nav-bg-color="#f0f0f0"
    info-description-headings-in-navbar="true"
    font-size="largest"
  >
    <div slot="footer">
      <slot name="footer" />
    </div>
  </rapi-doc>
</template>

<script setup lang="ts">
import "rapidoc";
</script>
