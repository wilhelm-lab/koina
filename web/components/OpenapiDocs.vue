<template>
  <rapi-doc
    ref="rapidoc"
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
    schema-style="table"
  >
    <template #footer>
      <div>
        <slot name="footer" />
      </div>
    </template>
  </rapi-doc>

  <Teleport v-if="spectraResults.length" :to="teleportTarget">
    <SpectraResults :spectras="spectraResults" />
  </Teleport>
</template>

<script setup lang="ts">
/* eslint-disable @typescript-eslint/no-explicit-any */
import "rapidoc";

import type { KoinaSpectrum } from "@/utils/spectrum";
import { koinaResponseToSpectra } from "@/utils/spectrum";

const rapidoc: any = ref(null);

const spectraResults = ref<KoinaSpectrum[]>([]);
const teleportTarget = ref<HTMLElement | null>(null);

onMounted(() => {
  document.body.classList.add("overflow-hidden");

  if (rapidoc.value && import.meta.client) {
    // This is a bit hacky, but there seems to be no other way to get the request body in after-try event.
    let lock = false;
    let requestBody: any = undefined;

    rapidoc.value.addEventListener("before-try", (e: CustomEvent) => {
      if (lock) {
        throw new Error(
          "Please wait for the previous 'try'-request to finish.",
        );
      }

      lock = true;
      spectraResults.value = [];

      try {
        requestBody = JSON.parse(e.detail.request.body);
      } finally {
        lock = false;
      }
    });

    rapidoc.value.addEventListener("after-try", (e: CustomEvent) => {
      try {
        const responseBody = e.detail.responseBody;

        spectraResults.value = koinaResponseToSpectra(
          requestBody.inputs,
          responseBody.outputs,
        );

        const modelPath = e.detail.request.url.match(
          /.*?(\/[^/]*?\/infer)$/,
        )![1];
        const apiRequestEl = rapidoc.value.shadowRoot.querySelector(
          `api-request[path="${modelPath}"]`,
        );

        const teleportTargetEl = document.createElement("div");
        apiRequestEl.insertAdjacentElement("afterend", teleportTargetEl);
        teleportTarget.value = teleportTargetEl;
      } finally {
        lock = false;
      }
    });
  }
});

onBeforeUnmount(() => {
  document.body.classList.remove("overflow-hidden");
});
</script>
