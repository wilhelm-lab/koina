<script setup lang="ts">
import TheHeader from "~/components/partials/TheHeader.vue";
import TheFooter from "~/components/partials/TheFooter.vue";
import { VTour } from "#components";
import type { TourStep } from "#nuxt-tour/props";

definePageMeta({
  layout: false,
});

// Store the tour component in a ref
const tour = ref<InstanceType<typeof VTour> | null>(null);
// define the steps for the tour
const htmlTarget1 = ref<HTMLElement | null>(null);
const htmlTarget2 = ref<HTMLElement | null>(null);
const htmlTarget3 = ref<HTMLElement | null>(null);

let rapiDoc: any = null;

const steps = computed<TourStep[]>(() => {
  return [
    {
      target: htmlTarget1.value,
      title: "HTML Elements",
      body: "You can also provide the HTML element as a target",
    },
    {
      target: htmlTarget2.value,
      title: "HTML Elements",
      body: "This allows for greater flexibility in targeting elements",
    },
    {
      target: htmlTarget3.value,
      title: "HTML Elements",
      body: "To make sure the DOM is fully rendered assign the element only in the onMounted hook",
    },
  ];
});

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const { $listen, $off } = useNuxtApp()

function resetTour() {
  tour.value?.resetTour();
}

async function updateTargets() {
  await nextTick();
  rapiDoc = document.querySelector('rapi-doc');
  if (rapiDoc && rapiDoc.shadowRoot) {
    await nextTick();
    const navElement = rapiDoc.shadowRoot.querySelector('#the-main-body > nav');
    if (navElement) {
      htmlTarget1.value = navElement;
      htmlTarget2.value = navElement;
      htmlTarget3.value = navElement;    
    }
    else {
      await sleep(2000); // Pauses for 2 seconds
      updateTargets();
    }
  }
  else {
    await sleep(2000); // Pauses for 2 seconds
    updateTargets();
  }
  tour.value?.startTour();
}

onMounted( () => {
  $listen('startTour', resetTour)
  $listen('rapi-doc-mounted', updateTargets)
});

onUnmounted(() => {
  $off('startTour', resetTour)
  $off('rapi-doc-mounted', updateTargets)
});
</script>

<template>
  <VTour ref="tour" name="index-tour" :steps="steps" />
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
