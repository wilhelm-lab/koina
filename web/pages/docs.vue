<script setup lang="ts">
import TheHeader from "~/components/partials/TheHeader.vue";
import TheFooter from "~/components/partials/TheFooter.vue";
import { VTour } from "#components";
import type { TourStep } from "#nuxt-tour/props";
import { querySelector } from 'shadow-dom-selector';


definePageMeta({
  layout: false,
});


async function centerOnElement(element: HTMLElement) {
  await sleep(2000)
  return element.scrollIntoView({ block: 'center', inline: 'center' });
}

// Store the tour component in a ref
const tour = ref<InstanceType<typeof VTour> | null>(null);
// define the steps for the tour
const steps = computed<TourStep[]>(() => {
  return [
    {
      target: 'rapi-doc$ #the-main-body > nav',
      title: "Available models",
      body: "Here you see all models currently implemented in Koina. You can click on a model to open its documentation. Models are grouped by the property the model is predicting e.g. fragment ion intensity, retention time, etc.",
      onNext: () => { // We need this hack because the normal scroll function doens't work probably because of the shadow DOM
        centerOnElement(steps.value[1].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ #the-main-body > nav > div',
      title: "Search models",
      body: "You can filter models by their name.",
      onNext: async () => {
        // This is janky but the nuxt internal navigation dosn't reload the dom
        // And I don't think a lot of people are ever going to see this
        console.log(window.location.href.slice(-5))
        if (window.location.href.slice(-5) == '/docs') {
          console.log('reloading')
          window.location.href += '#post-/Prosit_2019_intensity/infer'
          await sleep(2000)
        }
        centerOnElement(steps.value[2].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ [part="section-operation-summary"]',
      title: "Model documentation",
      body: "The documentation of all models is structured in the same way",
      onNext: () => {
        centerOnElement(steps.value[3].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ .m-markdown #summary',
      title: "Model summary",
      body: "There is a general summary that describes what the model is doing, how it was trained and what it can be used for.",
      onNext: () => {
        centerOnElement(steps.value[4].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ .m-markdown #citaton',
      title: "Citation",
      body: "If you use a model in your research please make sure to cite the authors of the model as well as Koina.",
      onNext: () => {
        centerOnElement(steps.value[5].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ .table-title',
      title: "Code samples",
      body: "Koina aims to be as user-friendly as possible. We provide ready to use code samples for different programming languages.",
      onNext: () => {
        centerOnElement(steps.value[6].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ api-request$ .request-body-container',
      title: "Interactive example ",
      body: "You can also use a model directly in your browser. This is a great way to test a model before you implement it in your own code.",
      onNext: () => {
        centerOnElement(steps.value[7].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ api-request$ .request-body-param-user-input',
      title: "Request body",
      body: "The request body is where you input your data. You can adjust the values to see how the model responds.",
      onNext: () => {
        centerOnElement(steps.value[8].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ api-request$ button.m-btn.primary.thin-border',
      title: "Try button",
      body: "Click the try button to send a request directly from your browser",
      onNext: async () => {
        centerOnElement(steps.value[9].target as HTMLElement)
      }
    },
    {
      target: 'rapi-doc$ api-request$ button.m-btn.primary.thin-border',
      title: "The Response",
      body: "The response will pop up directly below the try button. It's content will depend on the model you are using.",
    },
    {
      target: '#contact-button',
      title: "Reach out",
      body: "If you have any further questions or suggestions don't hesitate to reach out via email or on our GitHub repository.",
    },
  ];
});

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function resetTour() {
  updateTargets();
  tour.value?.resetTour();
}

async function updateTargets(sleepTime: number = 500) {
  await nextTick();
  let allElementsReplaced = false;

  while (!allElementsReplaced && sleepTime < 100000) {
    allElementsReplaced = true;
    steps.value.forEach((step, index) => {
      const element = typeof step.target === 'string' ? querySelector(step.target) : step.target;
      if (element) {
        steps.value[index].target = element as HTMLElement;
      } else {
        allElementsReplaced = false;
      }
    });

    if (!allElementsReplaced) {
      await sleep(sleepTime); // Pauses for the specified sleep time
      sleepTime *= 2; // Double the sleep time for the next iteration
    }
  }

  tour.value?.startTour();
}

const { $listen, $off } = useNuxtApp()

onMounted(() => {
  $listen('startTour', resetTour)
  $listen('rapi-doc-mounted', () => updateTargets())
});

onUnmounted(() => {
  $off('startTour', resetTour)
  $off('rapi-doc-mounted', () => updateTargets())
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
