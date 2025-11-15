<script setup lang="ts">
import TheHero from "~/components/partials/TheHero.vue";
import BaseButton from "~/components/ui/BaseButton.vue";
import { VTour } from "#components";
import type { TourStep } from "#nuxt-tour/props";
import { onMounted, ref } from 'vue';

// Store the tour component in a ref
const tour = ref<InstanceType<typeof VTour> | null>(null);
// define the steps for the tour
const steps: TourStep[] = [
  {
    target: "#logo",
    title: "Do you want a brief overview of Koina?",
  },
  {
    target: "#content-get-predictions-to-you",
    title: "What is the goal of Koina?",
    body: "The main goal of Koina is to make proteomics ML models easy to use.",
  },
  {
    target: "#content-easy-to-do",
    title: "Is it actually easy?",
    body: "If you can run code in any programming language, you can directly use Koina. If you don't want that, there are already a lot of tools that use Koina to make using ML in your data analysis even easier to do (Fragpipe, Skyline, EncyclopeDIA, Oktoberfest).",
  },
  {
    target: "#btn-docu",
    title: "How does it work?",
    body: "Check out the documentation to learn about how to directly use Koina.",
  },
  {
    target: "#btn-comp-spec",
    title: "How does it work?",
    body: "We also offer an option to compare fragment ion predictions of different models and experimental spectra.",
  },
  {
    target: "#get-involved",
    title: "Consider how you can support Koina",
    body: "Koina is open-source and free to use. If Koina is useful in your research please cite us as well as the model you used. If you want to do more, you can host your own Koina instance or add your own model to Koina! You can find more information on how to do that in our GitHub repository.",
  },
];

const { $listen, $off } = useNuxtApp()

function resetTour() {
  tour.value?.resetTour();
}

onMounted(() => {
  tour.value?.startTour();
  $listen('startTour', resetTour)
});

onUnmounted(() => {
  $off('startTour', resetTour)
});

</script>

<template>
  <VTour ref="tour" name="index-tour" :steps="steps" />
  <div>
    <TheHero />

    <div class="py-16 px-4 max-w-5xl mx-auto">
      <h2 id="get-started" class="text-3xl mb-4">What is Koina?</h2>
      <p id="content-get-predictions-to-you">
        Koina is a model repository enabling the remote execution of models. 
        Predictions are generated as a response to HTTP/S requests, the standard protocol used for nearly all web traffic. 
        As such, HTTP/S requests can be easily generated in any programming language without requiring specialized hardware. 
        This design also enables users to share centralized hardware to utilize it more efficiently. 
        It also allows for easy horizontal scaling depending on the demand of the user base.  
      </p>

      <p id="content-easy-to-do">
        To minimize the barrier of entry and “democratize” access to ML models, we provide a public network of Koina instances at <a href="https://koina.wilhelmlab.org">koina.wilhelmlab.org</a>. 
        The computational workload is automatically distributed to processing nodes hosted at different research institutions and spin-offs across Europe. 
        Each processing node provides computational resources to the service network, always aiming at just-in-time results delivery.
      </p>


      <p>
        In the spirit of open and collaborative science, we envision that this public Koina-Network can be scaled to meet the community’s needs 
        by various research groups or institutions dedicating hardware. 
        This can also vastly improve latency if servers are available geographically nearby. 
        Alternatively, if data security is a concern, private instances within a local network can be easily deployed using the provided <a href="https://github.com/wilhelm-lab/koina/pkgs/container/koina">docker image</a>.
      </p>

      <p>
        Koina is a community driven project. 
        It is fuly <a href="https://github.com/wilhelm-lab/koina">open-source</a>.
        We welcome all contributions and feedback! Feel free to reach out to <a href="mailto:Ludwig.Lautenbacher@tum.de">us</a> or open an issue on our <a href="https://github.com/wilhelm-lab/koina/issues/new">GitHub repository</a>.
      </p>
      
      <p>
        Check out the <a href="/docs">documentation</a> to learn more about how
        to use Koina.
      </p>
    </div>

    <div class="bg-neutral-100">
      <div class="py-16 px-4 max-w-5xl mx-auto">
        <h2 id="get-involved" class="text-3xl mb-4">Get involved!</h2>
        <p>
          There are two main ways to get involved with Koina. You can either host an
          instance of Koina to make more open resources available to the
          community or you can add your own model to Koina. If you want to make
          your Koina instance available via
          <a href="https://koina.wilhelmlab.org">koina.wilhelmlab.org</a>,
          please contact us via 
          <a href="mailto:Ludwig.Lautenbacher@tum.de"
            >E-Mail</a
          >.
        </p>
        <p class="mb-8">
          For more information on how to host an instance of Koina or add your
          own model, check out the links below.
        </p>
        <BaseButton to="https://github.com/wilhelm-lab/koina?tab=readme-ov-file#hosting-your-own-server">
          Host Koina
        </BaseButton>
        <BaseButton to="https://github.com/wilhelm-lab/koina?tab=readme-ov-file#adding-your-own-model" class="ml-4">
          Add your model
        </BaseButton>
      </div>
    </div>

    <div class="py-16 px-4 max-w-5xl mx-auto">
      <h2 class="text-3xl mb-4">Organizations hosting Koina</h2>
      <div class="flex justify-around flex-wrap">
        <a href="https://www1.ls.tum.de/en/compms/home/" class="mb-2">
          <img src="~/assets/img/tum-logo.png" alt="TUM logo" class="h-16" />
        </a>
        <a href="https://fgcz.ch/" class="ml-2 mb-2">
          <img src="~/assets/img/eth-uzh-logo.svg" alt="ETH UZH logo" class="h-16" />
        </a>
        <a href="https://www.msaid.de/" class="ml-2 mb-2">
          <img src="~/assets/img/msaid-logo.png" alt="MSAID logo" class="h-16" />
        </a>
        <a href="https://www.crg.eu/" class="ml-2 mb-2">
          <img src="~/assets/img/crg-logo.png" alt="BIYSC CRG logo" class="h-16" />
        </a>
        <a href="https://www.upf.edu/" class="ml-2 mb-2">
          <img src="~/assets/img/upf-logo.png" alt="UPF logo" class="h-16" />
        </a>
      </div>
    </div>
  </div>
</template>
