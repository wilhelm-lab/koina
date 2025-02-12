<script setup lang="ts">
import { Button } from "@/components/ui/button";
import { SpectrumConfigSource, type SpectrumFormModel } from "~/utils/types";
import {
  annotateKoinaSpectrum,
  koinaResponseToSpectra,
  proxiSpectrumsToKoinaSpectrums,
} from "~/utils/spectrum";
import { koinaSpectrumToMatchedFragmentPeaks } from "~/utils/spectrum";
import "biowc-spectrum";
import { pepSeqFromUsi } from "~/utils/usi";
import type { MatchedFragmentPeak } from "biowclib-mz";
import { VTour } from "#components";
import type { TourStep } from "#nuxt-tour/props";
import { onMounted, ref } from 'vue';


// Store the tour component in a ref
const tour = ref<InstanceType<typeof VTour> | null>(null);
// define the steps for the tour
const steps: TourStep[] = [
  {
    target: "#reference",
    title: "How do I compare spectra?",
    body: "Pick your spectrum source, either a Koina model or a universal spectrum explorer.",
  },
  {
    target: "#reference",
    title: "How do I compare spectra?",
    body: "Then just enter the required inputs.",
  },
  {
    target: "#mirror",
    title: "How do I compare spectra?",
    body: "The same applies to for the bottom spectrum.",
  },
  {
    target: "#btn-comp-spec",
    title: "How do I compare spectra?",
    body: "Then just click on 'Compare Spectra' to predict or fetch the spectra via their USI.",
  },
];

function resetTour() {
  tour.value?.resetTour();
}

const { $listen, $off } = useNuxtApp()

onMounted(() => {
  tour.value?.startTour();
  $listen('startTour', resetTour)
});

onUnmounted(() => {
  $off('startTour', resetTour)
});


const referenceConfig = ref<SpectrumFormModel>({
  source: SpectrumConfigSource.PREDICTION,
  model: {
    name: "Prosit_2019_intensity",
    inputs: [],
  },
});

const mirrorConfig = ref<SpectrumFormModel>({
  source: SpectrumConfigSource.PREDICTION,
  model: {
    name: "AlphaPept_ms2_generic",
    inputs: [],
  },
});

const referenceSpectrum = ref<KoinaSpectrum | undefined>();
const mirrorSpectrum = ref<KoinaSpectrum | undefined>();
const referenceMatchedPeaks = ref<MatchedFragmentPeak[] | undefined>();
const mirrorMatchedPeaks = ref<MatchedFragmentPeak[] | undefined>();

async function fetchSpectrumForConfig(config: SpectrumFormModel): Promise<
  | {
      spectrum: KoinaSpectrum;
      matchedPeaks: MatchedFragmentPeak[];
    }
  | undefined
> {
  if (config.source === SpectrumConfigSource.PREDICTION) {
    const response = await fetchKoinaPrediction(
      config.model?.name || "",
      config.model?.inputs || [],
    );

    const spectrum = koinaResponseToSpectra(
      config.model?.inputs || [],
      response.outputs || [],
    )[0];

    const matchedPeaks = koinaSpectrumToMatchedFragmentPeaks(spectrum);

    return { spectrum, matchedPeaks };
  } else {
    const result = await fetchSpectrum(config.usi || "");
    if (result.ok) {
      const peptideSequence = pepSeqFromUsi(config.usi || "") || "";

      const spectrum = proxiSpectrumsToKoinaSpectrums(
        result.val,
        config.model?.inputs || [],
      )[0];

      spectrum.peptideSequence = peptideSequence;

      const matchedPeaks = annotateKoinaSpectrum(spectrum, peptideSequence);

      return { spectrum, matchedPeaks };
    }
  }
}

const loading = ref(false);
const error = ref<string | undefined>();

const referenceProxiSpectrum = ref<ProxiSpectrum | undefined>();
const mirrorProxiSpectrum = ref<ProxiSpectrum | undefined>();

async function submit() {
  loading.value = true;
  error.value = undefined;

  try {
    const referenceResult = await fetchSpectrumForConfig(referenceConfig.value);
    if (referenceResult) {
      const { spectrum: rSpec, matchedPeaks: rMatchedPeaks } = referenceResult;
      referenceSpectrum.value = rSpec;
      referenceMatchedPeaks.value = rMatchedPeaks;
    }

    const mirrorResult = await fetchSpectrumForConfig(mirrorConfig.value);
    if (mirrorResult) {
      const { spectrum: mSpec, matchedPeaks: mMatchedPeaks } = mirrorResult;
      mirrorSpectrum.value = mSpec;
      mirrorMatchedPeaks.value = mMatchedPeaks;
    }

    if (referenceSpectrum.value) {
      referenceProxiSpectrum.value = koinaSpectrumToProxiSpectrum(
        referenceSpectrum.value,
      );
    }

    if (mirrorSpectrum.value) {
      mirrorProxiSpectrum.value = koinaSpectrumToProxiSpectrum(
        mirrorSpectrum.value,
      );
    }
  } catch (e: any) {
    console.error(e);
    error.value = `An error occurred: ${(e as Error).message}`;
  } finally {
    loading.value = false;
  }
}
</script>

<template>
  <VTour trapFocus ref="tour" name="index-tour" :steps="steps" />
  <div class="py-4 px-4 max-w-5xl mx-auto">
    <h1 class="text-2xl">Compare Spectra</h1>

    <CompareSpectrumForm id="reference" v-model="referenceConfig" class="mb-4">
      <template #title> Reference Spectrum </template>
    </CompareSpectrumForm>

    <CompareSpectrumForm id="mirror" v-model="mirrorConfig">
      <template #title> Mirror Spectrum </template>
    </CompareSpectrumForm>

    <Button id="btn-comp-spec" class="mt-2" @click="submit()"> Compare Spectra </Button>

    <biowc-spectrum
      v-if="referenceSpectrum && mirrorSpectrum && !loading && !error"
      .spectrum="referenceProxiSpectrum"
      .matchedIons="referenceMatchedPeaks"
      .mirrorSpectrum="mirrorProxiSpectrum"
      .mirrorMatchedIons="mirrorMatchedPeaks"
      .pepSeq="referenceSpectrum.peptideSequence"
      .mirrorPepSeq="mirrorSpectrum.peptideSequence"
      .charge="referenceSpectrum.precursorCharge"
      .mirrorCharge="mirrorSpectrum.precursorCharge"
      .normalizeIntensity="true"
      .hideErrorPlot="true"
    />

    <div v-if="loading">Loading...</div>

    <div class="text-red-500">{{ error }}</div>
  </div>
</template>
