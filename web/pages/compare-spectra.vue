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
  <div class="py-4 px-4 max-w-5xl mx-auto">
    <h1 class="text-2xl">Compare Spectra</h1>

    <CompareSpectrumForm id="reference" v-model="referenceConfig" class="mb-4">
      <template #title> Reference Spectrum </template>
    </CompareSpectrumForm>

    <CompareSpectrumForm id="mirror" v-model="mirrorConfig">
      <template #title> Mirror Spectrum </template>
    </CompareSpectrumForm>

    <Button class="mt-2" @click="submit()"> Compare Spectra </Button>

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
    >
    </biowc-spectrum>

    <div v-if="loading">Loading...</div>

    <div class="text-red-500">{{ error }}</div>
  </div>
</template>
