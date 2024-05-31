<script setup lang="ts">
import { AccordionTrigger } from "radix-vue";
import { ChevronDown } from "lucide-vue-next";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
} from "@/components/ui/accordion";
import type { KoinaSpectrum } from "@/utils/spectrum";
import { koinaSpectrumToMatchedFragmentPeaks } from "@/utils/spectrum";
import "biowc-spectrum";

const props = defineProps<{
  spectras: KoinaSpectrum[];
}>();

const keyedSpectras = computed(() => {
  return props.spectras.map((spectrum, index) => {
    return {
      ...spectrum,
      key: `${index}`,
      matchedIons: koinaSpectrumToMatchedFragmentPeaks(spectrum),
    };
  });
});

console.log("keyedSpectras", keyedSpectras);

watch(keyedSpectras, (newSpectras) => {
  console.log("newSpectras", newSpectras);
});

const openKey = ref("");

const setOpenKey = (key: string) => {
  if (openKey.value === key) {
    openKey.value = "";
  } else {
    openKey.value = key;
  }
};
</script>

<template>
  <h3>Result Spectra</h3>
  <!-- using inline styles here so the styles stil still work if they get teleported into rapidoc shadow DOM -->
  <Accordion type="multiple" collapsible>
    <AccordionItem
      v-for="spectrum in keyedSpectras"
      :key="spectrum.key"
      :value="spectrum.key"
      :style="{
        border: '1px solid #e5e7eb',
        'border-radius': '0.25rem',
        'background-color': 'white',
        'margin-top': '1rem',
      }"
    >
      <AccordionTrigger
        :style="{
          width: '100%',
          display: 'flex',
          'justify-content': 'space-between',
          'align-items': 'center',
          padding: '0.5rem 1rem',
          'font-size': '1rem',
          border: '1px solid #e5e7eb',
          'border-radius': '0.25rem',
          cursor: 'pointer',
        }"
        @click="setOpenKey(spectrum.key)"
      >
        <div
          :style="{
            display: 'flex',
            'align-items': 'center',
            gap: '1rem',
            'justify-content': 'start',
          }"
        >
          <code
            :style="{
              'font-size': '1.125rem',
              'font-family': 'monospace',
            }"
            >{{ spectrum.peptideSequence }}</code
          >
          <span>Precursor charge: {{ spectrum.precursorCharge }}+</span>
          <span v-if="spectrum.collisionEnergy"
            >Collision energy: {{ spectrum.collisionEnergy }}</span
          >
        </div>
        <ChevronDown
          :style="{
            height: '1.25rem',
            width: '1.25rem',
            transition: 'transform 0.2s',
            transform:
              openKey === spectrum.key ? 'rotate(180deg)' : 'rotate(0deg)',
          }"
        />
      </AccordionTrigger>

      <AccordionContent
        :style="{
          padding: '0.5rem',
        }"
      >
        <!-- eslint-disable vue/attribute-hyphenation -->
        <biowc-spectrum
          .spectrum="{
            attributes: [],
            intensities: spectrum.intensities,
            mzs: spectrum.mz,
          }"
          .pepSeq="spectrum.peptideSequence"
          .charge="spectrum.precursorCharge"
          .normalizeIntensity="true"
          .matchedIons="spectrum.matchedIons"
          .hideErrorPlot="true"
          :style="{
            display: 'block',
            marginTop: '2rem',
          }"
        />
        <!-- eslint-enable vue/attribute-hyphenation -->
      </AccordionContent>
    </AccordionItem>
  </Accordion>
</template>
