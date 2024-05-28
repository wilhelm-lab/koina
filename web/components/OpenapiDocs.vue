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
</template>

<script setup lang="ts">
/* eslint-disable @typescript-eslint/no-explicit-any */
import 'rapidoc'

import 'biowc-spectrum'

import type { BiowcSpectrum } from 'biowc-spectrum'
import type {
  KoinaSpectrumAnnotations,
  KoinaSpectrumIntensities,
  KoinaSpectrumMzs,
  KoinaSpectrumPeptideSequences,
  KoinaSpectrumPrecursorCharges } from '@/utils/transform-spectrum'
import {
  koinaToSpectra,
} from '@/utils/transform-spectrum'

const rapidoc: any = ref(null)

onMounted(() => {
  document.body.classList.add('overflow-hidden')

  if (rapidoc.value && import.meta.client) {
    // This is a bit hacky, but there seems to be no other way to get the request body in after-try event.
    let lock = false
    let peptideSequences: KoinaSpectrumPeptideSequences | null = null
    let precursorCharges: KoinaSpectrumPrecursorCharges | null = null

    rapidoc.value.addEventListener('before-try', (e: CustomEvent) => {
      if (lock) {
        throw new Error('Please wait for the previous \'try\'-request to finish.')
      }

      lock = true

      const requestBody = JSON.parse(e.detail.request.body)

      peptideSequences
        = requestBody.inputs.find((input: any) => input.name === 'peptide_sequences') as KoinaSpectrumPeptideSequences
      precursorCharges
        = requestBody.inputs.find((input: any) => input.name === 'precursor_charges') as KoinaSpectrumPrecursorCharges
    })

    rapidoc.value.addEventListener('after-try', (e: CustomEvent) => {
      try {
        if (!peptideSequences) {
          throw new Error('Peptide sequences from \'before-try\' not found.')
        }

        if (!precursorCharges) {
          throw new Error('Precursor charges from \'before-try\' not found.')
        }

        const annotations
          = e.detail.responseBody.outputs.find((output: any) => output.name === 'annotation') as KoinaSpectrumAnnotations
        const mzs
          = e.detail.responseBody.outputs.find((output: any) => output.name === 'mz') as KoinaSpectrumMzs
        const intensities
          = e.detail.responseBody.outputs.find((output: any) => output.name === 'intensities') as KoinaSpectrumIntensities

        const spectra = koinaToSpectra(annotations, mzs, intensities, peptideSequences, precursorCharges)

        const modelPath = e.detail.request.url.match(/.*?(\/[^/]*?\/infer)$/)![1]
        const apiRequestEl = rapidoc.value.shadowRoot.querySelector(`api-request[path="${modelPath}"]`)

        const biowcSpectrumElements = spectra.map((spectrum) => {
          const biowcSpectrumEl = document.createElement('biowc-spectrum') as BiowcSpectrum
          biowcSpectrumEl.spectrum = {
            attributes: [],
            intensities: spectrum.intensities,
            mzs: spectrum.mz,
          }
          biowcSpectrumEl.pepSeq = spectrum.peptideSequence
          biowcSpectrumEl.charge = spectrum.precursorCharge
          biowcSpectrumEl.normalizeIntensity = true
          biowcSpectrumEl.matchedIons = spectrumToMatchedFragmentPeaks(spectrum)
          biowcSpectrumEl.hideErrorPlot = true
          biowcSpectrumEl.style.display = 'block'
          biowcSpectrumEl.style.marginTop = '2rem'

          return biowcSpectrumEl
        })

        const previousBiowcSpectrumElements = apiRequestEl.parentElement!.querySelectorAll('biowc-spectrum') as NodeListOf<BiowcSpectrum>

        previousBiowcSpectrumElements.forEach((biowcSpectrumEl) => {
          biowcSpectrumEl.remove()
        })

        biowcSpectrumElements.reverse().forEach((biowcSpectrumEl) => {
          apiRequestEl.insertAdjacentElement('afterend', biowcSpectrumEl)
        })
      }
      finally {
        peptideSequences = null
        precursorCharges = null
        lock = false
      }
    })
  }
})

onBeforeUnmount(() => {
  document.body.classList.remove('overflow-hidden')
})
</script>
