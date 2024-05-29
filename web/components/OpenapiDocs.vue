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

  <Teleport
    v-if="spectraResults.length"
    :to="teleportTarget"
  >
    <SpectraResults :spectras="spectraResults" />
  </Teleport>
</template>

<script setup lang="ts">
/* eslint-disable @typescript-eslint/no-explicit-any */
import 'rapidoc'

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

const spectraResults = ref<Spectrum[]>([])
const teleportTarget = ref<HTMLElement | null>(null)

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
      spectraResults.value = []

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

        const modelPath = e.detail.request.url.match(/.*?(\/[^/]*?\/infer)$/)![1]
        const apiRequestEl = rapidoc.value.shadowRoot.querySelector(`api-request[path="${modelPath}"]`)

        const teleportTargetEl = document.createElement('div')
        apiRequestEl.insertAdjacentElement('afterend', teleportTargetEl)
        teleportTarget.value = teleportTargetEl

        spectraResults.value = koinaToSpectra(annotations, mzs, intensities, peptideSequences, precursorCharges)
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
