import type { MatchedFragmentPeak } from 'biowclib-mz'
import isEqual from 'lodash/isEqual'

export interface KoinaSpectrumAnnotations {
  data: string[]
  datatype: 'BYTES'
  name: 'annotation'
  shape: number[]
}

export interface KoinaSpectrumMzs {
  data: number[]
  datatype: 'FP32'
  name: 'mz'
  shape: number[]
}

export interface KoinaSpectrumIntensities {
  data: number[]
  datatype: 'FP32'
  name: 'intensities'
  shape: number[]
}

export interface KoinaSpectrumPeptideSequences {
  data: string[]
  datatype: 'BYTES'
  name: 'peptide_sequences'
  shape: number[]
}

export interface KoinaSpectrumPrecursorCharges {
  data: number[]
  datatype: 'INT32'
  name: 'precursor_charges'
  shape: number[]
}

export interface Spectrum {
  annotation: string[]
  mz: number[]
  intensities: number[]
  peptideSequence: string
  precursorCharge: number
}

export function koinaToSpectra(
  annotations: KoinaSpectrumAnnotations,
  mzs: KoinaSpectrumMzs,
  intensities: KoinaSpectrumIntensities,
  peptideSequences: KoinaSpectrumPeptideSequences,
  precursorCharges: KoinaSpectrumPrecursorCharges,
): Spectrum[] {
  if (!isEqual(annotations.shape, mzs.shape) || !isEqual(annotations.shape, intensities.shape)) {
    throw new Error('Shape mismatch')
  }

  if (!isEqual(peptideSequences.shape, precursorCharges.shape)) {
    throw new Error('Shape mismatch')
  }

  if (annotations.shape[0] !== peptideSequences.shape[0] || annotations.shape[0] !== precursorCharges.shape[0]) {
    throw new Error('Shape mismatch')
  }

  if (annotations.data.length !== mzs.data.length || annotations.data.length !== intensities.data.length) {
    throw new Error('Data length mismatch')
  }

  const spectra: Spectrum[] = []

  for (let i = 0; i < mzs.shape[0]; i++) {
    const spectrumMzs = []
    const spectrumIntensities = []
    const spectrumAnnotations = []

    for (let j = i * mzs.shape[1]; j < (i + 1) * mzs.shape[1]; j++) {
      if ((mzs.data[j] === -1 && intensities.data[j] !== -1) || (mzs.data[j] !== -1 && intensities.data[j] === -1)) {
        throw new Error(`Missing value mismatch at index ${j}`)
      }

      if (mzs.data[j] !== -1) {
        spectrumMzs.push(mzs.data[j])
        spectrumIntensities.push(intensities.data[j])
        spectrumAnnotations.push(annotations.data[j])
      }
    }

    spectra.push({
      annotation: spectrumAnnotations,
      mz: spectrumMzs,
      intensities: spectrumIntensities,
      peptideSequence: peptideSequences.data[i],
      precursorCharge: precursorCharges.data[i],
    })
  }

  return spectra
}

export function spectrumToMatchedFragmentPeaks(spectrum: Spectrum): MatchedFragmentPeak[] {
  const matchedFragmentPeaks: MatchedFragmentPeak[] = []

  const seqLen = spectrum.peptideSequence.length

  for (const [i, annotation] of spectrum.annotation.entries()) {
    const annotationSplit
      = annotation.match(/(?<ion_type>y|b)(?<frag_index>\d+)\+(?<charge>\d+)/)?.groups

    const frag_index = parseInt(annotationSplit?.frag_index ?? '-1', 10)
    const ion_type = annotationSplit?.ion_type ?? ''
    const charge = parseInt(annotationSplit?.charge ?? '-1', 10)
    const aa_position = ion_type === 'b' ? frag_index : seqLen - frag_index + 1

    matchedFragmentPeaks.push({
      aa_position,
      charge,
      frag_index,
      ion_type,
      mz_error: 0,
      peak_index: i,
      peak_intensity: spectrum.intensities[i],
      peak_mz: spectrum.mz[i],
      theo_mz: spectrum.mz[i],
      free() { }, // TODO: biowc-spectrum type should be refactored to not directly use the MatchedFragmentPeak class
    })
  }

  return matchedFragmentPeaks
}
