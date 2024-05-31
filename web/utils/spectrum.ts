import type { MatchedFragmentPeak } from "biowclib-mz";
import { MzErrorTolType, annotateSpectrum } from "biowclib-mz";
import isEqual from "lodash/isEqual";

import type {
  KoinaSpectrumAnnotations,
  KoinaSpectrumMzs,
  KoinaSpectrumIntensities,
  KoinaSpectrumPeptideSequences,
  KoinaSpectrumPrecursorCharges,
  KoinaSpectrumCollisionEnergies,
  KoinaSpectrumInstrumentTypes,
} from "@/utils/koina-api";
import type { ProxiSpectrum } from "./fetch-proxi-spectrum";

export interface KoinaSpectrum {
  annotation: string[];
  mz: number[];
  intensities: number[];
  peptideSequence: string;
  precursorCharge: number;
  collisionEnergy?: number;
  instrumentType?: string;
}

export function koinaResponseToSpectra(
  inputs: KoinaData[],
  outputs: KoinaData[],
): KoinaSpectrum[] {
  const annotations = findKoinaDataOrThrow<KoinaSpectrumAnnotations>(
    outputs,
    "annotation",
  );

  const mzs = findKoinaDataOrThrow<KoinaSpectrumMzs>(outputs, "mz");
  const intensities = findKoinaDataOrThrow<KoinaSpectrumIntensities>(
    outputs,
    "intensities",
  );

  const peptideSequences = findKoinaDataOrThrow<KoinaSpectrumPeptideSequences>(
    inputs,
    "peptide_sequences",
  );

  const precursorCharges = findKoinaDataOrThrow<KoinaSpectrumPrecursorCharges>(
    inputs,
    "precursor_charges",
  );

  const collisionEnergies = findKoinaData<KoinaSpectrumCollisionEnergies>(
    inputs,
    "collision_energies",
  );

  const instrumentTypes = findKoinaData<KoinaSpectrumInstrumentTypes>(
    inputs,
    "instrument_types",
  );

  if (
    !isEqual(annotations.shape, mzs.shape) ||
    !isEqual(annotations.shape, intensities.shape)
  ) {
    throw new Error("Shape mismatch");
  }

  if (!isEqual(peptideSequences.shape, precursorCharges.shape)) {
    throw new Error("Shape mismatch");
  }

  if (
    annotations.shape[0] !== peptideSequences.shape[0] ||
    annotations.shape[0] !== precursorCharges.shape[0]
  ) {
    throw new Error("Shape mismatch");
  }

  if (
    annotations.data.length !== mzs.data.length ||
    annotations.data.length !== intensities.data.length
  ) {
    throw new Error("Data length mismatch");
  }

  const spectra: KoinaSpectrum[] = [];

  for (let i = 0; i < mzs.shape[0]; i++) {
    const spectrumMzs = [];
    const spectrumIntensities = [];
    const spectrumAnnotations = [];

    for (let j = i * mzs.shape[1]; j < (i + 1) * mzs.shape[1]; j++) {
      if (
        (mzs.data[j] === -1 && intensities.data[j] !== -1) ||
        (mzs.data[j] !== -1 && intensities.data[j] === -1)
      ) {
        throw new Error(`Missing value mismatch at index ${j}`);
      }

      if (mzs.data[j] !== -1) {
        spectrumMzs.push(mzs.data[j]);
        spectrumIntensities.push(intensities.data[j]);
        spectrumAnnotations.push(annotations.data[j]);
      }
    }

    spectra.push({
      annotation: spectrumAnnotations,
      mz: spectrumMzs,
      intensities: spectrumIntensities,
      peptideSequence: peptideSequences.data[i],
      precursorCharge: precursorCharges.data[i],
      collisionEnergy: collisionEnergies?.data[i],
      instrumentType: instrumentTypes?.data[i],
    });
  }

  return spectra;
}

export function koinaSpectrumToMatchedFragmentPeaks(
  spectrum: KoinaSpectrum,
): MatchedFragmentPeak[] {
  const matchedFragmentPeaks: MatchedFragmentPeak[] = [];

  const seqLen = spectrum.peptideSequence.length;

  for (const [i, annotation] of spectrum.annotation.entries()) {
    const annotationSplit = annotation.match(
      /(?<ion_type>y|b)(?<frag_index>\d+)\+(?<charge>\d+)/,
    )?.groups;

    const frag_index = parseInt(annotationSplit?.frag_index ?? "-1", 10);
    const ion_type = annotationSplit?.ion_type ?? "";
    const charge = parseInt(annotationSplit?.charge ?? "-1", 10);
    const aa_position = ion_type === "b" ? frag_index : seqLen - frag_index + 1;

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
      free() {}, // TODO: biowc-spectrum type should be refactored to not directly use the MatchedFragmentPeak class
    });
  }

  return matchedFragmentPeaks;
}

export function koinaSpectrumToProxiSpectrum(
  spectrum: KoinaSpectrum,
): ProxiSpectrum {
  return {
    mzs: spectrum.mz,
    intensities: spectrum.intensities,
    attributes: [],
  };
}

export function proxiSpectrumsToKoinaSpectrums(
  spectrums: ProxiSpectrum[],
  inputs: KoinaData[],
): KoinaSpectrum[] {
  const peptideSequences = findKoinaDataOrThrow<KoinaSpectrumPeptideSequences>(
    inputs,
    "peptide_sequences",
  );

  const precursorCharges = findKoinaDataOrThrow<KoinaSpectrumPrecursorCharges>(
    inputs,
    "precursor_charges",
  );

  const collisionEnergies = findKoinaData<KoinaSpectrumCollisionEnergies>(
    inputs,
    "collision_energies",
  );

  const instrumentTypes = findKoinaData<KoinaSpectrumInstrumentTypes>(
    inputs,
    "instrument_types",
  );

  if (spectrums.length !== peptideSequences.shape[0]) {
    throw new Error("Shape mismatch");
  }

  const koinaSpectrums: KoinaSpectrum[] = [];

  for (let i = 0; i < spectrums.length; i++) {
    koinaSpectrums.push({
      annotation: [],
      mz: spectrums[i].mzs,
      intensities: spectrums[i].intensities,
      peptideSequence: peptideSequences.data[i],
      precursorCharge: precursorCharges.data[i],
      collisionEnergy: collisionEnergies?.data[i],
      instrumentType: instrumentTypes?.data[i],
    });
  }

  return koinaSpectrums;
}

export function annotateKoinaSpectrum(
  spectrum: KoinaSpectrum,
  pepSeq: string,
): MatchedFragmentPeak[] {
  return annotateSpectrum(
    pepSeq,
    new Float64Array(spectrum.mz),
    new Float64Array(spectrum.intensities),
    -20,
    20,
    MzErrorTolType.Ppm,
  );
}
