// TODO: peptide_sequences_1 and peptide_sequences_2 from Prosit_2023_intensity_XL_CMS2
//       are not yet supported by the current implementation.

import type { ProxiSpectrum } from "./fetch-proxi-spectrum";

export interface KoinaData {
  name: string;
  data: any;
  datatype: string;
  shape: number[];
}

export interface KoinaSpectrumAnnotations extends KoinaData {
  data: string[];
  datatype: "BYTES";
  name: "annotation";
  shape: number[];
}

export interface KoinaSpectrumMzs extends KoinaData {
  data: number[];
  datatype: "FP32";
  name: "mz";
  shape: number[];
}

export interface KoinaSpectrumIntensities extends KoinaData {
  data: number[];
  datatype: "FP32";
  name: "intensities";
  shape: number[];
}

export interface KoinaSpectrumPeptideSequences extends KoinaData {
  data: string[];
  datatype: "BYTES";
  name: "peptide_sequences";
  shape: number[];
}

export interface KoinaSpectrumPrecursorCharges extends KoinaData {
  data: number[];
  datatype: "INT32";
  name: "precursor_charges";
  shape: number[];
}

export interface KoinaSpectrumCollisionEnergies extends KoinaData {
  data: number[];
  datatype: "FP32";
  name: "collision_energies";
  shape: number[];
}

export interface KoinaSpectrumInstrumentTypes extends KoinaData {
  data: string[];
  datatype: "BYTES";
  name: "instrument_types";
  shape: number[];
}

export const EMPTY_KOINA_DATA: Record<string, KoinaData> = {
  annotation: {
    name: "annotation",
    data: [],
    datatype: "BYTES",
    shape: [],
  } as KoinaSpectrumAnnotations,
  mz: {
    name: "mz",
    data: [],
    datatype: "FP32",
    shape: [],
  } as KoinaSpectrumMzs,
  intensities: {
    name: "intensities",
    data: [],
    datatype: "FP32",
    shape: [],
  } as KoinaSpectrumIntensities,
  peptide_sequences: {
    name: "peptide_sequences",
    data: [],
    datatype: "BYTES",
    shape: [],
  } as KoinaSpectrumPeptideSequences,
  precursor_charges: {
    name: "precursor_charges",
    data: [],
    datatype: "INT32",
    shape: [],
  } as KoinaSpectrumPrecursorCharges,
  collision_energies: {
    name: "collision_energies",
    data: [],
    datatype: "FP32",
    shape: [],
  } as KoinaSpectrumCollisionEnergies,
  instrument_types: {
    name: "instrument_types",
    data: [],
    datatype: "BYTES",
    shape: [],
  } as KoinaSpectrumInstrumentTypes,
};

export interface KoinaModelConfigInput {
  name: string;
  data_type: string;
  format: string;
  dims: number[];
  is_shape_tensor: boolean;
  allow_rugged_batch: boolean;
  optional: boolean;
}

export interface KoinaModelConfig {
  name: string;
  input: KoinaModelConfigInput[];
}

export async function fetchKoinaPrediction(
  model: string,
  inputs: KoinaData[],
): Promise<any> {
  const response = await fetch(
    `https://koina.wilhelmlab.org/v2/models/${model}/infer`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        id: "0",
        inputs,
      }),
    },
  );

  if (!response.ok) {
    throw new Error("Failed to fetch Koina prediction");
  }

  return response.json();
}

export async function fetchModelTritonConfig(
  model: string,
): Promise<KoinaModelConfig> {
  const response = await fetch(
    `https://koina.wilhelmlab.org/v2/models/${model}/config`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );

  if (!response.ok) {
    throw new Error("Failed to fetch Koina model config");
  }

  return response.json();
}

export async function fetchKoinaProxiSpectrum(
  peptideSequence: string,
  precursorCharge: number,
  collisionEnergy?: number,
  instrumentType?: string,
): Promise<ProxiSpectrum> {
  const params = new URLSearchParams({
    peptide_sequences: peptideSequence,
    precursor_charges: precursorCharge.toString(),
  });

  if (collisionEnergy) {
    params.append("collision_energies", collisionEnergy.toString());
  }

  if (instrumentType) {
    params.append("instrument_types", instrumentType);
  }

  const response = await fetch(
    `https://koina.wilhelmlab.org/v2/models/Prosit_2019_intensity/usi?${params.toString()}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    },
  );

  if (!response.ok) {
    throw new Error("Failed to fetch Koina ProxiSpectrum");
  }

  return response.json();
}

export function findKoinaData<T extends KoinaData>(
  data: KoinaData[],
  name: string,
): T | undefined {
  return data.find((d) => d.name === name) as T | undefined;
}

export function findKoinaDataOrThrow<T extends KoinaData>(
  data: KoinaData[],
  name: string,
): T {
  const found = findKoinaData<T>(data, name);

  if (!found) {
    throw new Error(`Data "${name}" not found`);
  }

  return found;
}
