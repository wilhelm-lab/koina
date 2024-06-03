export enum SpectrumConfigSource {
  PREDICTION = "prediction",
  USI = "usi",
}

export interface SpectrumFormModel {
  source: SpectrumConfigSource;
  usi?: string;
  model?: {
    name: string;
    inputs: KoinaData[];
  };
}

// TODO: generate from a common ground truth. Source for this as of this writing
// is description in the openapi spec.
export enum AlphaPeptInstrumentType {
  QE = "QE",
  LUMOS = "LUMOS",
  TIMSTOF = "TIMSTOF",
  SCIEXTOF = "SCIEXTOF",
}

export enum FragmentationType {
  HCD = "HCD",
  CID = "CID",
}
