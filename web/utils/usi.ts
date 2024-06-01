export function pepSeqFromUsi(usi: string): string | undefined {
  return usi.split(":")[5].split("/")[0] || undefined;
}

export function chargeFromUsi(usi: string): number | undefined {
  return parseInt(usi.split(":")[5].split("/")[1]) || undefined;
}
