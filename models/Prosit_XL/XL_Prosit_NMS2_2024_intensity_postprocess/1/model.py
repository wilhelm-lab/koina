import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from postprocess import create_masking, apply_masking
from pyteomics import proforma
import constants
from typing import Dict, List, Optional, Tuple


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


def find_crosslinker_position(peptide_sequence: str):
    peptide_sequence = re.sub(r"\[UNIMOD:(?!1898).*?\]", "", peptide_sequence)
    crosslinker_position = re.search(r"K(?=\[UNIMOD:1898\])", peptide_sequence)
    crosslinker_position = crosslinker_position.start() + 1
    return crosslinker_position


def _get_modifications(
    peptide_sequence: str,
) -> Optional[Tuple[Dict[int, float], int, str]]:
    """
    Get modification masses and position in a peptide sequence.

    :param peptide_sequence: Modified peptide sequence
    :return: tuple with - dictionary of modification_position => mod_mass
                        - 2 if there is an isobaric tag on the n-terminal, else 1
                        - sequence without modifications
    """
    modification_deltas = {}
    tmt_n_term = 1
    modifications = constants.MOD_MASSES.keys()
    modification_mass = constants.MOD_MASSES

    # Handle terminal modifications here
    for possible_tmt_mod in constants.TMT_MODS.values():
        if peptide_sequence.startswith(possible_tmt_mod):  # TMT_6
            tmt_n_term = 2
            modification_deltas.update({0: constants.MOD_MASSES[possible_tmt_mod]})
            peptide_sequence = peptide_sequence[len(possible_tmt_mod) :]
            break

    if "(" in peptide_sequence:
        return None

    while "[" in peptide_sequence:
        found_modification = False
        modification_index = peptide_sequence.index("[")
        for mod in modifications:
            if (
                peptide_sequence[modification_index : modification_index + len(mod)]
                == mod
            ):
                if modification_index - 1 in modification_deltas:
                    modification_deltas.update(
                        {
                            modification_index
                            - 1: modification_deltas[modification_index - 1]
                            + modification_mass[mod]
                        }
                    )
                else:
                    modification_deltas.update(
                        {modification_index - 1: modification_mass[mod]}
                    )
                peptide_sequence = (
                    peptide_sequence[0:modification_index]
                    + peptide_sequence[modification_index + len(mod) :]
                )
                found_modification = True
        if not found_modification:
            return None

    return modification_deltas, tmt_n_term, peptide_sequence


def compute_peptide_mass(sequence: str) -> float:
    """
    Compute the theoretical mass of the peptide sequence.

    :param sequence: Modified peptide sequence
    :raises AssertionError: if an unknown modification has been found in the peptide sequence
    :return: Theoretical mass of the sequence
    """
    peptide_sequence = sequence
    modifications = _get_modifications(peptide_sequence)
    if modifications is None:
        raise AssertionError("Modification not found.")
    else:
        modification_deltas, tmt_n_term, peptide_sequence = modifications

    peptide_length = len(peptide_sequence)
    if peptide_length > 30:
        # return [], -1, ""
        return -1.0

    n_term_delta = 0.0

    # get mass delta for the c-terminus
    c_term_delta = 0.0

    n_term = constants.ATOM_MASSES["H"] + n_term_delta  # n-terminal delta [N]
    c_term = (
        constants.ATOM_MASSES["O"] + constants.ATOM_MASSES["H"] + c_term_delta
    )  # c-terminal delta [C]
    h = constants.ATOM_MASSES["H"]

    ion_type_offsets = [n_term - h, c_term + h]

    # calculation:
    forward_sum = 0.0  # sum over all amino acids from left to right (neutral charge)

    for i in range(0, peptide_length):  # generate substrings
        forward_sum += constants.AA_MASSES[peptide_sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
    return forward_sum + ion_type_offsets[0] + ion_type_offsets[1]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in_1 = (
                pb_utils.get_input_tensor_by_name(request, "peptides_in_1:0")
                .as_numpy()
                .tolist()
            )

            peptide_in_2 = (
                pb_utils.get_input_tensor_by_name(request, "peptides_in_2:0")
                .as_numpy()
                .tolist()
            )

            peaks_in = pb_utils.get_input_tensor_by_name(
                request, "peaks_in:0"
            ).as_numpy()

            precursor_charges_in = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges_in:0"
            ).as_numpy()

            peptide_in_1 = [x[0].decode("utf-8") for x in peptide_in_1]
            peptide_in_2 = [x[0].decode("utf-8") for x in peptide_in_2]

            peptide_2_sequence = [s.replace("[UNIMOD:1898]", "") for s in peptide_in_2]
            peptide_in_2_mass = [compute_peptide_mass(s) for s in peptide_2_sequence]

            crosslinker_position = [find_crosslinker_position(x) for x in peptide_in_1]
            peptide_length = [len(x) for x in internal_without_mods(peptide_in_1)]

            charge = [
                min(np.argmax(sublist) + 1, 3) for sublist in precursor_charges_in
            ]

            mask = create_masking(precursor_charges_in, peptide_length)
            masked_peaks = apply_masking(peaks_in, mask)

            fragmentmz = [
                initialize_peaks(seq, chg, mass, pos)
                for seq, chg, mass, pos in zip(
                    peptide_in_1, charge, peptide_in_2_mass, crosslinker_position
                )
            ]
            fragmentmz = np.array(fragmentmz)
            fragmentmz = fragmentmz.astype("float32")

            fragmentmz[np.isnan(masked_peaks)] = -1
            masked_peaks[np.isnan(masked_peaks)] = -1

            output_tensors = [
                pb_utils.Tensor("intensities", masked_peaks.astype(self.output_dtype)),
                pb_utils.Tensor("mz", fragmentmz.astype(self.output_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses


def initialize_peaks(
    peptide_alpha: str, charge: int, peptide_beta_mass: int, crosslinker_pos: int
) -> np.ndarray:

    modifications = _get_modifications(peptide_alpha)
    if modifications is None:
        raise AssertionError("Modification not found.")
    else:
        modification_deltas, _, peptide_sequence = modifications

    peptide_length = len(peptide_sequence)

    n_term_delta = 0.0
    c_term_delta = 0.0
    n_term = constants.ATOM_MASSES["H"] + n_term_delta  # n-terminal delta [N]
    c_term = (
        constants.ATOM_MASSES["O"] + constants.ATOM_MASSES["H"] + c_term_delta
    )  # c-terminal delta [C]
    h = constants.ATOM_MASSES["H"]

    ion_type_offsets = [n_term - h, c_term + h]
    ion_type_masses = [0.0, 0.0]
    number_of_ion_types = len(ion_type_offsets)

    ion_masses = [None] * 174

    forward_sum = 0.0  # sum over all amino acids from left to right (neutral charge)
    backward_sum = 0.0  # sum over all amino acids from right to left (neutral charge)
    for i in range(0, peptide_length - 1):  # generate substrings
        forward_sum += constants.AA_MASSES[peptide_sequence[i]]  # sum left to right
        if i in modification_deltas:  # add mass of modification if present
            forward_sum += modification_deltas[i]
        backward_sum += constants.AA_MASSES[
            peptide_sequence[peptide_length - i - 1]
        ]  # sum right to left
        if (
            peptide_length - i - 1 in modification_deltas
        ):  # add mass of modification if present
            backward_sum += modification_deltas[peptide_length - i - 1]

        ion_type_masses[0] = forward_sum + ion_type_offsets[0]  # b ion - ...
        ion_type_masses[1] = backward_sum + ion_type_offsets[1]  # y ion

        for ion_charge in range(
            constants.MIN_CHARGE, charge + 1
        ):  # generate ion in different charge states
            # positive charge is introduced by protons (or H - ELECTRON_MASS)
            charge_delta = ion_charge * constants.PARTICLE_MASSES["PROTON"]
            for ion_type in range(0, number_of_ion_types):  # generate all ion types

                if ion_type == 0 and i + 1 >= crosslinker_pos:
                    # b_xl-ions
                    ion_mass_with_peptide_beta = (
                        ion_type_masses[ion_type] + peptide_beta_mass
                    )
                    mass = (ion_mass_with_peptide_beta + charge_delta) / ion_charge

                elif ion_type == 1 and i >= peptide_length - crosslinker_pos:
                    # y_xl-ions
                    ion_mass_with_peptide_beta = (
                        ion_type_masses[ion_type] + peptide_beta_mass
                    )
                    mass = (ion_mass_with_peptide_beta + charge_delta) / ion_charge

                else:
                    mass = (ion_type_masses[ion_type] + charge_delta) / ion_charge

                if ion_type == 1:  # y ions
                    index = i * 6 + ion_charge - 1
                else:  # b ions
                    index = i * 6 + ion_charge + 3 - 1

                # Store the calculated mass in the appropriate index
                ion_masses[index] = mass

    return np.array(ion_masses)
