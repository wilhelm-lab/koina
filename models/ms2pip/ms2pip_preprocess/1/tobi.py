import psm_utils
from psm_utils import Peptidoform
from psm_utils.io.peptide_record import proforma_to_peprec
from ms2pip.cython_modules import ms2pip_pyx
from ms2pip.ms2pipC import apply_mods
from ms2pip.peptides import AMINO_ACID_IDS, Modifications
import numpy as np
import re


def remove_mods(seq, regex=r"\[.*?\]|\-"):
    """
    Function to remove any proforma mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    return re.sub(regex, "", seq)


class MinimalMS2PIP:
    def __init__(self, peptide: str, charge: int, maximal_length: int = 30):
        self.modifications = "-"  # modifications do not impact intensities
        self.maximal_length = maximal_length
        self.peptide = remove_mods(peptide)
        self.charge = charge
        self.peptide = self.peptide.upper().replace("L", "I")
        self.peptideArray = np.array(
            [0] + [AMINO_ACID_IDS[x] for x in self.peptide] + [0], dtype=np.uint16
        )
        self.modification_lists = []
        self.mod_info = Modifications()
        self.mod_info.modifications = {"ptm": {}, "sptm": {}}
        self.mod_info.add_from_ms2pip_modstrings(self.modification_lists)
        self.modpeptide = apply_mods(
            self.peptideArray, self.modifications, self.mod_info.ptm_ids
        )

    def ms2pipInput(self):
        out = np.zeros((self.maximal_length - 1, 139), dtype=np.uint16)
        real_predictions = np.array(
            ms2pip_pyx.get_vector(self.peptideArray, self.modpeptide, self.charge),
            dtype=np.uint16,
        )
        out[0 : real_predictions.shape[0], :] = real_predictions
        return out
