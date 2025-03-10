import triton_python_backend_utils as pb_utils
import numpy as np
from modifications import ProformaParser, Unimod
import re
import json

dict_index_pos = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4, "P": 5}

aa_count_dict = {
    "A": [1, 2, 0, 0, 0, 0],
    "R": [4, 9, 3, 0, 0, 0],
    "N": [2, 3, 1, 1, 0, 0],
    "D": [2, 2, 0, 2, 0, 0],
    "C": [1, 2, 0, 0, 1, 0],
    "Q": [3, 5, 1, 1, 0, 0],
    "E": [3, 4, 0, 2, 0, 0],
    "G": [0, 0, 0, 0, 0, 0],
    "H": [4, 4, 2, 0, 0, 0],
    "I": [4, 8, 0, 0, 0, 0],
    "L": [4, 8, 0, 0, 0, 0],
    "K": [4, 9, 1, 0, 0, 0],
    "M": [3, 6, 0, 0, 1, 0],
    "F": [7, 6, 0, 0, 0, 0],
    "P": [3, 4, 0, 0, 0, 0],
    "S": [1, 2, 0, 1, 0, 0],
    "T": [2, 4, 0, 1, 0, 0],
    "W": [9, 7, 1, 0, 0, 0],
    "Y": [7, 6, 0, 1, 0, 0],
    "V": [3, 6, 0, 0, 0, 0],
    "X": [0, 0, 0, 0, 0, 0],
    "U": [0, 0, 0, 0, 0, 0],
    "B": [0, 0, 0, 0, 0, 0],
    "J": [0, 0, 0, 0, 0, 0],
    "O": [0, 0, 0, 0, 0, 0],
    "Z": [0, 0, 0, 0, 0, 0],
}

unimod = Unimod()


def atom_count_str_list(atom_count, atom_count_list):
    atom_count = atom_count[1:-1]
    atom_count = atom_count.split(" ")
    for atoms in atom_count:
        count = re.findall(r"\(\d+\)", atoms)[1:-1]
        atom_key = re.findall("|".join(dict_index_pos.keys()), atoms)[0]
        if len(count) > 0:
            atom_count_list[dict_index_pos[atom_key]] += int(count[0])
        else:
            atom_count_list[dict_index_pos[atom_key]] += 1
    return atom_count_list


def get_ac(seq):
    seq = unimod.lookup_sequence_m(
        ProformaParser.parse_sequence(seq), keys_to_lookup=["delta_composition"]
    )[1:-1]
    aa_ac_placeholder = np.zeros([60, 6])
    aa_ac_list = []
    for aa in seq:
        current_ac = aa_count_dict[aa[0]].copy()
        if aa[1] != "-":
            current_ac = atom_count_str_list(aa[1], current_ac)
        aa_ac_list.append(current_ac)
    aa_ac_placeholder[: len(aa_ac_list),] = aa_ac_list
    return aa_ac_placeholder


def get_ac_all(sequences):
    aa_ac = [get_ac(seq) for seq in sequences]
    return aa_ac


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "single_ac"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            fill = np.array(get_ac_all(peptide_in_list))
            t = pb_utils.Tensor("single_ac", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
