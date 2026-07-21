import triton_python_backend_utils as pb_utils
import numpy as np
from modifications import ProformaParser, Unimod
import re
import json


dict_index_pos = {"H": 0, "C": 1, "N": 2, "O": 3, "P": 4, "S": 5}

dict_ptm_atom_count_loss = {
    #        H   C   N   O   P   S
    "C_4": "H S",  # C(cam)   + H(4) C(2) N O
    "C_1263": "H S",
    "S_21": "H O",
    "T_21": "H O",
    "Y_21": "H O",
    "H_21": "H N",
    "P_21": "H O",
    "M_35": "H(3) C S",
    "W_35": "H",
    "C_35": "H S",
    "K_35": "H(2) N",
    "P_35": "H(4) C(2) N",
    "H_35": "H N",
    
    "Q_34": "H(2) N",

    "K_34": "H(2) N",
    "R_34": "H(2) N",
    "C_34": "H S",
    "H_34": "H N",
    "D_34": "H O(2)",
    "E_34": "H O(2)",
    "L_34": "H(2) N",
    "I_34": "H(2) N",
    "N_34": "H(2) C N O",
    "Q_7": "H(2) N",
    "N_7": "H(2) N",
    "K_4":"H O",
    "C_312": "H S",
    "R_7": "H N",
    "K_737": "H(2) N",
    "_737": "H(2) N",
    "_1": "H(2) N",
    "K_1": "H(2) N",
    "K_121": "H(2) N",
    "E_27": "H(9) C(4) N(2) O",
    "Q_28": "H(9) C(4) N(2) O",
    "K_1848": "H(2) N",
    "K_535": "H(2) N",
    "K_37": "H(2) N",
    "K_1293": "H(2) N",
    "K_1990": "H(2) N",
    "K_36": "H(2) N",
    "R_36": "H(2) N",
    "S_43": "H O",
    "T_43": "H O",
    "C_2062": "H S",  
    "K_2016": "H(2) N",  
    "_2016": "H(2) N",   
    "K_214": "H(2) N",
    "_214": "H(2) N",
    "K_730": "H(2) N",
    "_730": "H(2) N",
    
    "K_58": "H(2) N",
    "_58": "H(2) N",
    "K_59": "H(2) N",
    "_59": "H(2) N",
    "_411": "H(2) N",
    "K_1289": "H(2) N",
    "K_5634": "H(2) N", #Acetyl_label+monomethyl
    "K_56": "H(2) N",
    "K_12118":"H(2) N",
    "K_129317":"H(2) N",
    "K_19903":"H(2) N",
    "K_1263": "H(2) N",
    "K_4": "H(2) N",
    "R_267": ""

}


unimod = Unimod()


def atom_count_str_list(atom_count, atom_count_list):
    atom_count = atom_count
    if atom_count == "":
        return atom_count_list
    atom_count = atom_count.split(" ")
    for atoms in atom_count:
        m = re.search(r"([H|C|N|O|P|S])\(?(\d*)\)?", atoms)
        atom = m.group(1)
        count = m.group(2)
        if count != "":
            atom_count_list[dict_index_pos[atom]] += int(count)
        else:
            atom_count_list[dict_index_pos[atom]] += 1
    return atom_count_list


def get_ac(seq, logger):
    seq = unimod.lookup_sequence_m(
        ProformaParser.parse_sequence(seq), keys_to_lookup=["record_id"]
    )
    aa_ac_placeholder = np.ones([32, 6])
    aa_ac_list = []
    for aa in seq:
        current_ac = [1, 1, 1, 1, 1, 1]
        if aa[1] != "-" and aa[1] != "":
            current_ac = atom_count_str_list(
                dict_ptm_atom_count_loss[aa[0] + "_" + aa[1][1:-1]], current_ac
            )
        aa_ac_list.append(current_ac)
    aa_ac_placeholder[: len(aa_ac_list),] = aa_ac_list
    return aa_ac_placeholder


def get_ac_all(sequences, logger):
    aa_ac = [get_ac(seq, logger) for seq in sequences]
    return aa_ac


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "ac_loss"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        logger = pb_utils.Logger
        peptide_in_str = []
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            fill = np.array(get_ac_all(peptide_in_list, logger))
            t = pb_utils.Tensor("ac_loss", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
