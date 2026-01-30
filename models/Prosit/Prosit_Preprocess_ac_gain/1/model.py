import triton_python_backend_utils as pb_utils
import numpy as np
from modifications import ProformaParser, Unimod
import re
import json


dict_index_pos = {"H": 0, "C": 1, "N": 2, "O": 3, "P": 4, "S": 5}

dict_ptm_atom_count_gain = {
    #        H   C   N   O   P   S
    "C_4": "H(4) C(2) N O S",  # C(cam)   + H(4) C(2) N O
    "C_1263": "H(4) C(2) N O S",
    "S_21": "H(2) O(4) P",
    "T_21": "H(2) O(4) P",
    "Y_21": "H(2) O(4) P",
    "H_21": "H(2) O(3) N P",
    "P_21": "H(2) O(4) P",
    "M_35": "H(3) C O S",
    "W_35": "H O",
    "C_35": "H O S",
    "K_35": "H(2) N O",
    "P_35": "H(4) C(2) N O",
    "H_35": "H N O",
    
    "Q_34": "H(4) C N",

    "K_34": "H(4) C N",
    "R_34": "H(4) C N",
    "C_34": "H(3) C S",
    "H_34": "H(3) C N",
    "D_34": "H(3) C O(2)",
    "E_34": "H(3) C O(2)",
    "L_34": "H(4) C N",
    "I_34": "H(4) C N",
    "N_34": "H(4) C(2) N O",
    "K_4": "H(5) C(2) N(2) O",
    "Q_7": "H O",
    "N_7": "H O",
    "K_4": "H(4) C(2) N O(2)",
    "R_267": "",


    "C_312": "H(6) C(3) N O(2) S(2)",
    "R_7": "O",
    "K_737": "H(22) C(12) N(3) O(2)", 	
    "_737": "H(22) C(12) N(3) O(2)",
    "_1": "H(4) C(2) N O",
    "K_1": "H(4) C(2) N O",

    "K_121": "H(8) C(4) N(3) O(2)",
    "E_27": "H(7) C(4) N(2)",
    "Q_28": "H(6) C(4) N O",
    "K_1848": "H(8) C(5) N O(3)",
    "K_535": "H(31) C(16) N(8) O(4)	",
    "K_37": "H(8) C(3) N",
    "K_1293": "H(23) C(13) N(6) O(6)",
    "K_1990": "H(24) C(13) N(5) O(4) S",
    "K_36": "H(6) C(2) N",
    "R_36": "H(6) C(2) N",
    "S_43": "H(14) C(8) N O(6)",
    "T_43": "H(14) C(8) N O(6)",
    "C_2062": "H(25) C(14) N(4) O(3) S",  

    "K_2016": "H(27) C(15) N(4) O(3)",  
    "_2016": "H(27) C(15) N(4) O(3)",   
    
    "K_214": "H(14) C(7) N(3) O",
    "_214": "H(14) C(7) N(3) O",
    "K_730": "H(26) C(14) N(5) O(3)",
    "_730": "H(26) C(14) N(5) O(3)",


    "K_58": "H(6) C(3) N O",
    "_58": "H(6) C(3) N O",
    "K_59": "H(6) C(3) N O",
    "_59": "H(6) C(3) N O",
    "_411": "H(6) C(3) N O",
    "K_1289": "H(8) C(4) N O",
    "K_5634": "H(6) C(3) N O",
    "K_56": "H(4) C(2) N O",
    "K_12118":"H(20) C(10) N(7) O(3)",
    "K_129317":"H(15) C(8) N(4) O(4)",
    "K_19903":"H(19) C(10) N(4) O(3)",
    "K_1263":"H(5) C(2) N(2) O",
    "K_4":"H(5) C(2) N(2) O"

}



unimod = Unimod()


def atom_count_str_list(atom_count, atom_count_list):
    atom_count = atom_count
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
                dict_ptm_atom_count_gain[aa[0] + "_" + aa[1][1:-1]], current_ac
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
            self.model_config, "ac_gain"
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
            t = pb_utils.Tensor("ac_gain", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
