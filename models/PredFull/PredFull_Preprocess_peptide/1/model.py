import json
import numpy as np
import triton_python_backend_utils as pb_utils

Alist = list("ACDEFGHIKLMNPQRSTVWYZ")
charMap = {"*": 0, "]": len(Alist) + 1, "[": len(Alist) + 2}
for i, a in enumerate(Alist):
    charMap[a] = i + 1

ENCODING_DIMENSION = 24

mono = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032029,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.04768,
    "C": 160.03019,
    "L": 113.08406,
    "I": 113.08406,
    "D": 115.02694,
    "Q": 128.05858,
    "K": 128.09496,
    "E": 129.04259,
    "M": 131.04048,
    "m": 147.0354,
    "H": 137.05891,
    "F": 147.06441,
    "R": 156.10111,
    "Y": 163.06333,
    "N": 114.04293,
    "W": 186.07931,
    "O": 147.03538,
}


def getmod(pep):
    mod = np.zeros(len(pep))

    if pep.isalpha():
        return pep, mod

    seq = []

    i = -1
    while len(pep) > 0:
        if pep[0] == "[":
            if pep[8:11] == "35]":
                mod[i] = 1
                pep = pep[11:]
            else:  # not oxM
                mod[i] = -1
                return pep, mod
        else:
            seq += pep[0]
            pep = pep[1:]
            i = len(seq) - 1

    return "".join(seq), mod[: len(seq)]


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "input")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            sequences = []
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            pep_dimension = 30
            for seq in peptide_in_list:  # can make this more efficient
                seq, mod = getmod(seq)

                if len(seq) + 2 > pep_dimension:
                    pep_dimension = len(seq) + 2

            for seq in peptide_in_list:
                # check for modifications
                seq, mod = getmod(seq)

                embedding = np.zeros([pep_dimension, 29], dtype="float32")

                if np.any(mod == -1):
                    raise RuntimeError("Only Oxidation modification is supported")

                # process base peptide sequence
                seq = seq.replace("L", "I")
                embedding[len(seq)][ENCODING_DIMENSION - 1] = 1  # ending pos
                for i, aa in enumerate(seq):
                    embedding[i][charMap[aa]] = 1
                    embedding[i][ENCODING_DIMENSION] = (
                        mono[aa] / 200
                    )  # mass of AA divided by 200
                embedding[: len(seq), ENCODING_DIMENSION + 1] = (
                    np.arange(len(seq)) / 1000
                )  # position info #position divided by 1000
                embedding[len(seq) + 1, 0] = 1  # padding info

                # still need to encode mod
                for i, modi in enumerate(mod):
                    embedding[i][ENCODING_DIMENSION + 2 + int(modi)] = 1

                sequences.append(embedding)

            sequences = np.array(sequences)

            t = pb_utils.Tensor("input", sequences.astype(self.output_dtype))

            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
