import triton_python_backend_utils as pb_utils
import numpy as np
import json
from pyteomics import mass


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
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "precursor_mass"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]
            masses = []
            masses_with_oxm = []  # for postprocessing

            for seq in peptide_in_list:
                seq, mod = getmod(seq)  # move getmod as separate python file
                if np.any(mod == -1):
                    raise RuntimeError("Only Oxidation modification is supported")

                base = mass.fast_mass(seq, ion_type="M", charge=1)
                base += 57.021 * seq.count("C")
                base_with_oxm = base + 15.9949 * sum(mod)
                base /= 20000.0
                masses.append(base)
                masses_with_oxm.append(base_with_oxm)

            masses = np.array(masses)
            # masses = np.reshape(masses, (1, 1))  # testing
            masses_with_oxm = np.array(masses_with_oxm)
            # masses_with_oxm = np.reshape(masses_with_oxm, (1, 1))  # testing
            t = pb_utils.Tensor(
                "precursor_mass",
                masses.astype(self.output_dtype),
            )
            t2 = pb_utils.Tensor(
                "precursor_mass_with_oxM",
                masses_with_oxm.astype(self.output_dtype),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t, t2]))
        return responses

    def finalize(self):
        pass
