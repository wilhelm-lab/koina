import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "pos_ac")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            ac_in = pb_utils.get_input_tensor_by_name(request, "single_ac")
            single_ac = ac_in.as_numpy()
            pep_lengths = pb_utils.get_input_tensor_by_name(request, "peptide_length")
            pep_lengths = pep_lengths.as_numpy()
            pos_ac = []
            for peptide_ac, pep_length in zip(single_ac, pep_lengths):
                pep_length = int(pep_length)
                first_four = peptide_ac[:4]
                last_four = peptide_ac[pep_length - 4 : pep_length]
                first_four = np.asarray(first_four).flatten()
                last_four = np.asarray(last_four).flatten()
                pos_ac.append(np.concatenate([first_four, last_four]))
            pos_ac = np.array(pos_ac)
            t = pb_utils.Tensor("pos_ac", pos_ac.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass