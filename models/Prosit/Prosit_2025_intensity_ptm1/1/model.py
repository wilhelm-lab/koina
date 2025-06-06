import triton_python_backend_utils as pb_utils
import numpy as np
from sequence_conversion import character_to_array, ALPHABET_MOD
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "peptides_in:0"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []

        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "modified_sequence")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            sequences = np.asarray(
                [character_to_array(seq).flatten() for seq in peptide_in_list]
            )

            t = pb_utils.Tensor("peptides_in:0", sequences.astype(self.output_dtype))

            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
