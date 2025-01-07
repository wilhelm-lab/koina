import json
import triton_python_backend_utils as pb_utils
import numpy as np
from sequence_conversion import peptide_to_array


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.seq_out_dtype = None

    def initialize(self, args):
        model_config = model_config = json.loads(args["model_config"])
        seq_out_config = pb_utils.get_output_config_by_name(
            model_config, "encoded_seq:0"
        )
        self.seq_out_dtype = pb_utils.triton_string_to_numpy(
            seq_out_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptide_in_list = [
                x[0].decode("utf-8") for x in peptide_in.as_numpy().tolist()
            ]

            sequences = np.asarray(
                [peptide_to_array(seq).flatten() for seq in peptide_in_list], "int64"
            )

            seq_out = pb_utils.Tensor(
                "encoded_seq:0", sequences.astype(self.seq_out_dtype)
            )

            responses.append(pb_utils.InferenceResponse(output_tensors=[seq_out]))
        return responses

    def finalize(self):
        pass
