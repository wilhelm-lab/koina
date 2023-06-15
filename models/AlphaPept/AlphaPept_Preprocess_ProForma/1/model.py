import json
import triton_python_backend_utils as pb_utils
from lib import (
    encode_mod_features,
    strip_mod_profroma,
    character_to_array,
    get_mod_features,
)


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.seq_out_dtype = None
        self.mod_out_dtype = None

    def initialize(self, args):
        model_config = model_config = json.loads(args["model_config"])
        seq_out_config = pb_utils.get_output_config_by_name(
            model_config, "encoded_seq:0"
        )
        mod_out_config = pb_utils.get_output_config_by_name(
            model_config, "encoded_mod_feature:0"
        )
        self.seq_out_dtype = pb_utils.triton_string_to_numpy(
            seq_out_config["data_type"]
        )
        self.mod_out_dtype = pb_utils.triton_string_to_numpy(
            mod_out_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptide_in_list = [
                x[0].decode("utf-8") for x in peptide_in.as_numpy().tolist()
            ]
            sequences = character_to_array(strip_mod_profroma(peptide_in_list))

            raw_mod_features = [get_mod_features(s) for s in peptide_in_list]
            encoded_mod_features = encode_mod_features(
                mods=[x[1] for x in raw_mod_features],
                mod_sites=[x[0] for x in raw_mod_features],
                num_aa=[x[2] for x in raw_mod_features],
            )

            seq_out = pb_utils.Tensor(
                "encoded_seq:0", sequences.astype(self.seq_out_dtype)
            )
            mod_out = pb_utils.Tensor(
                "encoded_mod_feature:0",
                encoded_mod_features.astype(self.mod_out_dtype),
            )

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[seq_out, mod_out])
            )
        return responses

    def finalize(self):
        pass
