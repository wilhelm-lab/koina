import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "general_features"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            peptide_length = pb_utils.get_input_tensor_by_name(
                request, "peptide_length"
            ).as_numpy()
            peptide_pos_ac = pb_utils.get_input_tensor_by_name(
                request, "pos_ac"
            ).as_numpy()
            peptide_ac = pb_utils.get_input_tensor_by_name(request, "sum_ac").as_numpy()
            ccs_feat = pb_utils.get_input_tensor_by_name(request, "ccs_feat").as_numpy()
            precursor_charges = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges"
            ).as_numpy()

            general_features = np.hstack(
                [
                    peptide_ac,
                    peptide_length,
                    ccs_feat,
                    precursor_charges,
                    peptide_pos_ac,
                ]
            )

            t = pb_utils.Tensor(
                "general_features", general_features.astype(self.output_dtype)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass
