import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "norm_intensities"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        logger = pb_utils.Logger
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "raw_intensities")
            peptides = 2**peptide_in.as_numpy()-0.001
            logger.log_info(f"peptides.shape {peptides.shape}")
            peptides[peptides < 0] = 0
            peptides = peptides.reshape((-1,29))
            logger.log_info(f"peptides.shape {peptides.shape}")
            output_tensors = [pb_utils.Tensor("norm_intensities", peptides.astype(self.output_dtype))]
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses

    def finalize(self):
        print("Cleaning up")
