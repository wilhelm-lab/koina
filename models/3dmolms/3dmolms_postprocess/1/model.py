import json
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = {}

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "int")
        self.output_dtype["int"] = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        output0_config = pb_utils.get_output_config_by_name(model_config, "mz")
        self.output_dtype["mz"] = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "3dmolms_out").as_numpy()
            norm = (raw / raw.max(axis=1)) ** 2
            mz = np.arange(0, norm.shape[1] * 0.2, 0.2)

            int_tensor = pb_utils.Tensor("int", norm.astype(self.output_dtype["int"]))
            mz_tensor = pb_utils.Tensor("mz", mz.astype(self.output_dtype["mz"]))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[int_tensor, mz_tensor])
            )
        return responses

    def finalize(self):
        pass
