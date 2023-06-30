import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            r_int_y = pb_utils.get_input_tensor_by_name(
                request, "raw_intensities_y"
            ).as_numpy()
            r_int_b = pb_utils.get_input_tensor_by_name(
                request, "raw_intensities_b"
            ).as_numpy()

            tmp = np.full(r_int_y.shape, np.nan)
            # flip y ions so they are in ascending order making annotation easier
            tmp[np.flip(~np.isnan(r_int_y), 0)] = np.flip(r_int_y[~np.isnan(r_int_y)])
            r_int_y = np.flip(tmp, 0)

            int_out = np.hstack([r_int_b, r_int_y])

            int_out = int_out / np.nansum(int_out, 1).reshape(-1, 1)
            int_out[np.isnan(int_out)] = -1

            output_tensors = [
                pb_utils.Tensor("intensities", int_out.astype(self.output_dtype))
            ]
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses

    def finalize(self):
        pass
