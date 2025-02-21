import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        super().__init__()

    def execute(self, requests):
        responses = []

        for request in requests:
            params = eval(request.parameters())

            max_frags = int(params["max_frags"]) if "max_frags" in params else 1000

            annotations = pb_utils.get_input_tensor_by_name(
                request, "annotations"
            ).as_numpy()

            mzs = pb_utils.get_input_tensor_by_name(request, "mz").as_numpy()

            intensities = pb_utils.get_input_tensor_by_name(
                request, "intensities"
            ).as_numpy()

            for i in range(annotations.shape[0]):
                indices = np.argsort(-intensities[i])
                annotations[i] = annotations[i][indices]
                mzs[i] = mzs[i][indices]
                intensities[i] = intensities[i][indices]
                if intensities[i][0] > 0:
                    intensities[i][intensities[i] > 0] /= intensities[i][0]

            intf = pb_utils.Tensor(
                "intensities_filtered", intensities[:, 0:max_frags]
            )
            af = pb_utils.Tensor(
                "annotations_filtered", annotations[:, 0:max_frags]
            )
            mf = pb_utils.Tensor("mz_filtered", mzs[:, 0:max_frags])
            responses.append(pb_utils.InferenceResponse(output_tensors=[intf, af, mf]))

        return responses

    def finalize(self):
        pass
