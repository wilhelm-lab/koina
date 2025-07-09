import triton_python_backend_utils as pb_utils
import numpy as np
import time


class TritonPythonModel:
    def initialize(self, args):
        super().__init__()

    def execute(self, requests):
        responses = []
        for request in requests:
            params = eval(request.parameters())

            max_frags = int(params["max_frags"]) if "max_frags" in params else 200

            annotations = pb_utils.get_input_tensor_by_name(
                request, "annotations"
            ).as_numpy()

            mzs = pb_utils.get_input_tensor_by_name(request, "mz").as_numpy()

            coefficients = pb_utils.get_input_tensor_by_name(
                request, "coefficients"
            ).as_numpy()

            knots = pb_utils.get_input_tensor_by_name(request, "knots").as_numpy()

            AUCs = pb_utils.get_input_tensor_by_name(request, "AUC").as_numpy()

            for i in range(annotations.shape[0]):
                indices = np.argsort(-AUCs[i])
                annotations[i] = annotations[i][indices]
                coefficients[i] = coefficients[i][:, indices]
                mzs[i] = mzs[i][indices]

            cf = pb_utils.Tensor(
                "coefficients_filtered", coefficients[:, :, 0:max_frags]
            )
            kf = pb_utils.Tensor("knots_filtered", knots[0])
            af = pb_utils.Tensor("annotations_filtered", annotations[:, 0:max_frags])
            mf = pb_utils.Tensor("mz_filtered", mzs[:, 0:max_frags])
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[cf, kf, af, mf])
            )
            
        return responses

    def finalize(self):
        pass
