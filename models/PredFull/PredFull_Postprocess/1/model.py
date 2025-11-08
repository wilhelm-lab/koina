import json
import numpy as np
import triton_python_backend_utils as pb_utils
import math


def sparse(
    x, y, th=0.001
):  # original python version had th = 0.0002. That feel s abit unnecessary
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y, dtype="float32")

    y /= np.max(y)

    return x[y > th], y[y > th]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        all_mzs = []
        all_ints = []
        for request in requests:
            # mtx = pb_utils.get_input_tensor_by_name(request, "spectrum").as_numpy()
            # np.save("mtx.npy", mtx)
            spectra = pb_utils.get_input_tensor_by_name(request, "spectrum").as_numpy()
            precursor_mass_with_oxm = pb_utils.get_input_tensor_by_name(
                request, "precursor_mass_with_oxM"
            ).as_numpy()

            # format this for whole batch. May require not prematurely cutting spectra short based on precursor mass
            for i, mass in enumerate(precursor_mass_with_oxm):
                spectra[i, min(math.ceil(mass / 0.1), spectra.shape[1]) :] = 0
            spectra = np.square(spectra)

            final_shape1 = 0
            for i in range(spectra.shape[0]):
                spectrum = spectra[i, :]
                imz = np.arange(0, spectra.shape[1], dtype="int32") * 0.1
                mzs, its = sparse(imz, spectrum)
                all_mzs.append(mzs)
                all_ints.append(its)
                if its.size > final_shape1:
                    final_shape1 = its.size

            for i, (mzs, ints) in enumerate(zip(all_mzs, all_ints)):
                minus_int_array = np.full(final_shape1 - mzs.size, -0.001)
                minus_mz_array = np.full(final_shape1 - mzs.size, -1)
                all_mzs[i] = np.append(mzs, minus_mz_array)
                all_ints[i] = np.append(ints, minus_int_array)

            all_mzs = np.array(all_mzs)
            all_ints = np.array(all_ints)

            all_ints *= 1000
            all_mzs = np.round(all_mzs, 2)
            all_ints = np.round(all_ints, 4)

            output_tensors = [
                pb_utils.Tensor("mzs", all_mzs.astype(self.output_dtype)),
                pb_utils.Tensor("intensities", all_ints.astype(self.output_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses
