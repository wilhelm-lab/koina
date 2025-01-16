import triton_python_backend_utils as pb_utils
import numpy as np
import json

ftypes = {"UN": 0, "CID": 1, "ETD": 2, "HCD": 3, "ETHCD": 4, "ETCID": 5}


def map_fragtypes(data):
    datanum = ftypes[data]
    targets = np.array([datanum])
    return np.int_((np.eye(30)[targets])).tolist()[0]


def to_one_hot(fragtypes):
    array = [map_fragtypes(x) for x in fragtypes]
    return np.array(array, dtype=float)


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "fragmentation_types_encoding"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            fragmentation_types = pb_utils.get_input_tensor_by_name(
                request, "fragmentation_types"
            )
            fragmentation_types = sum(
                np.char.upper(fragmentation_types.as_numpy().astype(str)).tolist(), []
            )
            fragmentation_types_encoding = to_one_hot(fragmentation_types)

            t = pb_utils.Tensor(
                "fragmentation_types_encoding",
                fragmentation_types_encoding.astype(self.output_dtype),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
