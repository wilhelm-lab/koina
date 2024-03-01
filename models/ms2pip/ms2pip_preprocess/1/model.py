import triton_python_backend_utils as pb_utils
import numpy as np
from psm_utils import Peptidoform, PSM, PSMList
import json
from tobi import MinimalMS2PIP


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "xgboost_input"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        ## every request is up to abatch_size
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "proforma")
            peptides_ = peptide_in.as_numpy().flatten().tolist()

            charge_in = pb_utils.get_input_tensor_by_name(request, "charge_in")
            charge_ = charge_in.as_numpy().flatten().tolist()

            list_ms2pip_input = []
            for peptide, charge in zip(peptides_, charge_):
                peptide_in_list = peptide.decode("utf-8")
                ms2 = MinimalMS2PIP(peptide_in_list, charge)
                inter = ms2.ms2pipInput()
                list_ms2pip_input.append(inter)

            more_fun = np.vstack(list_ms2pip_input)

            output_tensors = []
            output_tensors.append(
                pb_utils.Tensor("xgboost_input", more_fun.astype(self.output_dtype))
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses

    def finalize(self):
        pass
