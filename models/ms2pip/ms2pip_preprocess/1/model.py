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
        
        
        logger = pb_utils.Logger
        ## every request is up to abatch_size
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "proforma")
            peptides_ = peptide_in.as_numpy().tolist()
            list_ms2pip_input = []
            for peptide in peptides_:
                peptide_in_list = peptide[0].decode("utf-8")
                logger.log_info(peptide_in_list)
                ms2 = MinimalMS2PIP(peptide_in_list)
                inter = ms2.ms2pipInput()
                logger.log_info(f"inter.shape: {inter.shape}")
                list_ms2pip_input.append(inter)
                
            more_fun = np.vstack(list_ms2pip_input)
            
            logger.log_info(f"more_fun.shape {more_fun.shape}")

            output_tensors = []
            output_tensors.append(
                pb_utils.Tensor("xgboost_input", more_fun.astype(self.output_dtype))
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        return responses

    def finalize(self):
        print("Cleaning up")
