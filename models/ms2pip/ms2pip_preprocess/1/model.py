import triton_python_backend_utils as pb_utils
import numpy as np
from psm_utils import Peptidoform, PSM, PSMList
import json
from tobi import MinimalMS2PIP


class TritonPythonModel:
    def initialize(self,args):
        print("Preprocessing of the Peptide_input")
        peptidoform = Peptidoform("ACDEK/2")
        print(peptidoform.theoretical_mass)
        print("Preprocessing of the Peptide_input")
        self.model_config = model_config = json.loads(args['model_config'])
        print(self.model_config)
        print("--------------")
        output0_config = pb_utils.get_output_config_by_name(
                                          self.model_config, "xgboost_input")
        print(output0_config)
        print("preprocess_peptide type: " + str(output0_config))
        self.output_dtype = pb_utils.triton_string_to_numpy(
                                    output0_config['data_type'])

    def execute(self, requests):
        responses = []
        t = []
        list_ms2pip_input = []
        logger = pb_utils.Logger
        ## every request is up to abatch_size
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "proforma")
            peptides_ = peptide_in.as_numpy().tolist()
            for peptide in peptides_:
                peptide_in_list = peptide[0].decode('utf-8')
                logger.log_info(peptide_in_list)
            # peptide_in_list = [x[0].decode('utf-8')  for x in peptides_ ]
            # peptidoform = Peptidoform("ACDEK/2")
            # t = pb_utils.Tensor("out",sequences.astype(self.output_dtype) )
            # inter = np.array([str(peptidoform.theoretical_mass)])
            # inter = np.zeros((1,139), dtype=np.float32)
                ms2 = MinimalMS2PIP(peptide_in_list)
                inter = ms2.ms2pipInput()
                logger.log_info(str(inter.shape[0]))
                list_ms2pip_input.append(inter)
                # logger.log_info(inter.shape)
            # inter = np.array( [str("halloTobi")])
            
            #            logger.log_info("shape of t[0] " +str(t[0].shape))
            more_fun = np.vstack(list_ms2pip_input)
            t.append(  pb_utils.Tensor("xgboost_input", more_fun.astype(self.output_dtype) ))
            responses.append(pb_utils.InferenceResponse(output_tensors=t))
        return responses

    def finalize(self):
        print('Cleaning up')
