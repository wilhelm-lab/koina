import triton_python_backend_utils as pb_utils
import numpy as np
import pandas as pd
import json


def calc_ccs_feats(peptide_in_list):
    ccs_feat = {"H_count": [], "FWY_count": [], "DE_count": [], "KR_count": []}
    for seq in peptide_in_list:
        seq_len = len(seq)
        ccs_feat["H_count"].append((seq.count("H")) / float(seq_len))
        ccs_feat["FWY_count"].append(
            (seq.count("F") + seq.count("W") + seq.count("Y")) / float(seq_len)
        )
        ccs_feat["DE_count"].append((seq.count("D") + seq.count("E")) / float(seq_len))
        ccs_feat["KR_count"].append((seq.count("K") + seq.count("R")) / float(seq_len))

    return pd.DataFrame(ccs_feat).values


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "ccs_feat"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(
                request, "stripped_peptide"
            ).as_numpy()
            peptide_in_list = [x[0].decode("utf-8") for x in peptide_in]

            arr = calc_ccs_feats(peptide_in_list)
            t = pb_utils.Tensor("ccs_feat", arr.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
