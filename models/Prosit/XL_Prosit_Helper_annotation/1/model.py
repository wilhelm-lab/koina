import triton_python_backend_utils as pb_utils
from spectrum_fundamentals.annotation.annotation import peak_pos_xl_cms2
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import re


def find_crosslinker_position(peptide_sequences_1: str):
    pattern = r"K\[UNIMOD:\d+\]"
    match = re.search(pattern, peptide_sequences_1)
    if match:
        crosslinker_position = match.start() + 1
        return crosslinker_position
    else:
        return None

def gen_annotation_linear_pep():
    ions = [
        "y",
        "b",    
    ]
    charges = ["1", "2", "3"]
    positions = [x for x in range(1, 30)]
    annotation = []
    for pos in positions:
        for ion in ions:
            for charge in charges:
                annotation.append(ion + str(pos) + "+" + charge)
    return annotation

def gen_annotation_xl(crosslinker_position: int):
    annotations = gen_annotation_linear_pep()
    annotation = np.concatenate((annotations, annotations))
    annotation = annotation.tolist()
    peaks_range, peaks_y, peaks_b, peaks_yshort, peaks_bshort, peaks_ylong, peaks_blong = peak_pos_xl_cms2("K" * 30, crosslinker_position)
    for pos in peaks_yshort:
         annotation[pos] = "y_short" + annotation[pos][1:]
    for pos in peaks_bshort:
         annotation[pos] = "b_short" + annotation[pos][1:]
    for pos in peaks_ylong:
         annotation[pos] = "y_long" + annotation[pos][1:]
    for pos in peaks_blong:
         annotation[pos] = "b_long" + annotation[pos][1:]
    pos_none = [num + 174 for num in peaks_y] + [num + 174 for num in peaks_b]
    for pos in pos_none:
         annotation[pos] = "None"
    return np.array(annotation).astype(np.object_)
 
class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "annotation"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            batchsize = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charges")
                .as_numpy()
                .shape[0]
            )
            unmod_seq_pep_1 = pb_utils.get_input_tensor_by_name(request, "peptides_in_1_str:0")
            crosslinker_position = find_crosslinker_position(unmod_seq_pep_1)
            annotation = np.tile(gen_annotation_xl(crosslinker_position), batchsize).reshape((-1, 348))
            t = pb_utils.Tensor("annotation", annotation)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass