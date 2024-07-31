import triton_python_backend_utils as pb_utils
import numpy as np
import json
import re



def find_crosslinker_position(peptide_sequence: str):
    peptide_sequence = re.sub(r"\[UNIMOD:(?!1898).*?\]", "", peptide_sequence)
    crosslinker_position = re.search(r"K(?=\[UNIMOD:1898\])", peptide_sequence)
    crosslinker_position = crosslinker_position.start() + 1
    return crosslinker_position

def gen_annotation_linear_pep(unmod_seq: str, precursor_charge: int):
    peptide_length = len(unmod_seq)
    ions = ["y", "b"]
    positions = [x for x in range(1, 30)]  # Generate positions 1 through 29
    annotation = []
    max_charge = 3  # Always consider three charges
    
    for pos in positions:
        for ion in ions:
            for charge in range(1, max_charge + 1):
                if charge > precursor_charge or pos >= peptide_length:
                    annotation.append(None)  # Append None for invalid charges and positions beyond the sequence length
                else:
                    annotation.append(f"{ion}{pos}+{charge}")
                    
    return annotation

def gen_annotation_xl(annotation, unmod_seq: str, crosslinker_position: int):
    peptide_length = len(unmod_seq)

    # Annotate b-ions with crosslink from crosslinker position onward
    for i in range(crosslinker_position, peptide_length):
        for charge in ["1", "2", "3"]:
            b_ion = f"b{i}+{charge}"
            if b_ion in annotation:
                index = annotation.index(b_ion)
                annotation[index] = f"b_xl{i}+{charge}"

     # Correcting y-ion annotation to move backward from the end
    for i in range(peptide_length, 1, -1):
        for charge in ["1", "2", "3"]:
            y_ion = f"y{i}+{charge}"
            if y_ion in annotation:
                index = annotation.index(y_ion)
                
                # Only annotate with '_xl' if within the crosslinker adjusted position
                if i >= peptide_length - crosslinker_position + 1:
                    annotation[index] = f"y_xl{i}+{charge}"
                else:
                    annotation[index] = y_ion  # this line could be omitted if y_ions need no re-assignment

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
        annotation = np.empty((0, 174))

        for request in requests:
            batchsize = pb_utils.get_input_tensor_by_name(request, "precursor_charges").as_numpy().shape[0]
            precursor_charges = pb_utils.get_input_tensor_by_name(request, "precursor_charges").as_numpy()

            peptide_sequences_1 = pb_utils.get_input_tensor_by_name(
                request, "peptide_sequences_1"
            ).as_numpy()
            
            for i in range(batchsize):
                regular_sequence = peptide_sequences_1[i][0].decode("utf-8")
                crosslinker_position = find_crosslinker_position(regular_sequence)
                precursor_charge = int(precursor_charges[i][0])
                unmod_seq = re.sub(r"\[.*?\]", "", regular_sequence)
                annotation_lin = gen_annotation_linear_pep(unmod_seq, precursor_charge)
                annotation_i = gen_annotation_xl(annotation_lin, unmod_seq, crosslinker_position)
                annotation = np.vstack((annotation, annotation_i))

            t = pb_utils.Tensor("annotation", annotation)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass