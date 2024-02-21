import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from postprocess import create_masking, apply_masking
from pyteomics import proforma


def find_crosslinker_position(peptide_sequence: str):
    peptide_sequence = re.sub(r"\[UNIMOD:(?!1896|1884\]).*?\]", "", peptide_sequence)
    crosslinker_position = re.search(r"K(?=\[UNIMOD:(?:1896|1884)\])", peptide_sequence)
    crosslinker_position = crosslinker_position.start() + 1
    return crosslinker_position


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        logger = pb_utils.Logger
        responses = []
        for request in requests:
            peptide_in = (
                pb_utils.get_input_tensor_by_name(request, "peptides_in_1:0")
                .as_numpy()
                .tolist()
            )

            peptide_in_numpy = pb_utils.get_input_tensor_by_name(
                request, "peptides_in_1:0"
            ).as_numpy()

            crosslinker_position = []
            for i in range(len(peptide_in_numpy)):
                regular_sequence = peptide_in_numpy[i][0].decode("utf-8")
                crosslinker_position.append(find_crosslinker_position(regular_sequence))

            peptide_in = [x[0].decode("utf-8") for x in peptide_in]
            unmod_seq = [x for x in internal_without_mods(peptide_in)]

            peaks_in = pb_utils.get_input_tensor_by_name(
                request, "peaks_in:0"
            ).as_numpy()

            mask = create_masking(unmod_seq, crosslinker_position)
            masked_peaks = apply_masking(peaks_in, mask)

            # considr mass of short and long part of crosslinker
            peptide_in_s = []
            peptide_in_l = []
            for pep in peptide_in:
                if "K[UNIMOD:1896]" in pep:
                    pep_s = pep.replace("K[UNIMOD:1896]", "K[UNIMOD:1881]")
                    pep_l = pep.replace("K[UNIMOD:1896]", "K[UNIMOD:1882]")
                    peptide_in_s.append(pep_s)
                    peptide_in_l.append(pep_l)
                else:
                    pep_s = pep.replace("K[UNIMOD:1884]", "K[UNIMOD:1886]")
                    pep_l = pep.replace("K[UNIMOD:1884]", "K[UNIMOD:1885]")
                    peptide_in_s.append(pep_s)
                    peptide_in_l.append(pep_l)

            fragmentmz_s = self.get_fragments(peptide_in_s)
            fragmentmz_l = self.get_fragments(peptide_in_l)
            fragmentmz = np.concatenate((fragmentmz_s, fragmentmz_l), axis=1)
            fragmentmz[np.isnan(masked_peaks)] = -1
            masked_peaks[np.isnan(masked_peaks)] = -1

            output_tensors = [
                pb_utils.Tensor("intensities", masked_peaks.astype(self.output_dtype)),
                pb_utils.Tensor("mz", fragmentmz.astype(self.output_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def get_fragments(self, sequences):
        tensor_inputs = [
            pb_utils.Tensor("ProForma", np.array(sequences, dtype=np.object_)),
            pb_utils.Tensor("charges", np.array([1, 2, 3], dtype=np.int32)),
            pb_utils.Tensor("ion_series", np.array(["y", "b"], dtype=np.object_)),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="fragment_mz",
            requested_output_names=["output_fragmentmz"],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = np.zeros((len(sequences), 174), dtype=np.float128)

            tmp = pb_utils.get_output_tensor_by_name(
                resp, "output_fragmentmz"
            ).as_numpy()

            output[:, 0::6] = tmp[:, 0, 0, :29]  # y charge 1
            output[:, 1::6] = tmp[:, 0, 1, :29]  # y charge 2
            output[:, 2::6] = tmp[:, 0, 2, :29]  # y charge 3

            output[:, 3::6] = tmp[:, 1, 0, :29]  # b charge 1
            output[:, 4::6] = tmp[:, 1, 1, :29]  # b charge 2
            output[:, 5::6] = tmp[:, 1, 2, :29]  # b charge 3

            return output
