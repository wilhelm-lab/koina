import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils


def internal_without_mods(sequences):
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq.decode("utf-8")) for seq in sequences]


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = model_config = json.loads(args["model_config"])
        seq_out_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        mod_out_config = pb_utils.get_output_config_by_name(model_config, "mz")
        anno_config = pb_utils.get_output_config_by_name(model_config, "annotation")
        self.seq_out_dtype = pb_utils.triton_string_to_numpy(
            seq_out_config["data_type"]
        )
        self.mod_out_dtype = pb_utils.triton_string_to_numpy(
            mod_out_config["data_type"]
        )
        self.anno_out_dtype = pb_utils.triton_string_to_numpy(anno_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = (
                pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
                .as_numpy()
                .flatten()
            )
            charge_in = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charges")
                .as_numpy()
                .flatten()
            )
            ce_in = (
                pb_utils.get_input_tensor_by_name(request, "collision_energies")
                .as_numpy()
                .flatten()
            )
            inst_in = (
                pb_utils.get_input_tensor_by_name(request, "instrument_types")
                .as_numpy()
                .flatten()
            )

            peptide_in_nomod = internal_without_mods(peptide_in)
            pep_len = np.vectorize(len)(peptide_in_nomod)

            oint = np.full(
                (peptide_in.shape[0], (np.max(pep_len) - 1) * 4), -1, dtype=np.float32
            )
            omz = np.full(oint.shape, -1, dtype=np.float32)
            oan = np.full(oint.shape, "", dtype=np.dtype("U5"))
            for l in set(pep_len):
                idx = l == pep_len
                tmp = self.predict_batch(
                    peptide_in[idx], charge_in[idx], ce_in[idx], inst_in[idx]
                )
                oint[idx, : tmp[0].shape[1]] = tmp[0]
                omz[idx, : tmp[1].shape[1]] = tmp[1]
                oan[idx, : tmp[2].shape[1]] = tmp[2]

            output_tensors = [
                pb_utils.Tensor(
                    "intensities",
                    oint.astype(self.output_dtype),
                ),
                pb_utils.Tensor("mz", omz.astype(self.output_dtype)),
                pb_utils.Tensor("annotation", oan.astype(self.anno_out_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def predict_batch(self, seq, charge, ce, inst):
        tensor_inputs = [
            pb_utils.Tensor("peptide_sequences", seq.reshape((-1, 1))),
            pb_utils.Tensor("precursor_charges", charge.reshape((-1, 1))),
            pb_utils.Tensor("collision_energies", ce.reshape((-1, 1))),
            pb_utils.Tensor("instrument_types", inst.reshape((-1, 1))),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="AlphaPept_ms2_generic_sb",
            requested_output_names=["intensities", "mz", "annotation"],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = [
                pb_utils.get_output_tensor_by_name(resp, "intensities").as_numpy(),
                pb_utils.get_output_tensor_by_name(resp, "mz").as_numpy(),
                pb_utils.get_output_tensor_by_name(resp, "annotation").as_numpy(),
            ]

            return output
