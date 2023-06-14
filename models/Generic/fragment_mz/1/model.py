import json
import triton_python_backend_utils as pb_utils
from pyteomics import proforma
import numpy as np


def assign_fragments_to_array(idx, arr, frag):
    try:
        arr[idx, :, :, : frag.shape[-1]] = frag
        return arr
    except ValueError:
        arr = np.concatenate((arr, np.zeros(shape=arr.shape)), axis=-1)
        arr = assign_fragments_to_array(idx, arr, frag)
    return arr


def get_fragments(sequences, charges, ion_series):
    """Function to calculate all possible fragment mz

    Args:
        sequences (numpy.array): 1d array of peptide sequences
        charges (numpy.array): 1d array of fragment charges to generate, if 0 is provided the uncharged mass is provided
        ion_series (numpy.array): 1d array of ion_series to generate i.e. a,b,c,x,y,z

    Returns:
        numpy.array: 4 dimensional array of fragment mz.
        First dimension is equal to the number of sequences.
        Second dimension is the fragment ion series.
        Third dimension is the fragment charge.
        Fourth dimesion is the fragment number.
    """
    arr = np.zeros((len(sequences), len(ion_series), len(charges), 32))
    for idx, seq in enumerate(sequences):
        tmp = proforma.ProForma.parse(seq)
        tmp = np.array([[tmp.fragments(it, c) for c in charges] for it in ion_series])
        arr = assign_fragments_to_array(idx, arr, tmp)
    return arr


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.frag_mz_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = model_config = json.loads(args["model_config"])
        frag_mz_config = pb_utils.get_output_config_by_name(
            model_config, "output_fragmentmz"
        )
        self.frag_mz_dtype = pb_utils.triton_string_to_numpy(
            frag_mz_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "ProForma")
            peptide_in_list = [
                x.decode("utf-8") for x in peptide_in.as_numpy().tolist()
            ]

            charges_in_list = (
                pb_utils.get_input_tensor_by_name(request, "charges")
                .as_numpy()
                .tolist()
            )

            ion_series_in = pb_utils.get_input_tensor_by_name(request, "ion_series")
            ion_series_in_list = [
                x.decode("utf-8") for x in ion_series_in.as_numpy().tolist()
            ]

            fragment_mz = get_fragments(
                peptide_in_list, charges_in_list, ion_series_in_list
            )

            fragment_mz_out = pb_utils.Tensor(
                "output_fragmentmz", fragment_mz.astype(self.frag_mz_dtype)
            )

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[fragment_mz_out])
            )
        return responses

    def finalize(self):
        pass