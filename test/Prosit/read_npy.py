import numpy as np
import h5py
from pyteomics import proforma


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


sequences = np.array(["CCDD", "DDCC"])
charges = np.array([1, 2])
ion_series = np.array(["y", "b"])
get_fragments = get_fragments(sequences, charges, ion_series)
print(get_fragments)
