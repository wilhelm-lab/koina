import numpy as np

SEQ_LEN = 30  # Sequence length for prosit
VEC_LENGTH = 174


def create_masking(charges_array, sequences_lengths):
    """
    assume reshaped output of prosit, shape sould be (num_seq, 174)
    set filtered output where not allowed positions are set to -1
    prosit output has the form:
    y1+1 y1+2 y1+3 b1+1 b1+2 b1+3 y2+1     y2+2 y2+3     b2+1     b2+2 b2+3
    if charge >= 3: all allowed
    if charge == 2: all +3 invalid
    if charge == 1: all +2 & +3 invalid
    """

    assert len(charges_array) == len(sequences_lengths)

    mask = np.ones(shape=(len(charges_array), VEC_LENGTH), dtype=np.int32)

    for i in range(len(charges_array)):
        charge_one_hot = charges_array[i]
        len_seq = sequences_lengths[i]
        m = mask[i]

        # filter according to peptide charge
        if np.array_equal(charge_one_hot, [1, 0, 0, 0, 0, 0]):
            invalid_indexes = [(x * 3 + 1) for x in range((SEQ_LEN - 1) * 2)] + [
                (x * 3 + 2) for x in range((SEQ_LEN - 1) * 2)
            ]
            m[invalid_indexes] = -1

        elif np.array_equal(charge_one_hot, [0, 1, 0, 0, 0, 0]):
            invalid_indexes = [x * 3 + 2 for x in range((SEQ_LEN - 1) * 2)]
            m[invalid_indexes] = -1

        if len_seq < SEQ_LEN:
            invalid_indexes = range((len_seq - 1) * 6, VEC_LENGTH)
            m[invalid_indexes] = -1

    return mask


def apply_masking(peaks, mask):
    peaks[peaks < 0] = 0
    out = np.multiply(peaks, mask)
    out = (out.T / np.max(out, axis=1)).T
    return out
