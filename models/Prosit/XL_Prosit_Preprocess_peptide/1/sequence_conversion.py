import numpy as np
import triton_python_backend_utils as pb_utils

SEQ_LEN = 30
ALPHABET_UNMOD = {
    "A": 1,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "C": 24,
}
MAX_CHARGE = 6
ALPHABET_MOD = {"M[UNIMOD:35]": 21, "C[UNIMOD:4]": 2, "K[UNIMOD:1896]": 22, "K[UNIMOD:1884]": 23 }

# ALPHABET contains all amino acid and ptm abbreviations and
ALPHABET = {**ALPHABET_UNMOD, **ALPHABET_MOD}


def parse_modstrings(sequences, alphabet, translate=False, filter=False):
    """
    :param sequences: List of strings
    :param ALPHABET: dictionary where the keys correspond to all possible 'Elements' that can occur in the string
    :param translate: boolean to determine if the Elements should be translated to the corresponding values of ALPHABET
    :return: generator that yields a list of sequence 'Elements' or the translated sequence "Elements"
    """
    import re
    from itertools import repeat

    def split_modstring(sequence, r_pattern):
        split_seq = r_pattern.findall(sequence)
        if "".join(split_seq) == sequence:
            if translate:
                return [alphabet[aa] for aa in split_seq]
            elif not translate:
                return split_seq
        elif filter:
            return [0]

    pattern = sorted(alphabet, key=len, reverse=True)
    pattern = [re.escape(i) for i in pattern]
    regex_pattern = re.compile("|".join(pattern))
    return map(split_modstring, sequences, repeat(regex_pattern))


def character_to_array(character):
    array = np.zeros((1, SEQ_LEN), dtype=np.uint8)
    logger = pb_utils.Logger

    logger.log_info(str(character))
    generator_sequence_numeric = parse_modstrings(
        [character], alphabet=ALPHABET, translate=True, filter=True
    )
    enum_gen_seq_num = enumerate(generator_sequence_numeric)
    print(enum_gen_seq_num)
    for i, sequence_numeric in enum_gen_seq_num:
        if len(sequence_numeric) > SEQ_LEN:
            pass  # don't overwrite 0 in the array that is how we can differentiate
        else:
            array[i, 0 : len(sequence_numeric)] = sequence_numeric
    return array