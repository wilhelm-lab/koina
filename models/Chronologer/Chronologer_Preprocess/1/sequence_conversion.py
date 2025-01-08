import numpy as np
import re

# Dictionary for standard amino acids (1-20)
amino_acids = {
    "A": 1,
    "C": 2,
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
}

# Extended mapping for amino acid + UNIMOD combinations
amino_acid_unimod_map = {
    "C-4": 21,  # Carbamidomethyl Cysteine (C[UNIMOD:4])
    "M-35": 22,  # Oxidized Methionine (M[UNIMOD:35])
    "S-21": 25,  # Phosphoserine (S[UNIMOD:21])
    "T-21": 26,  # Phosphothreonine (T[UNIMOD:21])
    "Y-21": 27,  # Phosphotyrosine (Y[UNIMOD:21])
    "K-1": 28,  # Acetylated Lysine (K[UNIMOD:1])
    "K-125": 29,  # Succinylated Lysine (K[UNIMOD:125])
    "K-121": 30,  # Ubiquitinated Lysine (K[UNIMOD:121])
    "K-34": 31,  # Monomethyl Lysine (K[UNIMOD:34])
    "K-36": 32,  # Dimethyl Lysine (K[UNIMOD:36])
    "K-37": 33,  # Trimethyl Lysine (K[UNIMOD:37])
    "R-34": 34,  # Monomethyl Arginine (R[UNIMOD:34])
    "R-36": 35,  # Dimethyl Arginine (R[UNIMOD:36])
    "K-739": 36,  # TMT10-modified Lysine (K[UNIMOD:739])
    "K-737": 37,  # TMT0-modified Lysine (K[UNIMOD:737])
    "C-385": 23,  # S-carbamidomethylcysteine cyclization
    "E-27": 24,  # Pyroglutamate (E[UNIMOD:27])
    "Q-28": 24,  # Pyroglutamate (Q[UNIMOD:28])
}

# Corrected N-terminal modification mapping (TMT0 = 42, TMT10 = 43)
n_terminal_mod_map = {
    "": 38,  # Free N-terminal
    "1": 39,  # N-terminal acetylation
    "385": 41,  # S-carbamidomethylcysteine cyclization (index 0, first aa is 23)
    "27": 40,  # Pyroglutamate (Glu, first aa is 24)
    "28": 40,  # Pyroglutamate (Gln, first aa is 24)
    "737": 42,  # TMT0 N-terminal (swapped)
    "739": 43,  # TMT10 N-terminal (swapped)
}


# Function to parse peptide strings
def peptide_to_array(peptide: str) -> np.ndarray:
    # Initialize an array of zeros with a fixed length of 52
    arr = np.zeros(52, dtype=int)

    # Default N-terminal to free N-term (38) unless modified
    arr[0] = 38

    # Handle N-terminal modifications
    n_term_match = re.match(r"\[UNIMOD:(\d+)\]-([A-Z])", peptide)
    if n_term_match:
        unimod_id, aa = n_term_match.groups()
        arr[0] = n_terminal_mod_map.get(unimod_id, 38)  # Default free N-term is 38
        # Remove the N-terminal modification from the string
        peptide = re.sub(r"^\[UNIMOD:\d+\]-", "", peptide)

    # Regex to capture both unmodified and modified amino acids
    modified_peptide = re.findall(r"([A-Z])(?:\[UNIMOD:(\d+)\])?", peptide)

    # Iterate through amino acids and modifications
    index = 1
    for aa, unimod_id in modified_peptide:
        if unimod_id:  # If there's a modification
            arr[index] = amino_acid_unimod_map.get(f"{aa}-{unimod_id}", 0)
        else:  # If unmodified
            arr[index] = amino_acids.get(aa, 0)
        index += 1
        if index >= 51:  # Prevent overflow past the array size
            break

    # Set the C-terminal marker to 44, immediately after the final amino acid
    arr[index] = 44

    # Special handling for N-terminal cylclization modifications
    if arr[1] == 23:  # S-carbamidomethylcysteine cyclization
        arr[0] = 41
    elif arr[1] == 24:  # Pyroglutamate
        arr[0] = 40

    return arr
