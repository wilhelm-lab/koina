"""Atom counts for (residue, UNIMOD) pairs that are not in the hand-written tables.

Prosit-PTM encodes a modified position by the atoms it gains and loses, so the network itself is not
limited to the listed pairs. For every listed pair whose UNIMOD composition is readable, gain - loss
equals the UNIMOD delta composition (isotopes counted as their element), and loss is the residue's
attachment group. An unlisted pair is therefore encoded as

    loss = attachment group of the residue (below)
    gain = loss + UNIMOD delta composition

Residues are only supported where the loss table has one clear (majority) attachment group; E, N, P
and the C-terminus have conflicting or no precedent and stay unsupported. This file is identical in
Prosit_Preprocess_ac_gain and Prosit_Preprocess_ac_loss so both models accept and refuse the same
pairs.
"""

import re

ELEMENTS = ["H", "C", "N", "O", "P", "S"]

# Majority "loss" value per residue in dict_ptm_atom_count_loss; "" is the peptide N-terminus.
dict_attachment_loss = {
    "": "H(2) N",
    "C": "H S",
    "D": "H O(2)",
    "H": "H N",
    "I": "H(2) N",
    "K": "H(2) N",
    "L": "H(2) N",
    "M": "H(3) C S",
    "Q": "H(2) N",
    "R": "H(2) N",
    "S": "H O",
    "T": "H O",
    "W": "H",
    "Y": "H O",
}

# Monosaccharide symbols used in UNIMOD delta compositions.
SUGAR_COMPOSITION = {
    "Hex": {"C": 6, "H": 10, "O": 5},
    "HexNAc": {"C": 8, "H": 13, "N": 1, "O": 5},
    "dHex": {"C": 6, "H": 10, "O": 4},
    "Pent": {"C": 5, "H": 8, "O": 4},
    "HexA": {"C": 6, "H": 8, "O": 6},
    "NeuAc": {"C": 11, "H": 17, "N": 1, "O": 8},
    "NeuGc": {"C": 11, "H": 17, "N": 1, "O": 9},
    "Kdn": {"C": 9, "H": 14, "O": 8},
}

_TOKEN = re.compile(r"(\d*)([A-Z][A-Za-z]*?)(?:\((-?\d+)\))?")


def parse_composition(composition):
    """Counts of H C N O P S in a UNIMOD composition string, isotopes folded into their element."""
    counts = dict.fromkeys(ELEMENTS, 0)
    for token in composition.split():
        match = _TOKEN.fullmatch(token)
        if match is None:
            raise KeyError(f"cannot read composition token '{token}'")
        _, symbol, number = match.groups()
        number = int(number) if number else 1
        if symbol in SUGAR_COMPOSITION:
            for element, count in SUGAR_COMPOSITION[symbol].items():
                counts[element] += count * number
        elif symbol in counts:
            counts[symbol] += number
        else:
            raise KeyError(f"element '{symbol}' cannot be encoded (only H C N O P S)")
    return [counts[e] for e in ELEMENTS]


def zero_shot_counts(unimod, residue, record_id):
    """(gain, loss) atom counts for an unlisted pair, or KeyError explaining why it is unsupported."""
    key = f"{residue}_{record_id}"
    if residue not in dict_attachment_loss:
        raise KeyError(f"{key}: no attachment group for residue '{residue}'")
    entry = unimod.unimod_db_dict.get(f"UNIMOD:{record_id}")
    if entry is None or "delta_composition" not in entry:
        raise KeyError(f"{key}: UNIMOD:{record_id} has no delta composition")
    delta = parse_composition(entry["delta_composition"].strip('"'))
    loss = parse_composition(dict_attachment_loss[residue])
    gain = [a + b for a, b in zip(loss, delta)]
    if min(gain) < 0:
        raise KeyError(f"{key}: attachment group plus delta composition is negative")
    return gain, loss
