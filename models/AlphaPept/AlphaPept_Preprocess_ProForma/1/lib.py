import numpy as np
import pandas as pd
import re

MOD_DF = pd.read_csv("/models/AlphaPept/AlphaPept_Preprocess_ProForma/1/mod_df.csv")
MOD_DF.loc[MOD_DF["mod_name"] == "Oxidation@M", "mod_name"] = "M[UNIMOD:21]"
MOD_DF.loc[MOD_DF["mod_name"] == "Carbamidomethyl@C", "mod_name"] = "C[UNIMOD:4]"

MOD_ELEMENTS = [
    "C",
    "H",
    "N",
    "O",
    "P",
    "S",
    "B",
    "F",
    "I",
    "K",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Ac",
    "Ag",
    "Al",
    "Am",
    "Ar",
    "As",
    "At",
    "Au",
    "Ba",
    "Be",
    "Bi",
    "Bk",
    "Br",
    "Ca",
    "Cd",
    "Ce",
    "Cf",
    "Cl",
    "Cm",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Dy",
    "Er",
    "Es",
    "Eu",
    "Fe",
    "Fm",
    "Fr",
    "Ga",
    "Gd",
    "Ge",
    "He",
    "Hf",
    "Hg",
    "Ho",
    "In",
    "Ir",
    "Kr",
    "La",
    "Li",
    "Lr",
    "Lu",
    "Md",
    "Mg",
    "Mn",
    "Mo",
    "Na",
    "Nb",
    "Nd",
    "Ne",
    "Ni",
    "No",
    "Np",
    "Os",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Po",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rh",
    "Rn",
    "Ru",
    "Sb",
    "Sc",
    "Se",
    "Si",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "Xe",
    "Yb",
    "Zn",
    "Zr",
    "2H",
    "13C",
    "15N",
    "18O",
    "?",
]
MOD_ELEMENTS_TO_IDX = {elements: i for i, elements in enumerate(MOD_ELEMENTS)}


def get_mod_features(proforma_str):
    split_seq = ProformaParser().parse_sequence(proforma_str)
    return [
        ";".join([str(i) for i, x in enumerate(split_seq) if "UNIMOD" in x]),
        ";".join([str(x) for i, x in enumerate(split_seq) if "UNIMOD" in x]),
        len(split_seq) - 2,
    ]


def parse_mod_formula(formula):
    """
    Parse a modification formula to a feature vector
    """
    feature = np.zeros(len(MOD_ELEMENTS_TO_IDX))
    elems = formula.strip(")").split(")")
    for elem in elems:
        chem, num = elem.split("(")
        num = int(num)
        if chem in MOD_ELEMENTS_TO_IDX:
            feature[MOD_ELEMENTS_TO_IDX[chem]] = num
        else:
            feature[-1] += num
    return feature


MOD_TO_FEATURE = {}

for modname, formula in MOD_DF[["mod_name", "composition"]].values:
    MOD_TO_FEATURE[modname] = parse_mod_formula(formula)


def encode_mod_features(mods, mod_sites, nAA):
    mod_features_list = (
        pd.Series(mods)
        .str.split(";")
        .apply(
            lambda mod_names: [MOD_TO_FEATURE[mod] for mod in mod_names if len(mod) > 0]
        )
    )
    mod_sites_list = (
        pd.Series(mod_sites)
        .str.split(";")
        .apply(lambda mod_sites: [int(site) for site in mod_sites if len(site) > 0])
    )
    mod_x_batch = np.zeros(
        # (len(nAA.as_numpy()), int(nAA.as_numpy()[0])+2, len(MOD_ELEMENTS))
        (len(nAA), int(nAA[0]) + 2, len(MOD_ELEMENTS))
    )
    for i, (mod_feats, mod_sites) in enumerate(zip(mod_features_list, mod_sites_list)):
        if len(mod_sites) > 0:
            for site, feat in zip(mod_sites, mod_feats):
                # Process multiple mods on one site
                mod_x_batch[i, site, :] += feat
            # mod_x_batch[i,mod_sites,:] = mod_feats
    return mod_x_batch


class ProformaParser:
    TERMINAL_MODIFICATION_SEP = "-"
    UNIMOD_ONTROLOGY = "UNIMOD"

    """
    returns three strings with terminal modifcations and middle sequence, order is n, s, c
    """

    @staticmethod
    def extract_terminal_mods_and_seq(sequence, terminal_sep=TERMINAL_MODIFICATION_SEP):
        n, s, c = "", sequence, ""

        if terminal_sep not in sequence:
            return n, sequence, c

        splitted_seq = sequence.split(terminal_sep)

        if len(splitted_seq) == 3:
            n, s, c = splitted_seq
        elif splitted_seq[0].startswith("["):
            n, s = splitted_seq
            c = ""
        elif splitted_seq[1].startswith("["):
            n = ""
            s, c = splitted_seq
        else:
            raise ValueError(
                "Failed at extracting terminal modifications, invalid input representation."
            )

        return n, s, c

    @staticmethod
    def parse_sequence(sequence, ontology=UNIMOD_ONTROLOGY):
        n, s, c = ProformaParser.extract_terminal_mods_and_seq(sequence)
        aa_seq = ProformaParser.extract_amino_acids_and_mods(s, ontology)

        return [n, *aa_seq, c]

    @staticmethod
    def extract_amino_acids_and_mods(sequence, ontology=UNIMOD_ONTROLOGY):
        import re

        # regex to capture single amino-acids with optional modifications
        reg_ex = "[A-Z]{1}(?:\[" + ontology + ":{0,}\d*\]){0,}"
        return re.findall(reg_ex, sequence)


def strip_mod_profroma(proforma_strings):
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in proforma_strings]


def character_to_array(peptide_in_list):
    seq_array = np.array(peptide_in_list).astype("U")
    x = np.array(seq_array).view(np.int32).reshape(len(seq_array), -1) - ord("A") + 1
    return np.pad(x, [(0, 0)] * (len(x.shape) - 1) + [(1, 1)])
