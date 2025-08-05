import json
import csv
import math
import re
from collections import defaultdict
import numpy as np
import triton_python_backend_utils as pb_utils
from pyteomics.mass import std_aa_mass
from pyteomics.auxiliary import _nist_mass
from pyteomics.mass import Composition
from pyteomics import mass as pm


C13C12_MASSDIFF_U = _nist_mass["C"][13][0] - _nist_mass["C"][12][0]
PROTON_MASS_U = _nist_mass["H+"][0][0]


class TritonPythonModel:
    def __init__(self):
        self.ion_names = None
        self.max_charge = None
        self.max_length = None
        self.unimodID2mass = None
        self.nl2mass = None
        self.immonium2composition = None
        self.aa2nl2imm_annotations = None
        self.aa2nl2mod_imm_annotations = None
        self.ion2index = None
        self.index2ion = None
        self.dicsz = None

    def initialize(self, args):
        super().__init__()
        base_path = "Altimeter/Altimeter_2024_filter_splines/"
        with open(base_path + "config.json", "r", encoding="utf-8") as j:
            model_config = json.loads(j.read())
        self.parseIonDictionary(base_path + "ion_dictionary.txt")
        self.ion_names = np.array(list(self.ion2index.keys()), dtype=np.object_)

        self.max_charge = model_config["max_charge"]
        self.max_length = model_config["seq_len"]
        self.unimodID2mass = {
            int(k): v for k, v in model_config["unimodID2mass"].items()
        }
        self.nl2mass = {nl: pm.calculate_mass(formula=nl) for nl in model_config["NLs"]}

        # Precompute immonium masses and num sulfurs
        self.immonium2composition = {
            k: Composition(formula=v)
            for k, v in model_config["immonium2composition"].items()
        }
        self.aa2nl2imm_annotations = defaultdict(dict)
        self.aa2nl2mod_imm_annotations = defaultdict(dict)

        for i, ion in enumerate(self.ion_names):
            if ion[0] == "I":  # immonium
                annot = imm_annotation(ion, i)
                if "(" in annot.name:  # immonium + PTM
                    unimod = model_config["mod2unimodID"][
                        annot.name.split("(")[-1].split(")")[0]
                    ]
                    self.aa2nl2mod_imm_annotations[unimod][annot.getNLString()] = (
                        annot_stats(
                            annot.name,
                            annot.NL,
                            annot.NG,
                            annot.index,
                            self.immonium2composition,
                        )
                    )
                else:
                    self.aa2nl2imm_annotations[annot.name[1]][annot.getNLString()] = (
                        annot_stats(
                            annot.name,
                            annot.NL,
                            annot.NG,
                            annot.index,
                            self.immonium2composition,
                        )
                    )

    def execute(self, requests):

        responses = []
        for request in requests:

            params = json.loads(request.parameters())

            return_b = (
                "return_b_ions" not in params or params["return_b_ions"] == "True"
            )
            return_y = (
                "return_y_ions" not in params or params["return_y_ions"] == "True"
            )
            return_p = (
                "return_p_ions" not in params or params["return_p_ions"] == "True"
            )
            return_imm = (
                "return_imm_ions" not in params or params["return_imm_ions"] == "True"
            )
            # return_NL = (
            #    "return_neutral_losses" not in params
            #    or params["return_neutral_losses"] == "True"
            # )
            return_NL = False

            min_length = int(params["min_length"]) if "min_length" in params else 1
            min_mz = float(params["min_mz"]) if "min_mz" in params else 0
            max_mz = float(params["max_mz"]) if "max_mz" in params else math.inf

            peptides_in = (
                pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
                .as_numpy()
                .flatten()
            )
            peptides_in = [x.decode("utf-8") for x in peptides_in]

            precursor_charges = pb_utils.get_input_tensor_by_name(
                request, "precursor_charges"
            ).as_numpy()

            coefficients = pb_utils.get_input_tensor_by_name(
                request, "coefficients"
            ).as_numpy()

            knots = pb_utils.get_input_tensor_by_name(request, "knots").as_numpy()

            AUCs = pb_utils.get_input_tensor_by_name(request, "AUC").as_numpy()

            annotations = np.tile(
                self.ion_names.astype(dtype="S23"), knots.shape[0]
            ).reshape((-1, self.dicsz))
            mzs = -np.ones_like(annotations, dtype=np.float32)
            by_NLs = ["", "H2O", "NH3"] if return_NL else [""]

            for i, pep in enumerate(peptides_in):
                seq, mods = self.parseModifiedPeptide(pep)

                filt = self.filter(
                    seq,
                    mods,
                    precursor_charges[i],
                    mzs[i],
                    return_b,
                    return_y,
                    return_p,
                    return_imm,
                    min_length,
                    by_NLs,
                    min_mz,
                    max_mz,
                )

                annotations[i][filt] = ""
                coefficients[i, :, filt] = -1
                AUCs[i][filt] = -1

            cf = pb_utils.Tensor("coefficients_filtered", coefficients)
            kf = pb_utils.Tensor("knots_filtered", knots)
            af = pb_utils.Tensor("annotations_filtered", annotations)
            mf = pb_utils.Tensor("mz_filtered", mzs)
            aucf = pb_utils.Tensor("AUC_filtered", AUCs)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[cf, kf, af, mf, aucf])
            )

        return responses

    def finalize(self):
        pass

    def parseIonDictionary(self, path):
        self.ion2index = {}
        with open(path, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter="\t")
            for row in reader:
                self.ion2index[row[0]] = len(self.ion2index)
        self.index2ion = {b: a for a, b in self.ion2index.items()}
        self.dicsz = len(self.ion2index)

    def populateValidIon(self, i, mz, mzs, filt):
        mzs[i] = mz
        filt[i] = True

    def getAnnotName(self, frag_type, nl, frag_z):
        name = frag_type if nl == "" else frag_type + "-" + nl
        if frag_z > 1:
            name += "^" + str(frag_z)
        return name

    def getIonSeries(
        self, mono_mass_base, charge, frag_type, mzs, filt, by_NLs, min_mz, max_mz
    ):
        for nl in by_NLs:
            mono_mass = mono_mass_base - self.nl2mass[nl]

            for frag_z in range(1, 1 + min(self.max_charge, charge)):
                ion = self.getAnnotName(frag_type, nl, frag_z)

                if ion not in self.ion2index:
                    continue

                mz = (mono_mass / frag_z) + PROTON_MASS_U
                if mz < min_mz or mz > max_mz:
                    continue

                ion_index = self.ion2index[ion]

                self.populateValidIon(ion_index, mz, mzs, filt)

    def filter(
        self,
        seq,
        mods,
        charge,
        mzs,
        return_b,
        return_y,
        return_p,
        return_imm,
        min_length,
        by_NLs,
        min_mz,
        max_mz,
    ):
        charge = int(round(charge[0]))

        filt = np.full_like(mzs, False, dtype=np.bool_)

        if return_b:
            mono_mass_base = 0
            for i, aa in enumerate(seq[:-1]):
                mono_mass_base += std_aa_mass[aa] + self.unimodID2mass[mods[i]]
                if i >= min_length - 1:
                    self.getIonSeries(
                        mono_mass_base,
                        charge,
                        "b" + str(i + 1),
                        mzs,
                        filt,
                        by_NLs,
                        min_mz,
                        max_mz,
                    )

        if return_y:
            mono_mass_base = self.nl2mass["H2O"]
            for i, aa in enumerate(reversed(seq[1:])):
                mono_mass_base += (
                    std_aa_mass[aa] + self.unimodID2mass[mods[len(seq) - i - 1]]
                )
                if i >= min_length - 1:
                    self.getIonSeries(
                        mono_mass_base,
                        charge,
                        "y" + str(i + 1),
                        mzs,
                        filt,
                        by_NLs,
                        min_mz,
                        max_mz,
                    )

        if return_p:
            mono_mass_base = self.nl2mass["H2O"]
            for i, aa in enumerate(seq):
                mono_mass_base += std_aa_mass[aa] + self.unimodID2mass[mods[i]]

            self.getIonSeries(
                mono_mass_base, charge, "p", mzs, filt, by_NLs, min_mz, max_mz
            )

        if return_imm and min_length <= 1:
            unique_AAs = set(seq)
            for aa in unique_AAs:
                if aa in self.aa2nl2imm_annotations:
                    for nl in self.aa2nl2imm_annotations[aa]:
                        annot = self.aa2nl2imm_annotations[aa][nl]
                        if annot.mono_mass < min_mz or annot.mono_mass > max_mz:
                            continue
                        self.populateValidIon(annot.index, annot.mono_mass, mzs, filt)
            for _, mod in mods.items():
                for nl in self.aa2nl2mod_imm_annotations[mod]:
                    annot = self.aa2nl2mod_imm_annotations[mod][nl]
                    if annot.mono_mass < min_mz or annot.mono_mass > max_mz:
                        continue
                    self.populateValidIon(annot.index, annot.mono_mass, mzs, filt)

        return ~filt

    def parseModifiedPeptide(self, peptide):

        def find_mod_indices(subseq_list):
            inds = [-1]
            for s in subseq_list:
                inds.append(inds[-1] + len(s))

            return inds[1:]

        # Extract sequence and mods from sequence with unimod annotations
        modseq = peptide
        mss = peptide.split("[")

        mod = defaultdict(int)
        if len(mss) > 1:
            list2 = [n.split("]") for n in mss[1:]]  # [[UNIMOD:#, AGAGAGA],...]
            seq1 = [mss[0]]
            for o in list2:
                seq1.append(o[1])
            mod_inds = find_mod_indices(seq1[:-1])
            assert len(mod_inds) == len(list2), f"{len(mod_inds)} | {len(list2)}"

            seq = "".join(seq1)
            for o, p in zip(mod_inds, list2):
                mod[int(o)] = int(p[0].split(":")[-1])
        else:
            seq = modseq

        return seq, mod


class annot_stats:
    def __init__(self, name, NLs, NGs, index, immonium2composition):
        self.index = index
        self.mono_mass, self.sulfurs = self.computeStats(
            name, NLs, NGs, immonium2composition
        )

    def computeStats(self, name, NLs, NGs, immonium2composition):
        ef = immonium2composition[name]
        for nl in NLs:
            if nl[0].isdigit():
                nl_count = int(re.search("^\\d*", nl).group(0))
                for _ in range(nl_count):
                    ef -= Composition(formula=nl[len(str(nl_count)) :])
            else:
                ef -= Composition(formula=nl)
        for ng in NGs:
            if ng[0].isdigit():
                ng_count = int(re.search("^\\d*", ng).group(0))
                for _ in range(ng_count):
                    ef += Composition(formula=ng[len(str(ng_count)) :])
            else:
                ef += Composition(formula=ng)
        return ef.mass() + PROTON_MASS_U, ef["S"]


class imm_annotation:
    def __init__(self, entry, index):
        self.NG = []
        self.NL = []
        self.index = index

        # get neutral gain
        if "+" in entry:
            splits = entry.split("+")
            self.NG = splits[1:]
            entry = splits[0]

        # get neutral loss
        if "-" in entry:
            splits = entry.split("-")
            self.NL = splits[1:]
            entry = splits[0]

        # get remaining name
        self.name = entry

    def getNLString(self):
        out = ""
        if self.NL:
            out += "-" + "-".join(self.NL)
        if self.NG:
            out += "+" + "+".join(self.NG)
        return out
