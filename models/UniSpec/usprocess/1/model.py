import numpy as np
import re
import json
import triton_python_backend_utils as pb_utils
import os


def NCE2eV(nce, mz, charge, instrument="lumos"):
    """
    Allowed instrument types (230807):
    - QE: q_exactive, QEHFX: q_exactive_hfx, LUMOS: lumos, ELITE: elite, VELOS: velos,
      NONE: use nce (or eV) straight
    """
    assert instrument.upper() in [
        "QE",
        "QEHFX",
        "LUMOS",
        "VELOS",
        "ELITE",
        "NONE",
    ], instrument

    if instrument.lower() == "none":
        return nce

    if instrument.lower() == ("qe" or "qehfx" or "elite"):
        if charge == 2:
            cf = 0.9
        elif charge == 3:
            cf = 0.85
        elif charge == 4:
            cf = 0.8
        elif charge == 5:
            cf = 0.75
        else:
            RuntimeError("Charge not supported")
    if instrument.lower() == ("qe" or "qehfx"):
        ev = nce * mz / 500 * cf
    elif instrument.lower() == "elite":
        ev = nce * mz * 500 * cf
    elif instrument.lower() == "velos":
        if charge == 2:
            ev = (0.0015 * nce - 0.0004) * mz
        elif charge == 3:
            ev = (0.0012 * nce - 0.0006) * mz
        elif charge == 4:
            ev = (0.0008 * nce + 0.0061) * mz
        else:
            RuntimeError("Charge not supported")
    elif instrument.lower() == "lumos":
        if charge == 1:
            crosspoint = (-0.4873 * nce + 0.1931) / (-0.00094 * nce + 5.11e-4)
            if mz < crosspoint:
                ev = (9.85e-4 * nce + 5.89e-4) * mz + (0.4049 * nce + 5.752)
            else:
                ev = (1.920e-3 * nce + 7.84e-5) * mz - 8.24e-2 * nce + 5.945
        elif charge == 2:
            crosspoint = 0.4106 * nce / (7.836e-4 * nce - 2.704e-6)
            if mz < crosspoint:
                ev = (8.544e-4 * nce - 5.135e-5) * mz + 0.3383 * nce + 5.998
            else:
                ev = (1.638e-3 * nce - 5.405e-5) * mz - 0.07234 * nce + 5.998
        elif charge == 3:
            crosspoint = (-0.3802 * nce + 0.3261) / (-7.3e-4 * nce + 1.027e-3)
            if mz < crosspoint:
                ev = (8.09e-4 * nce + 1.011e-3) * mz + 0.3129 * nce + 5.673
            else:
                ev = (1.540e-3 * nce - 1.62e-5) * mz - 0.0673 * nce + 5.999
        elif charge >= 4:
            crosspoint = (0.3083 * nce + 0.9073) / (5.61e-4 * nce + 2.143e-3)
            if mz < crosspoint:
                ev = (8.79e-4 * nce - 2.183e-3) * mz + 0.245 * nce + 6.917
            else:
                ev = (1.44e-3 * nce - 4e-5) * mz - 0.0633 * nce + 6.010
        else:
            RuntimeError("Charge not supported")
    else:
        RuntimeError("instrument type not found")

    return ev


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

        chlim = [1, 8]
        self.chrng = chlim[-1] - chlim[0] + 1

        P = "UniSpec/usprocess/"
        self.dic = {b: a for a, b in enumerate("ARNDCQEGHILKMFPSTWYVX")}
        self.revdic = {b: a for a, b in self.dic.items()}
        self.mdic = {
            b: a + len(self.dic)
            for a, b in enumerate(
                [""] + open(P + "modifications.txt").read().split("\n")
            )
        }
        # unimod mappings
        self.mdicum = {
            1: "Acetyl",
            4: "Carbamidomethyl",
            28: "Gln->pyro-Glu",
            27: "Glu->pyro-Glu",
            35: "Oxidation",
            21: "Phospho",
            26: "Pyro-carbamidomethyl",
            # 4: 'CAM'
        }
        self.rev_mdicum = {n: m for m, n in self.mdicum.items()}
        self.um2ch = lambda num: self.mdic[self.mdicum[num]]

        self.revmdic = {b: a for a, b in self.mdic.items()}
        self.mass = {
            line.split()[0]: float(line.split()[1]) for line in open(P + "masses.txt")
        }
        self.dictionary = {
            line.split()[0]: int(line.split()[1]) for line in open(P + "dictionary.txt")
        }
        self.revdictionary = {b: a for a, b in self.dictionary.items()}
        self.dicsz = len(self.dictionary)

        self.seq_channels = len(self.dic) + len(self.mdic)
        self.channels = len(self.dic) + len(self.mdic) + self.chrng + 1
        self.seq_len = 40

    def initialize(self, args):
        model_config = json.loads(args["model_config"])

    def calcmass(self, seq, pcharge, mods, ion):
        """
        Calculating the mass of fragments

        Parameters
        ----------
        seq : Peptide sequence (str)
        pcharge : Precursor charge (int)
        mods : Modification lists of list ([[pos(int), typ(int)],...])
        ion : Ion type (str)

        Returns
        -------
        mass as a float

        """
        # modification
        modamt = len(mods)
        modlst = []
        if modamt > 0:
            for mod in mods:
                [pos, typ] = mod  # mod position, amino acid, and type
                typ = self.mdicum[typ]
                modlst.append([int(pos), self.mass[typ]])

        # isotope
        isomass = self.mass["iso1"] if "+i" in ion else self.mass["iso2"]
        if ion[-1] == "i":  # evaluate isotope and set variable iso
            hold = ion.split("+")
            iso = 1 if hold[-1] == "i" else int(hold[-1][:-1])
            ion = "+".join(hold[:-1])  # join back if +CO was split
        else:
            iso = 0

        # If internal calculate here and return
        if ion[:3] == "Int":
            ion = ion.split("+")[0] if iso != 0 else ion
            ion = ion.split("-")
            nl = self.mass[ion[1]] if len(ion) > 1 else 0
            [start, extent] = [int(z) for z in ion[0][3:].split(">")]
            modmass = sum(
                [
                    ms[1]
                    for ms in modlst
                    if ((ms[0] >= start) & (ms[0] < (start + extent)))
                ]
            )
            return (
                sum([self.mass[aa] for aa in seq[start : start + extent]])
                - nl
                + iso * isomass
                + self.mass["i"]
                + modmass
            )
        # if TMT, calculate here and return
        if ion[:3] == "TMT":
            ion = ion.split("+")[0] if iso != 0 else ion

        # product charge
        hold = ion.split("^")  # separate off the charge at the end, if at all
        charge = 1 if len(hold) == 1 else int(hold[-1])  # no ^ means charge 1
        # extent
        letnum = hold[0].split("-")[0]
        let = letnum[0]  # ion type and extent is always first string separated by -
        num = (
            int(letnum[1:]) if ((let != "p") & (let != "I")) else 0
        )  # p type ions never have number

        # neutral loss
        nl = 0
        hold = hold[0].split("-")[1:]  # most are minus, separated by -
        """If NH2-CO-CH2SH, make the switch to C2H5NOS. Get rid of CO and CH2SH."""
        if len(hold) > 0 and ("NH2" in hold[0]):
            mult = (
                int(hold[0][0])
                if ((ord(hold[0][0]) >= 48) & (ord(hold[0][0]) <= 57))
                else ""
            )
            hold[0] = str(mult) + "C2H5NOS"
            del (hold[1], hold[1])
        for item in hold:
            if "+" in item:  # only CO can be a +
                items = item.split("+")  # split it e.g. H2O+CO
                nl -= self.mass[items[0]]  # always minus the first
                nl += self.mass[items[1]]  # always plus the second
            else:
                if (ord(item[0]) >= 48) & (
                    ord(item[0]) <= 57
                ):  # if there are e.g. 2 waters -> 2H2O
                    mult = int(item[0])
                    item = item[1:]
                else:
                    mult = 1
                nl -= mult * self.mass[item]

        if let == "a":
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum(
                [mod[1] for mod in modlst if num > mod[0]]
            )  # if modification is before extent from n terminus
            return (
                self.mass["i"]
                + (sm + modmass - self.mass["CO"] + nl + iso * isomass) / charge
            )
        elif let == "b":
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum(
                [mod[1] for mod in modlst if num > mod[0]]
            )  # if modification is before extent from n terminus
            return self.mass["i"] + (sm + modmass + nl + iso * isomass) / charge
        elif let == "p":
            # p eq.: proton + (aa+H2O+mods+i)/charge
            sm = sum([self.mass[aa] for aa in seq])
            charge = int(pcharge) if ((charge == 1) & ("^1" not in ion)) else charge
            modmass = sum([mod[1] for mod in modlst])  # add all modifications
            return (
                self.mass["i"]
                + (sm + self.mass["H2O"] + modmass + nl + iso * isomass) / charge
            )
        elif let == "y":
            sm = sum([self.mass[aa] for aa in seq[-num:]])
            modmass = sum(
                [mod[1] for mod in modlst if ((len(seq) - num) <= mod[0])]
            )  # if modification is before extent from c terminus
            return (
                self.mass["i"]
                + (sm + self.mass["H2O"] + modmass + nl + iso * isomass) / charge
            )  # 0.9936 1.002
        elif let == "I":
            sm = self.mass[letnum]
            return (sm + iso * isomass) / charge
        else:
            return False

    def rawinp2tsr(self, sequence, charge, ce, inst):
        Evs = []

        def find_mod_indices(subseq_list):
            inds = [-1]
            for s in subseq_list:
                inds.append(inds[-1] + len(s))

            return inds[1:]

        bs = len(sequence)
        outseq = np.zeros((bs, self.channels, self.seq_len), dtype=np.float32)

        info = []
        for m, (mse, ch, nce, ins) in enumerate(zip(sequence, charge, ce, inst)):

            # Extract sequence and mods from sequence with unimod annotations
            modseq = str(mse)[2:-1]
            mss = modseq.split("[")

            if len(mss) > 1:

                list2 = [n.split("]") for n in mss[1:]]  # [[UNIMOD:#, AGAGAGA],...]

                seq1 = [mss[0]]
                for o in list2:
                    seq1.append(o[1])
                mod_inds = find_mod_indices(seq1[:-1])
                assert len(mod_inds) == len(list2), "%d | %d" % (
                    len(mod_inds),
                    len(list2),
                )

                seq = "".join(seq1)
                mod = [[o, int(p[0].split(":")[-1])] for o, p in zip(mod_inds, list2)]
            else:
                seq = modseq
                mod = []

            # Convert nce to ev
            mz = self.calcmass(seq, ch, mod, "p")
            ins = str(ins)[2:-1]
            ev = NCE2eV(nce, mz, ch, ins)

            info.append((seq, mod, ch, ev, nce))
            out = self.inptsr(info[-1])
            outseq[m] = out

        return outseq, info

    def inptsr(self, info):

        (seq, mod, charge, ev, nce) = info
        output = np.zeros((self.channels, self.seq_len), dtype=np.float32)

        # Sequence
        assert len(seq) <= self.seq_len, "Exceeded maximum peptide length."
        intseq = [self.dic[o] for o in seq] + (self.seq_len - len(seq)) * [
            self.dic["X"]
        ]
        assert len(intseq) == self.seq_len
        output[: len(self.dic)] = np.eye(len(self.dic))[intseq].T

        # PTMs
        output[len(self.dic)] = 1.0
        if len(mod) > 0:
            for n in mod:
                [pos, modtyp] = n
                output[self.um2ch(modtyp), int(pos)] = 1.0
                output[len(self.dic), int(pos)] = 0.0

        output[self.seq_channels + int(charge) - 1] = 1.0
        output[-1, :] = float(ev) / 100.0

        return output

    def filter_fake(self, pepinfo, masses, ions):
        """
        Filter out the ions which cannot possibly occur for the peptide being
         predicted.

        Parameters
        ----------
        pepinfo : tuple of (sequence, mods, charge, ev, nce). Identical to
                   second output of str2dat().
        masses: array of predicted ion masses
        ions : array or list of predicted ion strings.

        Returns
        -------
        Return a numpy boolean array which you can use externally to select
         indices of m/z, abundance, or ion arrays

        """
        (seq, mods, charge, ev, nce) = pepinfo

        # modification
        modlst = []
        if len(mods) > 0:
            for mod in mods:
                [pos, typ] = mod
                modlst.append(self.mdicum[typ])

        filt = []
        for ion in ions:
            ext = (
                len(seq)
                if ion[0] == "p"
                else (
                    int(ion[1:].split("-")[0].split("+")[0].split("^")[0])
                    if (ion[0] in ["a", "b", "y"])
                    else 0
                )
            )
            a = True
            if "Int" in ion:
                [start, ext] = [
                    int(j) for j in ion[3:].split("+")[0].split("-")[0].split(">")[:2]
                ]
                # Do not write if the internal extends beyond length of peptide
                if (start + ext) >= len(seq):
                    a = False
            if (ion[0] in ["a", "b", "y"]) and (
                int(ion[1:].split("-")[0].split("+")[0].split("^")[0]) > (len(seq) - 1)
            ):
                # Do not write if a/b/y is longer than length-1 of peptide
                a = False
            if ("H3PO4" in ion) & ("Phospho" not in modlst):
                # Do not write Phospho specific neutrals for non-phosphopeptide
                a = False
            if ("CH3SOH" in ion) & ("Oxidation" not in modlst):
                a = False
            if ("CH2SH" in ion) & ("Carbamidomethyl" not in modlst):
                a = False
            if ("IPA" in ion) & ("P" not in seq):
                a = False
            filt.append(a)
        # Qian says all masses must be >60 and <1900
        return (np.array(filt)) & (masses > 60)

    def batch_mz(self, infos):
        mz = np.zeros((len(infos), 7919)).astype(np.float32)
        for m, info in enumerate(infos):
            (seq, mod, charge, ev, nce) = info
            mzs = np.array(
                [self.calcmass(seq, charge, mod, ion) for ion in self.dictionary.keys()]
            )
            mz[m] = mzs

        return mz

    def convert_internal_batch(self, ions, info):
        for i, j in enumerate(info):
            seq, mod, charge, ev, nce = j
            for k, ion in enumerate(ions[i]):
                if "Int" in ion:
                    [start, ext] = [
                        int(j)
                        for j in ion[3:].split("+")[0].split("-")[0].split(">")[:2]
                    ]
                    back = ion[len(str(start)) + len(str(ext)) + 4 :]
                    ret_ion = "Int/%s/%d" % (seq[start : start + ext] + back, start)
                    ions[i, k] = ret_ion

    def ToSpec(
        self,
        pred,
        pepinfo,
        top=200,
        rm_lowmz=True,
        rm_fake=True,
    ):

        pred /= pred.max(1, keepdims=True)
        rdinds = np.argsort(pred, axis=-1)[:, -top:]
        tile = np.tile(np.arange(pred.shape[0])[:, None], [1, top])
        pions = np.array(list(self.dictionary.keys()), dtype=np.object_)[rdinds]
        
        pints = pred[tile, rdinds]
        pmass = np.array(
            [
                [
                    self.calcmass(pepinfo[n][0], pepinfo[n][2], pepinfo[n][1], ion)
                    for ion in pions[n]
                ]
                for n in range(pred.shape[0])
            ]
        )

        if rm_fake:
            filt = np.array(
                [
                    self.filter_fake(pepinfo[n], pmass[n], pions[n])
                    for n in range(len(pepinfo))
                ]
            )
            pints[filt == False] = -1
        
        self.convert_internal_batch(pions, pepinfo)
        sort = np.argsort(pions, axis=1)
        pmass = pmass[tile, sort]
        pints = pints[tile, sort]
        pions = pions[tile, sort]
        return (pmass, pints, pions)

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

            input_tensor, info = self.rawinp2tsr(peptide_in, charge_in, ce_in, inst_in)
            tmp = self.predict_batch(input_tensor)
            (mzs, ints, anns) = self.ToSpec(tmp[0], info)

            output_tensors = [
                pb_utils.Tensor("intensities", ints.astype(self.output_dtype)),
                pb_utils.Tensor("mz", mzs.astype(self.output_dtype)),
                pb_utils.Tensor("annotation", anns.astype(np.object_)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def predict_batch(self, input_tensor):

        tensor_inputs = [pb_utils.Tensor("input_tensor", input_tensor)]

        infer_request = pb_utils.InferenceRequest(
            model_name="unispec23",
            requested_output_names=["intensities"],
            inputs=tensor_inputs,
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU, 0
            ),
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = [
                pb_utils.get_output_tensor_by_name(resp, "intensities").as_numpy(),
            ]

            return output
