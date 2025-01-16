import triton_python_backend_utils as pb_utils
import numpy as np
import json
from collections import defaultdict


class TritonPythonModel:
    def initialize(self, args):
        super().__init__()
        base_path = "altimeter/Altimeter_2024_core/"
        with open(base_path + "config.json", "r") as j:
            model_config = json.loads(j.read())

        self.dic = {b: a for a, b in enumerate(model_config["AAs"])}
        self.mdic = {
            b: a + len(self.dic)
            for a, b in enumerate(model_config["mod2unimodID"].keys())
        }
        self.seq_len = model_config["seq_len"]
        self.chlim = [model_config["min_charge"], model_config["max_charge"]]

        self.chrng = self.chlim[-1] - self.chlim[0] + 1
        self.seq_channels = len(self.dic) + len(self.mdic)
        self.channels = len(self.dic) + len(self.mdic) + self.chrng + 1
        # unimod mappings
        self.mdicum = {v: k for k, v in model_config["mod2unimodID"].items()}
        self.um2ch = lambda num: self.mdic[self.mdicum[num]]

    def execute(self, requests):

        responses = []
        for request in requests:
            print(eval(request.parameters()), flush=True)

            peptides_in = (
                pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
                .as_numpy()
                .flatten()
            )

            peptides_in = [x.decode("utf-8") for x in peptides_in]

            peptides_out = self.encode_peptides(peptides_in)

            t = pb_utils.Tensor("sequence_encoded", peptides_out)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass

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
            assert len(mod_inds) == len(list2), "%d | %d" % (
                len(mod_inds),
                len(list2),
            )

            seq = "".join(seq1)
            for o, p in zip(mod_inds, list2):
                mod[int(o)] = int(p[0].split(":")[-1])
        else:
            seq = modseq
        return seq, mod

    def encode_peptides(self, peptides):

        bs = len(peptides)
        output_encoding = np.zeros(
            (bs, self.seq_channels, self.seq_len), dtype=np.float32
        )

        for i, pep in enumerate(peptides):
            seq, mod = self.parseModifiedPeptide(pep)

            # Input validation
            if len(seq) > self.seq_len:
                raise Exception(
                    "Peptide too long. Sequence:"
                    + pep
                    + " length:"
                    + str(len(seq))
                    + " Peptide_index:"
                    + str(i)
                    + " Max length allowed:"
                    + str(self.seq_len)
                )

            for j, aa in enumerate(seq):
                if aa not in self.dic or aa == "X":
                    raise Exception(
                        "Invalid AA in requested peptide. AA:"
                        + aa
                        + " Sequence:"
                        + pep
                        + " Peptide_index:"
                        + str(i)
                    )
                if aa == "C" and mod[j] != 4:
                    raise Exception(
                        "Only carbamidomethylated cysteines are supported. Sequence:"
                        + pep
                        + " Peptide_index:"
                        + str(i)
                        + " Expected [UNIMOD:4] after C"
                    )

            for pos, ptm in mod.items():
                if ptm not in self.mdicum:
                    raise Exception(
                        "Modification not supported. Sequence:"
                        + pep
                        + " mod:"
                        + str(ptm)
                        + " Peptide_index:"
                        + str(i)
                        + " Only methionine oxidation '[UNIMOD:35]' and cysteine carbamidomethylation '[UNIMOD:4]' are allowed."
                    )
                if ptm == 4 and seq[pos] != "C":
                    raise Exception(
                        "Invalid modification position. Sequence:"
                        + pep
                        + " mod:"
                        + str(ptm)
                        + " AA:"
                        + seq[pos]
                        + " Peptide_index:"
                        + str(i)
                        + " Carbamidomethylation must come after a C"
                    )
                if ptm == 35 and seq[pos] != "M":
                    raise Exception(
                        "Invalid modification position. Sequence:"
                        + pep
                        + " mod:"
                        + str(ptm)
                        + " AA:"
                        + seq[pos]
                        + " Peptide_index:"
                        + str(i)
                        + " Oxidation must come after a M"
                    )

            intseq = [self.dic[o] for o in seq] + (self.seq_len - len(seq)) * [
                self.dic["X"]
            ]

            # one-hot
            output_encoding[i, 0 : len(self.dic)] = np.eye(len(self.dic))[intseq].T

            # Unmodified position indicator
            output_encoding[i, len(self.dic)] = 1

            # PTM one-hot
            for pos, modtyp in mod.items():
                output_encoding[i, self.um2ch(modtyp), int(pos)] = 1
                output_encoding[i, len(self.dic), int(pos)] = 0

        return output_encoding
