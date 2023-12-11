import numpy as np
import re
import json
import triton_python_backend_utils as pb_utils
import os

class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger
        
        chlim = [1,8]
        self.chrng = chlim[-1] - chlim[0] + 1
        
        P = 'UniSpec/uspreprocess/'
        self.dic = {b:a for a,b in enumerate('ARNDCQEGHILKMFPSTWYVX')}
        self.revdic = {b:a for a,b in self.dic.items()}
        self.mdic = {
            b : a+len(self.dic) 
            for a,b in enumerate(['']+open(P+"modifications.txt").read().split("\n"))
        }

        self.revmdic = {b:a for a,b in self.mdic.items()}
        self.mass = {
            line.split()[0] : float(line.split()[1])
            for line in open(P+"masses.txt")
        }
        self.dictionary = {
            line.split()[0] : int(line.split()[1])
            for line in open(P+"dictionary.txt")
        }
        self.revdictionary = {b:a for a,b in self.dictionary.items()}
        self.dicsz = len(self.dictionary)

        self.seq_channels = len(self.dic) + len(self.mdic)
        self.channels = len(self.dic) + len(self.mdic) + self.chrng + 1
        self.seq_len = 40
                    

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
    
    def input_from_str(self, strings):
        bs = len(strings)
        outseq = np.zeros((bs, self.channels, self.seq_len), dtype=np.float32)

        info = []
        for m in range(len(strings)):
            # input comes in as np.array([[byte1], [byte2] ... [byteN]])
            # - each byte looks like e.g. b'AGAGAGA'
            # - str(b'AGAGAGA') == "b'hello'"
            [seq, other] = str(strings[m][0])[2:-1].split('/')
            osplit = other.split("_")
            [charge, mod, ev, nce] = osplit
            charge = int(charge);ev = float(ev[:-2]);nce = float(nce[3:])
            info.append((seq,mod,charge,ev,nce))
            out = self.inptsr(info[-1])
            outseq[m] = out[0]
        
        return outseq, info

    def inptsr(self, info):
        
        (seq, mod, charge, ev, nce) = info
        output = np.zeros((self.channels, self.seq_len), dtype=np.float32)

        # Sequence
        assert len(seq) <= self.seq_len, "Exceeded maximum peptide length."
        intseq = (
            [self.dic[o] for o in seq] + (self.seq_len-len(seq))*[self.dic['X']]
        )
        assert len(intseq) == self.seq_len
        output[:len(self.dic)] = np.eye(len(self.dic))[intseq].T

        # PTMs
        Mstart = mod.find('(') if mod!='0' else 1
        modamt = int(mod[0:Mstart])
        output[len(self.dic)] = 1.
        if modamt > 0:
            hold = [re.sub('[()]', '', n) for n in mod[Mstart:].split(")(")]
            for n in hold:
                [pos, aa, modtyp] = n.split(',')
                output[self.mdic[modtyp], int(pos)] = 1.
                output[len(self.dic), int(pos)] = 0.

        output[self.seq_channels+int(charge)-1] = 1.
        output[-1, :] = float(ev) / 100.

        return output

    def execute(self, requests):
        responses = []
        labels = []
        for request in requests:
            label_in = (
                pb_utils.get_input_tensor_by_name(request, "labels")
                .as_numpy()
                .flatten()
                .astype('object')
            )
            labels.append(label_in)

            input_tensor, info = self.input_from_str(labels)

            tmp = self.predict_batch(input_tensor)

            output_tensors = [pb_utils.Tensor("intensities", tmp[0].astype(self.output_dtype))]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))
        
        return responses

    def predict_batch(self, input_tensor):
        
        tensor_inputs = [
            pb_utils.Tensor("input_tensor", input_tensor)
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="unispec23",
            requested_output_names=['intensities'],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = [
                pb_utils.get_output_tensor_by_name(resp, 'intensities').as_numpy()
            ]

            return output



