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
        
        P = 'UniSpec/usprocess/'
        self.dic = {b:a for a,b in enumerate('ARNDCQEGHILKMFPSTWYVX')}
        self.revdic = {b:a for a,b in self.dic.items()}
        self.mdic = {
            b : a+len(self.dic) 
            for a,b in enumerate(['']+open(P+"modifications.txt").read().split("\n"))
        }
        # unimod mappings
        self.mdicum = {
            1: 'Acetyl',
            4: 'Carbamidomethyl',
            28: 'Gln->pyro-Glu',
            27: 'Glu->pyro-Glu',
            35: 'Oxidation',
            21: 'Phospho',
            26: 'Pyro-carbamidomethyl',
            4: 'CAM'
        }
        self.rev_mdicum = {n:m for n,m in self.mdicum.items()}
        self.um2ch = lambda num: self.mdic[self.mdicum[num]]

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
    
    def calcmass(self, seq, pcharge, mods, ion):
        """
        Calculating the mass of fragments

        Parameters
        ----------
        seq : Peptide sequence (str)
        pcharge : Precursor charge (int)
        mods : Modification string (str)
        ion : Ion type (str)

        Returns
        -------
        mass as a float

        """
        # modification
        Mstart = mods.find('(') if mods!='0' else 1
        modamt = int(mods[0:Mstart])
        modlst = []
        if modamt>0:
            Mods = [re.sub("[()]",'',m).split(',') for m in 
                     mods[Mstart:].split(')(')]
            for mod in Mods:
                [pos,aa,typ] = mod # mod position, amino acid, and type
                modlst.append([int(pos), self.mass[typ]])
        
        # isotope
        isomass = self.mass['iso1'] if '+i' in ion else self.mass['iso2']
        if ion[-1]=='i': # evaluate isotope and set variable iso
            hold = ion.split("+")
            iso = 1 if hold[-1]=='i' else int(hold[-1][:-1])
            ion = "+".join(hold[:-1]) # join back if +CO was split
        else:
            iso = 0
        
        # If internal calculate here and return
        if ion[:3]=='Int':
            ion = ion.split('+')[0] if iso!=0 else ion
            ion = ion.split('-')
            nl = self.mass[ion[1]] if len(ion)>1 else 0
            [start,extent] = [int(z) for z in ion[0][3:].split('>')]
            modmass = sum([ms[1] for ms in modlst 
                           if ((ms[0]>=start)&(ms[0]<(start+extent)))
                          ])
            return (sum([self.mass[aa] for aa in seq[start:start+extent]]) - nl
                    + iso*isomass + self.mass['i'] + modmass)
        # if TMT, calculate here and return
        if ion[:3]=='TMT':
            ion = ion.split('+')[0] if iso!=0 else ion

        # product charge
        hold = ion.split("^") # separate off the charge at the end, if at all
        charge = 1 if len(hold)==1 else int(hold[-1]) # no ^ means charge 1
        # extent
        letnum = hold[0].split('-')[0];let = letnum[0] # ion type and extent is always first string separated by -
        num = int(letnum[1:]) if ((let!='p')&(let!='I')) else 0 # p type ions never have number
        
        # neutral loss
        nl=0
        hold = hold[0].split('-')[1:] # most are minus, separated by -
        """If NH2-CO-CH2SH, make the switch to C2H5NOS. Get rid of CO and CH2SH.""" 
        if len(hold)>0 and ('NH2' in hold[0]):
            mult = (int(hold[0][0]) 
                    if ((ord(hold[0][0])>=48) & (ord(hold[0][0])<=57)) else '')
            hold[0] = str(mult)+'C2H5NOS';del(hold[1],hold[1])
        for item in hold:
            if '+' in item: # only CO can be a +
                items = item.split('+') # split it e.g. H2O+CO
                nl-=self.mass[items[0]] # always minus the first
                nl+=self.mass[items[1]] # always plus the second
            else:
                if (ord(item[0])>=48) & (ord(item[0])<=57): # if there are e.g. 2 waters -> 2H2O 
                    mult = int(item[0])
                    item = item[1:]
                else:
                    mult = 1
                nl-=mult*self.mass[item]

        if let=='a':
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum([mod[1] for mod in modlst if num>mod[0]]) # if modification is before extent from n terminus
            return self.mass['i'] + (sm + modmass - self.mass['CO'] + nl + iso*isomass) / charge
        elif let=='b':
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum([mod[1] for mod in modlst if num>mod[0]]) # if modification is before extent from n terminus
            return self.mass['i'] + (sm + modmass + nl + iso*isomass) / charge
        elif let=='p':
            # p eq.: proton + (aa+H2O+mods+i)/charge
            sm = sum([self.mass[aa] for aa in seq])
            charge = int(pcharge) if ((charge==1)&('^1' not in ion)) else charge
            modmass = sum([mod[1] for mod in modlst]) # add all modifications
            return self.mass['i'] + (sm + self.mass['H2O'] + modmass + nl + iso*isomass) / charge
        elif let=='y':
            sm = sum([self.mass[aa] for aa in seq[-num:]])
            modmass = sum([mod[1] for mod in modlst if ((len(seq)-num)<=mod[0])]) # if modification is before extent from c terminus
            return self.mass['i'] + (sm + self.mass['H2O'] + modmass + nl + iso*isomass)/charge #0.9936 1.002
        elif let=='I':
            sm = self.mass[letnum]
            return (sm + iso*isomass) / charge
        else:
            return False

    def input_from_str(self, strings):
        bs = len(strings[0])
        outseq = np.zeros((bs, self.channels, self.seq_len), dtype=np.float32)

        info = []
        #assert len(strings) == 50, "%s"%len(strings[0])
        for m in range(len(strings[0])):
            # input comes in as np.array([[byte1, byte2 ... byteN]])
            # - each byte looks like e.g. b'AGAGAGA'
            # - str(b'AGAGAGA') == "b'hello'"
            [seq, other] = str(strings[0][m])[2:-1].split('/')
            osplit = other.split("_")
            [charge, mod, ev, nce] = osplit
            charge = int(charge);ev = float(ev[:-2]);nce = float(nce[3:])
            info.append((seq,mod,charge,ev,nce))
            out = self.inptsr(info[-1])
            outseq[m] = out
        
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
        (seq,mods,charge,ev,nce) = pepinfo
        
        # modification
        # modlst = []
        # Mstart = mods.find('(') if mods!='0' else 1
        # modamt = int(mods[0:Mstart])
        # if modamt>0:
        #     Mods = mods[Mstart:].split(')(') # )( always separates modifications
        #     for mod in Mods:
        #         [pos,aa,typ] = re.sub('[()]', '', mod).split(',') # mod position, amino acid, and type
        #         modlst.append([int(pos), self.D.mass[typ]])
        
        filt = []
        for ion in ions:
            ext = (len(seq) if ion[0]=='p' else 
                   (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])
                    if (ion[0] in ['a','b','y']) else 0)
                   )
            a = True
            if "Int" in ion:
                [start,ext] = [
                    int(j) for j in 
                    ion[3:].split('+')[0].split('-')[0].split('>')[:2]
                ]
                # Do not write if the internal extends beyond length of peptide
                if (start+ext)>=len(seq): a = False
            if (
                (ion[0] in ['a','b','y']) and 
                (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])>(len(seq)-1))
                ):
                # Do not write if a/b/y is longer than length-1 of peptide
                a = False
            if ('H3PO4' in ion) & ('Phospho' not in mods):
                # Do not write Phospho specific neutrals for non-phosphopeptide
                a = False
            if ('CH3SOH' in ion) & ('Oxidation' not in mods):
                a = False
            if ('CH2SH' in ion) & ('Carbamidomethyl' not in mods):
                a = False
            if ('IPA' in ion) & ('P' not in seq):
                a = False
            filt.append(a)
        # Qian says all masses must be >60 and <1900
        return (np.array(filt)) & (masses>60)

    def ToSpec_1(self, 
                 pred, 
                 pepinfo, 
                 mint=5e-4, 
                 rm_lowmz=True, 
                 rm_fake=True
                 ):
        pred /= pred.max()
        piboo = pred > mint
        rdinds = np.where(piboo)
        pions = np.array([self.revdictionary[m] for m in rdinds])
        pints = np.array(pred[piboo])
        pmass = np.array([self.calcmass(seq,charge,mods,ion) for ion in pions])
        sort = np.argsort(pmass)
        pmass = pmass[sort]
        pints = pints[sort]
        pions = pions[sort]

        if rm_lowmz:
            filt = pmass>minmz
            pmass = pmass[filt]
            pints = pints[filt]
            pions = pions[filt]
        if rm_fake:
            filt = self.filter_fake(pepinfo[0], pmass, pions)
            pmass = pmass[filt]
            pints = pints[filt]
            pions = pions[filt]

        return (pmass,pints,pions)

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
            
            #assert len(info[0]) == 4, "%s,%s"%(info[0][0], info[0][1])

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
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU, 0
            )
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            output = [
                pb_utils.get_output_tensor_by_name(resp, 'intensities').as_numpy(),
            ]

            return output



