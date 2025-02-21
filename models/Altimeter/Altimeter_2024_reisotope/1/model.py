import triton_python_backend_utils as pb_utils
import numpy as np
from isotopes import IsotopeSplineDB
from pyteomics.auxiliary import _nist_mass

C13C12_MASSDIFF_U = _nist_mass['C'][13][0] - _nist_mass['C'][12][0]

lambda_decode = lambda x: x.decode("utf-8")
vectorized_decode = np.vectorize(lambda_decode)

class TritonPythonModel:
    def initialize(self, args):
        base_path = "Altimeter/Altimeter_2024_reisotope/"
        self.isotopeSplineDB = IsotopeSplineDB(base_path + "IsotopeSplines_10kDa_21isotopes.xml")
        self.charge2isoStep = [C13C12_MASSDIFF_U / z for z in range(1,10)]
        self.charge2isoStep.insert(0,0)
    
    def execute(self, requests):
        responses = []
        for request in requests:
            params = eval(request.parameters())
            
            return_iso = "return_non_monoisotopes" not in params or params["return_non_monoisotopes"] == "True"
            
            prec_masses = (
                pb_utils.get_input_tensor_by_name(request, "precursor_masses")
                .as_numpy()
            )
            
            frag_masses = (
                pb_utils.get_input_tensor_by_name(request, "fragment_masses")
                .as_numpy()
            )
            
            prec_sulfurs = (
                pb_utils.get_input_tensor_by_name(request, "precursor_sulfurs")
                .as_numpy()
            )
            
            frag_sulfurs = (
                pb_utils.get_input_tensor_by_name(request, "fragment_sulfurs")
                .as_numpy()
            )
            
            intensities = (
                pb_utils.get_input_tensor_by_name(request, "intensities")
                .as_numpy()
            )
            
            isotope_isolation_efficiencies = (
                pb_utils.get_input_tensor_by_name(request, "isotope_isolation_efficiencies")
                .as_numpy()
            )
            
            annotations = (
                pb_utils.get_input_tensor_by_name(request, "annotations")
                .as_numpy()
            )
            
            mzs = (
                pb_utils.get_input_tensor_by_name(request, "mz")
                .as_numpy()
            )
            
            
            intensities_iso = np.expand_dims(intensities, 2).repeat(5, axis=2)
            mzs = np.expand_dims(mzs, 2).repeat(5, axis=2)
            
            annotations = vectorized_decode(annotations).astype(np.object_)
            annotations = np.expand_dims(annotations, 2).repeat(5, axis=2)
            
            for peptide_i in range(annotations.shape[0]):
                mass2dist = dict()
                for frag_i in range(annotations.shape[1]):
                    
                    if frag_sulfurs[peptide_i][frag_i] >= 0:
                        if not frag_masses[peptide_i, frag_i] in mass2dist:
                            if prec_masses[peptide_i][0] == frag_masses[peptide_i, frag_i]:
                                iso_dist = self.isotopeSplineDB.estimate_for_precursor_from_weights_and_sulfur(prec_masses[peptide_i][0], 4,
                                                                        prec_sulfurs[peptide_i][0], isotope_isolation_efficiencies[peptide_i])
                            else:
                                iso_dist = self.isotopeSplineDB.estimate_for_fragment_from_weights_and_sulfur(prec_masses[peptide_i][0], frag_masses[peptide_i, frag_i], 
                                                        0, 4, prec_sulfurs[peptide_i][0], frag_sulfurs[peptide_i][frag_i], isotope_isolation_efficiencies[peptide_i])
                            iso_dist.normalize_to_total()
                            mass2dist[frag_masses[peptide_i, frag_i]] = iso_dist
                        else:       
                            iso_dist = mass2dist[frag_masses[peptide_i, frag_i]]

                        intensities_iso[peptide_i][frag_i] *= iso_dist.intensities
                        
                        if return_iso:
                            charge = self.getChargeFromAnnot(annotations[peptide_i][frag_i][0])
                            for iso in range(1,5):
                                if intensities_iso[peptide_i][frag_i][iso] > 0:
                                    if iso == 1:
                                        annotations[peptide_i][frag_i][iso] += "+i"
                                    else:
                                        annotations[peptide_i][frag_i][iso] += "+" + str(iso) + "i"
                                    mzs[peptide_i][frag_i][iso] = mzs[peptide_i][frag_i][iso-1] + self.charge2isoStep[charge]
                                else:
                                    annotations[peptide_i][frag_i][iso] = ""
                                    intensities_iso[peptide_i][frag_i][iso] = -1
                                    mzs[peptide_i][frag_i][iso] = -1
                        else:
                             for iso in range(1,5):
                                annotations[peptide_i][frag_i][iso] = ""
                                intensities_iso[peptide_i][frag_i][iso] = -1
                                mzs[peptide_i][frag_i][iso] = -1  
                        
            intf = pb_utils.Tensor("intensities_iso", intensities_iso.reshape(intensities_iso.shape[0], -1))
            af = pb_utils.Tensor("annotations", annotations.reshape(annotations.shape[0], -1))
            mf = pb_utils.Tensor("mz", mzs.reshape(mzs.shape[0], -1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[intf, af, mf]))
        
        return responses
            
    def finalize(self):
        pass
    
    def getChargeFromAnnot(self, annot):
        entry, _, z = annot.rpartition("^")
        if entry != "": return int(z)
        return 1
            
        
    
