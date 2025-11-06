import json
import triton_python_backend_utils as pb_utils
import numpy as np
from isotopes import IsotopeSplineDB
from pyteomics.auxiliary import _nist_mass

C13C12_MASSDIFF_U = _nist_mass["C"][13][0] - _nist_mass["C"][12][0]


def decode_bytes(x: bytes) -> str:
    return x.decode("utf-8")


vectorized_decode = np.vectorize(decode_bytes)


class TritonPythonModel:
    def __init__(self):
        self.isotopeSplineDB = None
        self.charge2isoStep = None

    def initialize(self, args):
        base_path = "Altimeter/Altimeter_2024_reisotope/"
        self.isotopeSplineDB = IsotopeSplineDB(
            base_path + "IsotopeSplines_10kDa_21isotopes.xml"
        )
        self.charge2isoStep = [C13C12_MASSDIFF_U / z for z in range(1, 10)]
        self.charge2isoStep.insert(0, 0)

    def execute(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())

            return_iso = (
                "return_non_monoisotopes" not in params
                or params["return_non_monoisotopes"] == "True"
            )

            prec_masses = pb_utils.get_input_tensor_by_name(
                request, "precursor_masses"
            ).as_numpy()

            frag_masses = pb_utils.get_input_tensor_by_name(
                request, "fragment_masses"
            ).as_numpy()

            prec_sulfurs = pb_utils.get_input_tensor_by_name(
                request, "precursor_sulfurs"
            ).as_numpy()

            frag_sulfurs = pb_utils.get_input_tensor_by_name(
                request, "fragment_sulfurs"
            ).as_numpy()

            intensities = pb_utils.get_input_tensor_by_name(
                request, "intensities"
            ).as_numpy()

            isotope_isolation_efficiencies = pb_utils.get_input_tensor_by_name(
                request, "isotope_isolation_efficiencies"
            ).as_numpy()

            annotations = pb_utils.get_input_tensor_by_name(
                request, "annotations"
            ).as_numpy()

            mzs = pb_utils.get_input_tensor_by_name(request, "mz").as_numpy()

            intensities_iso = np.expand_dims(intensities, 2).repeat(5, axis=2)
            mzs = np.expand_dims(mzs, 2).repeat(5, axis=2)

            annotations = vectorized_decode(annotations).astype(np.object_)
            annotations = np.expand_dims(annotations, 2).repeat(5, axis=2)

            for peptide_i in range(annotations.shape[0]):
                mass2dist = {}
                for frag_i in range(annotations.shape[1]):
                    if frag_sulfurs[peptide_i][frag_i] < 0:
                        continue

                    iso_dist = self._get_iso_dist(
                        mass2dist,
                        peptide_i,
                        frag_i,
                        prec_masses,
                        frag_masses,
                        prec_sulfurs,
                        frag_sulfurs,
                        isotope_isolation_efficiencies,
                    )

                    intensities_iso[peptide_i][frag_i] *= iso_dist.intensities

                    self._assign_isotope_annotations(
                        intensities_iso,
                        annotations,
                        mzs,
                        peptide_i,
                        frag_i,
                        return_iso,
                    )

            intf = pb_utils.Tensor(
                "intensities_iso", intensities_iso.reshape(intensities_iso.shape[0], -1)
            )
            af = pb_utils.Tensor(
                "annotations", annotations.reshape(annotations.shape[0], -1)
            )
            mf = pb_utils.Tensor("mz", mzs.reshape(mzs.shape[0], -1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[intf, af, mf]))

        return responses

    def finalize(self):
        pass

    def getChargeFromAnnot(self, annot):
        entry, _, z = annot.rpartition("^")
        if entry != "":
            return int(z)
        return 1

    def _get_iso_dist(
        self,
        mass2dist,
        peptide_i,
        frag_i,
        prec_masses,
        frag_masses,
        prec_sulfurs,
        frag_sulfurs,
        isotope_isolation_efficiencies,
    ):
        mass = frag_masses[peptide_i, frag_i]
        iso_dist = mass2dist.get(mass)
        if iso_dist is None:
            if prec_masses[peptide_i][0] == mass:
                iso_dist = (
                    self.isotopeSplineDB.estimate_for_precursor_from_weights_and_sulfur(
                        prec_masses[peptide_i][0],
                        4,
                        prec_sulfurs[peptide_i][0],
                        isotope_isolation_efficiencies[peptide_i],
                    )
                )
            else:
                iso_dist = (
                    self.isotopeSplineDB.estimate_for_fragment_from_weights_and_sulfur(
                        prec_masses[peptide_i][0],
                        mass,
                        0,
                        4,
                        prec_sulfurs[peptide_i][0],
                        frag_sulfurs[peptide_i][frag_i],
                        isotope_isolation_efficiencies[peptide_i],
                    )
                )
            iso_dist.normalize_to_total()
            mass2dist[mass] = iso_dist
        return iso_dist

    def _assign_isotope_annotations(
        self,
        intensities_iso,
        annotations,
        mzs,
        peptide_i,
        frag_i,
        return_iso,
    ):
        if not return_iso:
            for iso in range(1, 5):
                annotations[peptide_i][frag_i][iso] = ""
                intensities_iso[peptide_i][frag_i][iso] = -1
                mzs[peptide_i][frag_i][iso] = -1
            return

        charge = self.getChargeFromAnnot(annotations[peptide_i][frag_i][0])
        for iso in range(1, 5):
            if intensities_iso[peptide_i][frag_i][iso] > 0:
                annotations[peptide_i][frag_i][iso] += "+i" if iso == 1 else f"+{iso}i"
                mzs[peptide_i][frag_i][iso] = (
                    mzs[peptide_i][frag_i][iso - 1] + self.charge2isoStep[charge]
                )
            else:
                annotations[peptide_i][frag_i][iso] = ""
                intensities_iso[peptide_i][frag_i][iso] = -1
                mzs[peptide_i][frag_i][iso] = -1
