import logging
import base64
import struct
from xml.etree import ElementTree as ET
import numpy as np


class CubicSpline:
    def __init__(self, a, b, c, d, x):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = x

    def eval(self, x):
        xx = x - self.x
        index = max(0, int(xx // 1000))
        xx -= index * 1000
        coeffs = (self.d[index], self.c[index], self.b[index], self.a[index])
        result = 0.0
        for coeff in coeffs:
            result = result * xx + coeff
        return max(0, result)

    def in_bounds(self, x):
        return x <= self.x


class IsotopeSplineXMLParser:
    def parse(self, spline_path):
        try:
            tree = ET.parse(spline_path)
        except ET.ParseError as exc:
            logging.error("Malformed XML in %s: %s", spline_path, exc)
            return None
        except OSError as exc:
            logging.error("Could not open %s: %s", spline_path, exc)
            return None
        root = tree.getroot()
        return self.parse_document(root)

    def parse_document(self, root):
        num_models = 5  # int(root.get("maxIsotopeDepth"))
        num_models_sulfur = 21
        models = np.empty((num_models_sulfur, num_models), dtype=np.object_)
        for model_ele in root.findall("model"):
            num_sulfur = -1
            isotope = 0

            for attr_name, attr_value in model_ele.attrib.items():
                if attr_name == "isotope":
                    isotope = int(attr_value)
                elif attr_name == "S":
                    num_sulfur = int(attr_value)

            if isotope >= num_models:
                continue
            if num_sulfur >= num_models_sulfur or num_sulfur < 0:
                continue

            models[num_sulfur][isotope] = self.parse_model(model_ele)

        return models

    def parse_model(self, model_ele):
        first_knot = self.decode_double_list(self.get_text_value(model_ele, "knots"))[0]
        coefficients = self.decode_double_list(
            self.get_text_value(model_ele, "coefficients")
        )

        a, b, c, d = np.zeros((4, len(coefficients) >> 2), dtype=np.float32)
        for i in range(0, len(coefficients), 4):
            a[i >> 2] = coefficients[i]
            b[i >> 2] = coefficients[i + 1]
            c[i >> 2] = coefficients[i + 2]
            d[i >> 2] = coefficients[i + 3]

        return CubicSpline(a, b, c, d, first_knot)

    def decode_double_list(self, encoded):
        decoded = base64.b64decode(encoded)
        return struct.unpack("<" + "d" * (len(decoded) // 8), decoded)

    def get_text_value(self, element, tag_name):
        tag = element.find(tag_name)
        return tag.text if tag is not None else None


class IsotopeDistribution:
    def __init__(self, max_isotope=4):
        self.intensities = np.zeros(max_isotope + 1, dtype=np.float32)

    def size(self):
        return len(self.intensities)

    def normalize_to_base_peak(self):
        self.normalize_to_value(np.max(self.intensities))

    def normalize_to_value(self, value):
        if value != 0:
            self.intensities /= value

    def normalize_to_total(self):
        self.normalize_to_value(np.sum(self.intensities))


class IsotopeSplineDB:
    C13C12_MASSDIFF_U = 1.0033548378

    def __init__(self, spline_path):
        self.read_splines_from_file(spline_path)

    def read_splines_from_file(self, spline_path):
        parser = IsotopeSplineXMLParser()
        self.models = parser.parse(spline_path)

    def estimate_from_peptide_weight_and_sulfur(
        self, mono_mass, max_isotope, num_sulfur
    ):
        if self.in_model_bounds(mono_mass, num_sulfur):
            result = IsotopeDistribution(max_isotope)
            for isotope in range(max_isotope + 1):
                result.intensities[isotope] = self.models[num_sulfur][isotope].eval(
                    mono_mass
                )
            return result

        return IsotopeDistribution(max_isotope)

    def estimate_for_precursor_from_weights_and_sulfur(
        self, mono_peptide_mass, max_isotope, precursor_sulfur, iso2efficiency
    ):

        precursor_iso_dist = self.estimate_from_peptide_weight_and_sulfur(
            mono_peptide_mass, max_isotope, precursor_sulfur
        )

        precursor_iso_dist.intensities = (
            np.array(precursor_iso_dist.intensities) * iso2efficiency
        )

        return precursor_iso_dist

    def estimate_for_fragment_from_weights_and_sulfur(
        self,
        mono_peptide_mass,
        mono_fragment_mass,
        min_isotope,
        max_isotope,
        precursor_sulfur,
        fragment_sulfur,
        iso2efficiency,
    ):

        fragment = self.estimate_from_peptide_weight_and_sulfur(
            mono_fragment_mass, max_isotope, fragment_sulfur
        )

        comp_fragment = self.estimate_from_peptide_weight_and_sulfur(
            mono_peptide_mass - mono_fragment_mass,
            max_isotope,
            precursor_sulfur - fragment_sulfur,
        )

        return self.calc_fragment_isotope_distribution(
            fragment, comp_fragment, min_isotope, max_isotope, iso2efficiency
        )

    def in_model_bounds(self, mono_mass, num_sulfur):
        return (num_sulfur < self.models.shape[0]) and (mono_mass < 10000)

    def calc_fragment_isotope_distribution(
        self, fragment, comp_fragment, min_isotope, max_isotope, iso2efficiency
    ):
        result = IsotopeDistribution(max_isotope)

        for i in range(fragment.size()):
            for isotope in range(min_isotope, max_isotope + 1):
                if isotope >= i and (isotope - i) < comp_fragment.size():
                    result.intensities[i] += (
                        iso2efficiency[isotope] * comp_fragment.intensities[isotope - i]
                    )
            result.intensities[i] *= fragment.intensities[i]

        return result
