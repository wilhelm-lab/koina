from operator import itemgetter


class Unimod:
    OBO_FILE_URL = "https://www.unimod.org/obo/unimod.obo"

    def __init__(
        self,
        obo_file_url=OBO_FILE_URL,
        download_folder="./",
        no_modification_string="-",
        reverse_lookup_key="name",
    ):
        self.obo_file_url = obo_file_url
        self.download_folder = download_folder
        self.filename = "unimod.obo"
        self.no_modification_string = no_modification_string
        self.reverse_lookup_key = reverse_lookup_key

        self.local_obo_filepath = "/models/Deeplc/Deeplc_Preprocess_AC/1/unimod.obo"
        # self._download_file()
        self._parse_file_build_dicts()

    def _download_file(self):
        import requests
        import os

        target_folder = os.path.join(self.download_folder)
        os.makedirs(target_folder, exist_ok=True)

        target_filepath = os.path.join(target_folder, self.filename)

        if not os.path.exists(target_filepath):
            r = requests.get(Unimod.OBO_FILE_URL, allow_redirects=True)
            open(target_filepath, "wb").write(r.content)

        self.local_obo_filepath = target_filepath

    def _parse_file_build_dicts(self):
        with open(self.local_obo_filepath) as f:
            file_content = f.read()

        # extract version information
        term_list = file_content.split("[Term]")
        self.version_information = term_list.pop(0)

        # build a list of modifications extracted from the file
        unimod_database = [
            self._parse_unimod_modification_from_substring(term) for term in term_list
        ]

        self.unimod_database = unimod_database
        self._build_unimod_lookup_dicts(unimod_database)

    def _build_unimod_lookup_dicts(self, unimod_database):
        self.unimod_db_dict = {}

        for entry in unimod_database:
            idx, namex, defx = entry["id"], entry["name"], entry["def"]
            self.unimod_db_dict[idx] = {"name": namex, "def": defx}

            # add meta data to the dict of the corresponding modification (index)
            if "xref" in entry.keys():
                self.unimod_db_dict[idx].update(
                    self._parse_modification_metadata(entry["xref"])
                )

        #  reverse lookup using the name or the provided argument
        self.reverse_unimod_db_dict = {
            v[self.reverse_lookup_key]: k for k, v in self.unimod_db_dict.items()
        }

    def _parse_modification_metadata(self, xref):
        meta_dict = dict([tuple(e.split(" ", maxsplit=1)) for e in xref])
        return meta_dict

    def _parse_unimod_modification_from_substring(self, unimod_string):
        lines = unimod_string.split("\n")
        d = {}
        lines = [l for l in lines if l.strip() != ""]
        for l in lines:
            k, v = l.split(": ", maxsplit=1)
            if d.get(k, 0):
                if not isinstance(d[k], list):
                    d[k] = [d[k]]
                d[k].append(v)
            else:
                d[k] = v
        return d

    def lookup_sequence(self, split_seq, keys_to_lookup=("name")):
        decoded_mod_seq = []

        for aa in split_seq:
            current_aa = []
            if "[" in aa:
                splitted = aa.split("[")
                amino_acid = splitted[0]

                mod = self.unimod_db_dict[splitted[-1].split("]")[0]][keys_to_lookup]
                current_aa.extend([amino_acid, mod])
            else:
                current_aa.extend([aa, self.no_modification_string])
            decoded_mod_seq.append(current_aa)
        return decoded_mod_seq

    def lookup_sequence_m(self, split_seq, keys_to_lookup=("name")):
        decoded_mod_seq = []

        for aa in split_seq:
            current_aa = []
            if "[" in aa:
                splitted = aa.split("[")
                amino_acid = splitted[0]

                mods = []
                for unimod_string in splitted[1:]:
                    unimod_string = unimod_string.replace("]", "")

                    keys_getter = itemgetter(*keys_to_lookup)
                    if unimod_string == "UNIMOD:5634":
                        mod = '''"5634"'''
                    else:
                        mod = keys_getter(self.unimod_db_dict[unimod_string])
                    mods.append(mod)
                current_aa.extend([amino_acid, *mods])
            else:
                current_aa.extend([aa, self.no_modification_string])
            decoded_mod_seq.append(current_aa)
        return decoded_mod_seq


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
