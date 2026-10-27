"""Offline tests for the Prosit-PTM atom-count preprocessing (no Triton server needed).

Both preprocessing models are imported with a stub for triton_python_backend_utils and with Unimod
pointed at the unimod.obo in this repository.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
PROSIT = REPO / "models" / "Prosit"
OBO = REPO / "models" / "Deeplc" / "Deeplc_Preprocess_AC" / "1" / "unimod.obo"


def _load(model_dir, name):
    sys.modules.setdefault("triton_python_backend_utils", types.ModuleType("pb_utils"))
    src = PROSIT / model_dir / "1"
    spec = importlib.util.spec_from_file_location(
        "modifications", src / "modifications.py"
    )
    modifications = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modifications)
    original_init = modifications.Unimod.__init__

    def init(self, *args, **kwargs):
        # parse the repository copy instead of the container path
        self.local_obo_filepath = str(OBO)
        self.no_modification_string = "-"
        self.reverse_lookup_key = "name"
        self._parse_file_build_dicts()

    modifications.Unimod.__init__ = init
    sys.modules["modifications"] = modifications
    spec = importlib.util.spec_from_file_location("zero_shot", src / "zero_shot.py")
    zero_shot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zero_shot)
    sys.modules["zero_shot"] = zero_shot
    spec = importlib.util.spec_from_file_location(name, src / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    modifications.Unimod.__init__ = original_init
    return module


gain_model = _load("Prosit_Preprocess_ac_gain", "prosit_ac_gain")
loss_model = _load("Prosit_Preprocess_ac_loss", "prosit_ac_loss")
zero_shot = sys.modules["zero_shot"]


def _row(model, key):
    """Atom-count row of the modified residue in a short peptide carrying `key`."""
    residue, record_id = key.split("_")
    if residue == "":
        return model.get_ac(f"[UNIMOD:{record_id}]-PEPTIDEK", None)[0]
    return model.get_ac(f"PEP{residue}[UNIMOD:{record_id}]K", None)[4]


def _expected(table_value):
    return (
        [1 + c for c in zero_shot.parse_composition(table_value)]
        if table_value.strip()
        else [1] * 6
    )


@pytest.mark.parametrize("key", sorted(gain_model.dict_ptm_atom_count_gain))
def test_listed_pairs_are_encoded_from_the_tables(key):
    residue, record_id = key.split("_")
    if f"UNIMOD:{record_id}" not in gain_model.unimod.unimod_db_dict:
        pytest.skip(
            f"UNIMOD:{record_id} is not in unimod.obo, so the sequence cannot be parsed"
        )
    np.testing.assert_array_equal(
        _row(gain_model, key), _expected(gain_model.dict_ptm_atom_count_gain[key])
    )
    np.testing.assert_array_equal(
        _row(loss_model, key), _expected(loss_model.dict_ptm_atom_count_loss[key])
    )


# _411 (Phenylisocyanate) is listed with the same counts as propionyl; UNIMOD says H(5) C(7) N O.
KNOWN_TABLE_EXCEPTIONS = {"_411"}


@pytest.mark.parametrize("key", sorted(gain_model.dict_ptm_atom_count_gain))
def test_listed_gain_minus_loss_is_the_unimod_delta(key):
    """The rule the zero-shot encoding rests on, checked against every listed pair."""
    record_id = key.split("_")[1]
    entry = gain_model.unimod.unimod_db_dict.get(f"UNIMOD:{record_id}")
    if entry is None or key in KNOWN_TABLE_EXCEPTIONS:
        pytest.skip("no UNIMOD composition, or a known table exception")
    gain = _expected(gain_model.dict_ptm_atom_count_gain[key])
    loss = _expected(loss_model.dict_ptm_atom_count_loss[key])
    delta = zero_shot.parse_composition(entry["delta_composition"].strip('"'))
    assert [g - l for g, l in zip(gain, loss)] == delta


@pytest.mark.parametrize(
    "key, gain, loss",
    [
        # H C N O P S; phosphocysteine = thiol + HPO3
        ("C_21", [2, 0, 0, 3, 1, 1], [1, 0, 0, 0, 0, 1]),
        # crotonyl-K = amine + C4H4O
        ("K_1363", [6, 4, 1, 1, 0, 0], [2, 0, 1, 0, 0, 0]),
        # succinyl-K = amine + C4H4O3
        ("K_64", [6, 4, 1, 3, 0, 0], [2, 0, 1, 0, 0, 0]),
        # N-terminal dimethyl = amine + C2H4
        ("_36", [6, 2, 1, 0, 0, 0], [2, 0, 1, 0, 0, 0]),
    ],
)
def test_unlisted_pairs_are_encoded_zero_shot(key, gain, loss):
    assert key not in gain_model.dict_ptm_atom_count_gain
    np.testing.assert_array_equal(_row(gain_model, key), [1 + g for g in gain])
    np.testing.assert_array_equal(_row(loss_model, key), [1 + l for l in loss])


@pytest.mark.parametrize(
    "sequence",
    [
        "PEPG[UNIMOD:21]K",  # G has no attachment group
        "PEPY[UNIMOD:340]K",  # Bromo: delta composition H(-1) Br
    ],
)
def test_unsupported_pairs_raise_key_error_in_both_models(sequence):
    for model in (gain_model, loss_model):
        with pytest.raises(KeyError):
            model.get_ac(sequence, None)


def test_heavy_arginine_no_longer_crashes_the_gain_model():
    # R_267 is listed with an empty gain string, which used to raise AttributeError
    np.testing.assert_array_equal(_row(gain_model, "R_267"), [1] * 6)
