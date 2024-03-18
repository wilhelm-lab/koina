# USI proxy

## Development
Run development server with `npm run srv`

## Build for productions
`npm run build`

## Test cases

```bash
# Rquests to the non use endpoint are unaffected
curl serving:8501/v2/models/Prosit_2019_intensity/config
echo -e '\n--------------------------------------\n'
curl "serving:8501/v2/models/Prosit_2019_intensity/infer" \
        --data-raw '
        {
        "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
        "inputs": [
            {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
            {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
            {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
        ]
        }'
echo -e '\n--------------------------------------\n'

# you can specify all parameters in the query string
curl serving:8501/v2/models/Prosit_2019_intensity/usi?peptide_sequences=VLHPLEGAVVIIFK\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

# Works for different models as well
curl serving:8501/v2/models/ms2pip_2021_HCD/usi?peptide_sequences=VLHPLEGAVVIIFK\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'
curl serving:8501/v2/models/AlphaPept_ms2_generic/usi?peptide_sequences=VLHPLEGAVVIIFK\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

# You can also specify the USI which will be used to extract the necessary information
curl serving:8501/v2/models/AlphaPept_ms2_generic/usi?usi=mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2\&collision_energies=30\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

# If both the USI and the peptide sequence are specified, the peptide sequence will be used
curl serving:8501/v2/models/AlphaPept_ms2_generic/usi?usi=mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2\&peptide_sequences=AAAAAAAAAAAAA\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

# Non intensity models return an error
curl serving:8501/v2/models/Prosit_2019_irt/usi?peptide_sequences=VLHPLEGAVVIIFK\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

# compressed curl
curl --compressed serving:8501/v2/models/AlphaPept_ms2_generic/usi?usi=mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2\&peptide_sequences=AAAAAAAAAAAAA\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'

curl --compressed "serving:8501/v2/models/Prosit_2019_intensity/infer" \
        --data-raw '
        {
        "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
        "inputs": [
            {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
            {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
            {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
        ]
        }'
echo -e '\n--------------------------------------\n'

# Failing curl
curl --compressed "serving:8501/v2/models/Prosit_2020_intensity_TMT/infer" \
        --data-raw '
        {
        "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
        "inputs": [
            {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
            {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
            {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
        ]
        }'
echo -e '\n--------------------------------------\n'

# Failed USI
curl --compressed serving:8501/v2/models/Prosit_2020_intensity_TMT/usi?usi=mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2\&peptide_sequences=AAAAAAAAAAAAA\&collision_energies=30\&precursor_charges=2\&instrument_types=LUMOS
echo -e '\n--------------------------------------\n'
```