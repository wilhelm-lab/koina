# koinapy

Python client for [Koina](https://koina.wilhelmlab.org/), a community-driven
service that hosts machine learning models for proteomics and makes them
available over an open API.

`koinapy` talks to a Koina server over gRPC and returns predictions as pandas
DataFrames, so you can go from peptide sequences to predicted spectra, retention
times or collisional cross sections in a few lines.

## Installation

```bash
pip install koinapy
```

Or, if you manage your environments with conda, from
[Bioconda](https://bioconda.github.io/recipes/koinapy/README.html):

```bash
conda install -c conda-forge -c bioconda koinapy
```

The Bioconda recipe also produces a
[Biocontainer](https://quay.io/repository/biocontainers/koinapy?tab=tags), which
is convenient if you call Koina from a workflow manager such as Nextflow,
Snakemake or Galaxy. Tags are `<version>--<build>`, for example:

```bash
docker pull quay.io/biocontainers/koinapy:0.0.11--pyhdfd78af_0
```

## Usage

By default the client connects to the public Koina server at
`koina.wilhelmlab.org:443`, so no setup or API key is needed:

```python
import numpy as np
import pandas as pd
from koinapy import Koina

model = Koina("Prosit_2020_intensity_HCD")

inputs = pd.DataFrame({
    "peptide_sequences": np.array(["AAAAAKAKM[UNIMOD:35]", "LPQLC[UNIMOD:4]TDLK"]),
    "precursor_charges": np.array([1, 2]),
    "collision_energies": np.array([25.0, 30.0]),
})

predictions = model.predict(inputs)
```

`predictions` is a DataFrame with one row per predicted fragment ion, carrying
the inputs alongside the `intensities`, `mz` and `annotation` columns.

Notes on usage:

- Each model takes only the inputs it needs. You can keep extra columns in the
  DataFrame when comparing several models against the same input.
- If you are unsure what a model expects, inspect `model.model_inputs`.
- Pass `df_output=False` to get a dict of raw numpy arrays instead of a
  DataFrame.
- To use your own Koina deployment, pass `server_url` (and `ssl=False` for a
  plain-HTTP server).

## Documentation

The full list of available models, their inputs and outputs, and ready-to-use
code samples are in the [Koina documentation](https://koina.wilhelmlab.org/docs).

An R client, [koinar](https://github.com/wilhelm-lab/koinar), is available from
[Bioconductor](https://bioconductor.org/packages/koinar/).

## Citation

If you use Koina in your research, please cite the model you used as well as:

```bibtex
@article{Lautenbacher2025Koina,
  title    = {Koina: Democratizing machine learning for proteomics research},
  author   = {Lautenbacher, Ludwig and Yang, Kevin L. and Kockmann, Tobias and
              Panse, Christian and Gabriel, Wassim and Bold, Dulguun and
              Kahl, Elias and Chambers, Matthew and MacLean, Brendan X. and
              Li, Kai and Yu, Fengchao and Searle, Brian C. and
              Wilburn, Damien Beau and Shahneh, Mohammad Reza Zare and
              Hong, Yuhui and Tang, Haixu and Wang, Mingxun and
              Gabriels, Ralf and Bouwmeester, Robbin and Devreese, Robbe and
              Angelis, Jesse and Sabid{\'o}, Eduard and Schmidt, Tobias K. and
              Nesvizhskii, Alexey I. and Wilhelm, Mathias},
  journal  = {Nature Communications},
  volume   = {16},
  number   = {1},
  pages    = {9933},
  year     = {2025},
  month    = nov,
  doi      = {10.1038/s41467-025-64870-5},
  issn     = {2041-1723},
  url      = {https://doi.org/10.1038/s41467-025-64870-5},
}
```
