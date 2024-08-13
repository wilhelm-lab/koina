Test 3dmolms using randomly initialized arrays: 

```bash
python test_3dmolms.py
```

Test 3dmolms using `*.npy` files: 

```bash
python test_3dmolms_from_arr.py
```

Before testing 3dmolms with user inputs, please install the following packages:

```bash
conda install conda-forge::rdkit
pip install pyyaml
```

Test 3dmolms with user inputs: 

```bash
python test_3dmolms_from_raw.py
```

To do: 

1. Handle the raised errors.
2. Expand preprocessing to handle lists of inputs rather than single-item inputs. 