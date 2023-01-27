# README

This provides a starting point to retrieve online prediction for [ms2pip](https://github.com/compomics/ms2pip_c) models.

## Conversion of ms2pip models

1. Download the model of choice from the [ms2pip model repository](https://genesis.ugent.be/uvpublicdata/ms2pip/)
2. Convert the model
```python
import xgboost as xgb
bst = xgb.Booster()
bst.load_model("model_20210416_HCD2021_B.xgboost")
bst.save_model("xgboost.json")
```
  This conversion is necessary as ms2pip provides an older file format as accessible within the latest verion of Triton server.
  

