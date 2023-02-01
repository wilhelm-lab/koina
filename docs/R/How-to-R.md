
# Prerequisite

Understand the [Kserve API](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-request-examples)

## Get the model description
```
curl eubic2023.external.msaid.io:8501/v2/models/ms2pip_ensemble
```

which results in
```jq
{
  "name": "ms2pip_ensemble",
  "versions": [
    "1"
  ],
  "platform": "ensemble",
  "inputs": [
    {
      "name": "proforma_ensemble",
      "datatype": "BYTES",
      "shape": [
        -1,
        -1
      ]
    }
  ],
  "outputs": [
    {
      "name": "model_20210416_HCD2021_B_output",
      "datatype": "FP32",
      "shape": [
        -1,
        1
      ]
    }
  ]
}
```

## Get the model prediction
```
curl -X POST \
  http://eubic2023.external.msaid.io:8501/v2/models/ms2pip_ensemble/infer \
  -d '{
	"id" : "42",
  "inputs" : [ {
  "name" : "proforma_ensemble",
  "shape" : [ 1,1 ],
  "datatype"  : "BYTES",
  "data" : ["AAAAAA/2"]
} ],
"outputs" : [
    {
      "name" : "model_20210416_HCD2021_B_output"
    }
  ]
}'
```

which results in
```json
{
    "id": "42",
    "model_name": "ms2pip_ensemble",
    "model_version": "1",
    "parameters": {
        "sequence_id": 0,
        "sequence_start": false,
        "sequence_end": false
    },
    "outputs": [
        {
            "name": "model_20210416_HCD2021_B_output",
            "datatype": "FP32",
            "shape": [
                29
            ],
            "data": [
                -8.882865905761719,
                -3.8637025356292726,
                -3.980400323867798,
                -4.265582084655762,
                -5.151244640350342,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582,
                -8.570317268371582
            ]
        }
    ]
}
```

### Do the prediction in R

```R
require(httr)

headers = c(
  `Content-Type` = 'application/x-www-form-urlencoded'
)

data = '{\n\t"id" : "42",\n  "inputs" : [ {\n  "name" : "proforma_ensemble",\n  "shape" : [ 1,1 ],\n  "datatype"  : "BYTES",\n  "data" : ["AAAAAA/2"]\n} ],\n"outputs" : [\n    {\n      "name" : "model_20210416_HCD2021_B_output"\n    }\n  ]\n}'

res <- httr::POST(url = 'http://eubic2023.external.msaid.io:8501/v2/models/ms2pip_ensemble/infer', httr::add_headers(.headers=headers), body = data)
content(res)
```
