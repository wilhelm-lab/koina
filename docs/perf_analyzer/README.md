# Performance Analyzer utility
The performance analyzer kann be used to benchmark model inference speed

## Explicit inputs
An example for a ensemble model where inputs needs to be specified.
```bash
perf_analyzer -u 'serving:8501' -m Prosit_2019_intensity --concurrency-range 1:4 -b 10 --input-data docs/perf_analyzer/P
rosit_2019_intensity_input.json --shape 'peptide_sequences:1'
```

## Inferred inputs
An example for a tensorflow model where random inpus can be inferred.
```bash
perf_analyzer -u 'serving:8501' -m Prosit_2019_intensity_core --concurrency-range 1:4 -b 1000
```

## Simultaneous load
The `perfTest.sh` script is used to test stability of an instance under heavy inference load for multiple models.