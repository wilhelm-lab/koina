# Koina

## Accessing a public server
### cURL
Here is an example http request using only cURL sending a POST request to with a json body.

```bash
curl "https://koina.wilhelmlab.org/v2/models/Prosit_2019_intensity/infer" \
 --data-raw '
{
  "id": "LGGNEQVTR_GAGSSEPVTGLDAK",
  "inputs": [
    {"name": "peptide_sequences",   "shape": [2,1], "datatype": "BYTES", "data": ["LGGNEQVTR","GAGSSEPVTGLDAK"]},
    {"name": "collision_energies",  "shape": [2,1], "datatype": "FP32",  "data": [25,25]},
    {"name": "precursor_charges",    "shape": [2,1], "datatype": "INT32", "data": [1,2]}
  ]
}'
```
### Python
For examples of how to access models using python you can check out [our OpenAPI documentation ](https://koina.wilhelmlab.org/docs/#overview).

### R
TODO

## Hosting your own server

### Dependencies
Koina depends on [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

You can find an ansible script that installs all dependencies [here](docs/server/).

### How to run it
After installing the dependencies you can pull the docker image and run it. If you have multiple GPUs installed on your server you can choose which one is used by modifying `--gpus '"device=0"'`
When using this docker image you need to accept the terms in the [NVIDIA Deep Learning Container License](NVIDIA_Deep_Learning_Container_License.pdf)
```bash
docker run \
    --gpus '"device=0"' \
    --shm-size 8G \
    --name koina \
    -p 8500-8502:8500-8502 \
    -d \
    --restart unless-stopped \
    ghcr.io/wilhelm-lab/koina:latest
```

If you want to stay up to date with the latest version of Koina we suggest you also deploy containrrr/watchtower.


```bash
docker run \
  -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --restart unless-stopped \
  containrrr/watchtower -i 30
```

## Adding your own model
You can find a detailed description under [docs/](docs/README.md). If you run into any issues please open an issue on GitHub.