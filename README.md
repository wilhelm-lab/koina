# dlomix-serving

## Accessing a public server
### curl
TODO

### Python
See the examples in [docs/]()

### R
TODO


## Hosting your own server

### Dependencies
dlomix-serving depends on [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

### How to run it
After installing the dependencies you can pull the docker image and run it with. 
```
docker run \
    --gpus all \
    --shm-size 2G \
    -p 8500:8500 \
    -p 8501:8501 \
    -d \
    ghcr.io/wilhelm-lab/dlomix-serving:latest
```

## Adding your own model
Additionally to [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) we suggest you install [docker-compose](https://docs.docker.com/desktop/install/linux-install/)
