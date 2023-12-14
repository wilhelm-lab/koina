
# Setup of a server for dlomix-serving

The server should be provisioned via [ansible](https://www.ansible.com/). This can be installed via `pip install ansible`.

The installation of GPU drivers is complicated and this setup uses the [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software) by lambdalabs. This stack provides a convenient wrapper around all kernel updates and gpu drivers, but works only on Ubuntu 22.04 LTS, 20.04 LTS, 18.04 LTS, and 16.04 LTS.

If you want to use any other Linux distribution, please read up on how to install

- [cuDNN](https://developer.nvidia.com/cudnn)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [Nvidia Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#introduction)


## Ansible

This installation also takes care of installing `docker` and `nvidia-container-toolkit`

### Remote installation

1. Adjust `development.txt` to point to your server
2. Run it via
```shell
ansible-playbook ./gpu-driver.yaml -i development.txt --ask-become-pass
```

### Local installation

1. Checkout repo on your server
2. Execute it via `ansible-playbook --connection=local 127.0.0.1, gpu-driver.yaml`
