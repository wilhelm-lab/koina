# Ansible role: nvidia-container-toolkit

Role to install and configure the NVIDIA Container Toolkit (nvidia-docker / nvidia-container-runtime) on target hosts (Ubuntu only) so Docker/containers can access NVIDIA GPUs.

This role prepares the host by adding NVIDIA package repositories, installing the runtime/toolkit packages, and configuring Docker to use the NVIDIA runtime when requested.

## Features
- Adds NVIDIA apt repositories
- Installs nvidia-container-toolkit
- Configures Nvidia runtime for Docker and restarts Docker daemon

## Requirements
- Sudo/root privileges on target hosts
- Docker engine already installed (this role does not install Docker itself)
- Internet access to fetch NVIDIA packages and GPG keys
- Compatible NVIDIA driver installed on host (driver installation is out of scope)

## Example playbook
```yaml
- hosts: gpu-servers
  become: true
  roles:
    - nvidia-container-toolkit
```

Run with your inventory:
```bash
ansible-playbook -i inventory.ini playbooks/setup-gpu.yml --ask-become-pass
```
