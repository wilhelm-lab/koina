# Ansible role: nvidia-driver

Role to install the Nvidia driver on target hosts (Ubuntu only).

## Features
- Installs the latest Nvidia GPU driver and utilities
- Reboots the system if a new driver is installed
- Verifies the installation with `nvidia-smi`

## Requirements
- Sudo/root privileges on target hosts

## Example playbook
```yaml
- hosts: all
  become: true
  roles:
    - nvidia-driver
```
