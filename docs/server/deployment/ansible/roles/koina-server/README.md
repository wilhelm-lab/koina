# Ansible role: koina-server

Role to provision and configure a Koina inference server.

This role installs and configures the components required to run a Koina server (Docker + NVIDIA runtime, Triton model repository deployment steps, service configuration). It is intended for use from the repository's Ansible playbook [koina-server.yml](https://github.com/wilhelm-lab/koina/tree/main/docs/server/deployment/ansible/koina_server.yml).

## Features
- Deploys Koina container with GPU support.

## Requirements
- A target host with sudo privileges.
- Internet access to download packages and model artifacts.
- Docker, NVIDIA Container Toolkit, NVIDIA drivers.
- Other roles in the ansible roles directory as well as the [koina-server.yml](https://github.com/wilhelm-lab/koina/tree/main/docs/server/deployment/ansible/koina_server.yml) playbook.

## Role variables
Define role variables in your playbook or inventory group_vars/host_vars. Typical variables include (examples only — adjust for your environment):

- koina_container_name: "koina-server"
- koina_container_dir: ""
- koinarpc_docker_port: 8500
- koinahttp_docker_port: 8501
- koinametrics_docker_port: 8502
- koina_shm_size: '8gb'

(Note: Adapt the variables to your specific needs and also the Docker Compose template in the role's templates/ directory.)
