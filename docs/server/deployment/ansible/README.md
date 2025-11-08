# Ansible: Koina server deployment

This directory contains the Ansible playbooks and roles used to provision Koina inference servers. The main orchestration is performed by the `koina_server.yml` playbook and its included roles. Use this README as a quick reference for installing role requirements, running the playbook, and where to find common variables and templates.

_NOTE: The Ansible playbook to deploy KOINA server will fetch the TLS/SSL certificates automatically using Certbot for the specified domains from Let's Encrypt._

## Quick start

1. From the repo root, install collections and roles listed in requirements.yml:
```bash
cd docs/server/deployment/ansible
ansible-galaxy install -r requirements.yml
```

2. Update the inventory file `hosts` and the Nginx templates in `templates/nginx/` as needed for your environment and also the variables in `koina_server.yml` (domain names, email address, paths, etc.).

3. Run the main playbook (example inventory file and variables live in the repo):
```bash
ansible-playbook koina_server.yml --ask-become-pass
```
Adjust the inventory path, extra-vars, and become options for your environment.

## What is included

- koina_server.yml — main playbook that composes the server setup (Docker, KOINA/Triton, services, etc.).
- requirements.yml — pinned Ansible roles/collections required by the playbooks.
- roles/ — local roles used by the playbook (examples: koina-server, nvidia-container-toolkit, etc.).
- templates/ — templates used by roles (includes nginx vhost templates for the nginx role).
- defaults/ in each role — role-level default variables (recommended place to review defaults before overriding).
- tasks/, handlers/, files/ — standard Ansible role layout for each role.

## Variables & configuration

- Primary variables and the flow of configuration are defined in `koina_server.yml` and the defaults files of each role (`roles/<role>/defaults/main.yml`).
- Override values per-host or per-group using `host_vars/` or `group_vars/` or pass via `-e` on the command line.
- Ensure to review and modify accordingly:
    - `koina_server.yml` for the overall orchestration and variable flow.
    - `roles/*/defaults/main.yml` to find complete variable names and defaults.

## Nginx templates

- Nginx virtual host templates live under `templates/nginx` and are consumed by the `geerlingguy.nginx` role. Modify or copy these templates to customize upstreams, SSL, or proxy rules before running the playbook.

Example templates path:
```
templates/nginx/koina.conf.j2
templates/nginx/koinarpc.conf.j2
```

### TLS/SSL Certificates
- The playbook uses the `geerlingguy.certbot` role to automatically obtain and renew TLS/SSL certificates from Let's Encrypt using Certbot.
- Ensure that the domain names specified in the variables are correctly pointed to your server's IP address before running the playbook.
- Port 80 must be open and accessible for the HTTP-01 challenge used by Let's Encrypt.
- The email address provided in the variables is used for important account notifications from Let's Encrypt.