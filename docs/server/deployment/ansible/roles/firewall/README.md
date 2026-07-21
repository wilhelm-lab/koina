# Ansible role: firewall

Role to install and configure the required firewall on target hosts (Ubuntu only).

## Features
- Configures UFW with predefined rules
- Denies all incoming connections by default
- Allows all outgoing connections by default
- Opens specific ports for SSH, HTTP, HTTPS

## Requirements
- Sudo/root privileges on target hosts

## Example playbook
```yaml
- hosts: all
  become: true
  roles:
    - firewall
```
