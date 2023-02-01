
# Setup of a server for dlomix-serving

The server should be provisioned via [ansible](https://www.ansible.com/). This can be installed via `pip install ansible`.

Run it via

```shell
ansible-playbook ./gpu-driver.yaml -i development.txt --ask-become-pass
```

