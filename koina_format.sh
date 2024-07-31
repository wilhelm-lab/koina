#!/bin/bash
source ~/.bashrc
cd /workspace/koina/
poetry run -C clients/python black . $@
