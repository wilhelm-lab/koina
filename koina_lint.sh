#!/bin/bash
cd /workspace/koina/
poetry run -C clients/python pylint --recursive=y . $@
