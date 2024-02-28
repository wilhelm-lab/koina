#!/bin/sh
source ~/.bashrc
poetry install --with develop -C ./clients/python/
pyenv --version
npm --version
sleep infinity
