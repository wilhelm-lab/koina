#!/bin/sh
source ~/.bashrc
pipx install poetry
pipx install nox
pipx inject nox nox-poetry
pipx ensurepath
poetry install --with develop -C ./clients/python/
cd web
npm install
touch /tmp/done_setup # For healthcheck
sleep infinity
