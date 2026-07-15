#!/bin/bash
# Build isolated Linux venvs for Garak, PyRIT, and DeepTeam - one per tool so their
# conflicting dependencies never collide with each other or with the worker's own
# Python env. engine/tool_scan.py::_get_tool_python resolves these paths via
# GARAK_PYTHON/PYRIT_PYTHON/DEEPTEAM_PYTHON and invokes them only via subprocess -
# never imports them directly (see worker/tasks.py, worker/Dockerfile).
#
# Mirrors the existing local Windows venvs (tool-venvs/garak-venv, pyrit-venv,
# deepteam-venv) - same tool versions, same directory naming, just Linux paths
# (bin/python instead of Scripts/python.exe).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/tool-venvs"

echo "Building tool-venvs in ${VENV_DIR}"

~/.pyenv/versions/3.12.11/bin/python3 -m venv "${VENV_DIR}/garak-venv"
"${VENV_DIR}/garak-venv/bin/pip" install --upgrade pip
"${VENV_DIR}/garak-venv/bin/pip" install "garak==0.15.1"

~/.pyenv/versions/3.12.11/bin/python3 -m venv "${VENV_DIR}/pyrit-venv"
"${VENV_DIR}/pyrit-venv/bin/pip" install --upgrade pip
"${VENV_DIR}/pyrit-venv/bin/pip" install "pyrit==0.14.0"

~/.pyenv/versions/3.12.11/bin/python3 -m venv "${VENV_DIR}/deepteam-venv"
"${VENV_DIR}/deepteam-venv/bin/pip" install --upgrade pip
"${VENV_DIR}/deepteam-venv/bin/pip" install "deepteam==1.0.7"

echo ""
echo "Done. Set these in the worker's .env:"
echo "  GARAK_PYTHON=${VENV_DIR}/garak-venv/bin/python"
echo "  PYRIT_PYTHON=${VENV_DIR}/pyrit-venv/bin/python"
echo "  DEEPTEAM_PYTHON=${VENV_DIR}/deepteam-venv/bin/python"
