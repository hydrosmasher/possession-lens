#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import jupyterlab,ipykernel' 2>/dev/null; then
    LENS_PYTHON="$PWD/.venv/bin/python"
elif [ -x "$HOME/Documents/Codex/Python-Jupyter/.venv/bin/python" ] && "$HOME/Documents/Codex/Python-Jupyter/.venv/bin/python" -c 'import jupyterlab,ipykernel,numpy,pandas' 2>/dev/null; then
    LENS_PYTHON="$HOME/Documents/Codex/Python-Jupyter/.venv/bin/python"
else
    echo 'Create a Python 3.12 environment and install requirements-notebook.txt. See README.md.'
    read -r -p 'Press Enter to close.'
    exit 1
fi
export JUPYTER_CONFIG_DIR="$PWD/local/jupyter/config"
export JUPYTER_DATA_DIR="$PWD/local/jupyter/share/jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/local/jupyter/runtime"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
"$LENS_PYTHON" -m ipykernel install --prefix "$PWD/local/jupyter" --name possession-lens --display-name 'Python 3 (Possession Lens)'
exec "$LENS_PYTHON" -m jupyterlab --ServerApp.root_dir="$PWD" notebooks/01_Possession_Lens.ipynb
