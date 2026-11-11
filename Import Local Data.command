#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
    LENS_PYTHON="$PWD/.venv/bin/python"
elif [ -x "$HOME/Documents/Codex/Python-Jupyter/.venv/bin/python" ]; then
    LENS_PYTHON="$HOME/Documents/Codex/Python-Jupyter/.venv/bin/python"
else
    echo 'Create the Python environment described in README.md first.'
    read -r -p 'Press Enter to close.'
    exit 1
fi
"$LENS_PYTHON" -m possession_lens import-external --source "$HOME/Downloads" --destination data/external/raw
echo 'Local import complete. Originals remain in Downloads. No files were downloaded.'
read -r -p 'Press Enter to close.'
