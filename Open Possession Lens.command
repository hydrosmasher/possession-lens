#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -f local/dashboard.html ]; then
    open local/dashboard.html
else
    open dashboard.html
fi
