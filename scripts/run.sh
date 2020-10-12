#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit
cd "../"

source ./venv/bin/activate
jupyter lab --ip=0.0.0.0 --no-browser


