#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit
cd "../"

source ./venv/bin/activate
tensorboard --logdir=./logs --host 0.0.0.0

