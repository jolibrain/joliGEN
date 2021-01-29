#!/bin/bash

set -ex

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ $# -ne 1 ]
then
    echo "Usage: $0 [dataroot]"
    exit
fi

echo "Data root = $1"

python3 "${current_dir}/../train.py" --dataroot "$1" --model cycle_gan --pool_size 50 --no_dropout
