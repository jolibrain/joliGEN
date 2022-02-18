#!/bin/bash

set -ex

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ $# -ne 1 ]
then
    echo "Usage: $0 [dataroot]"
    exit
fi

echo "Data root = $1"

python3 "${current_dir}/../train.py" --dataroot "$1" --model_type cycle_gan --train_pool_size 50
