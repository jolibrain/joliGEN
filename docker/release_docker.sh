#!/bin/bash

gt=$(git describe --tags --abbrev=0)
echo $gt
command="docker build -t docker.jolibrain.com/joligen_build:$gt -f docker/Dockerfile.build ."
echo $command
eval $command
