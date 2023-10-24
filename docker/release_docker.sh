#!/bin/bash

gt=$(git describe --tags --abbrev=0)
echo $gt
command="docker build -t docker.jolibrain.com/joligen_build:$gt -f docker/Dockerfile.build ."
echo $command
eval $command

command="docker build -t docker.jolibrain.com/joligen_server:$gt --build-arg GTAG=$gt -f docker/Dockerfile.server ."
echo $command
eval $command

command="docker push docker.jolibrain.com/joligen_server:$gt"
echo $command
eval $command
