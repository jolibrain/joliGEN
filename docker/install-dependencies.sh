#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3-pip python3-dev python3-opencv python3-pytest ninja-build sudo wget unzip git ca-certificates
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt uvicorn fastapi
python3 -m pip check
rm -rf /var/lib/apt/lists/*
