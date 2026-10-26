#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/../ci/build-docker-images.sh" "$@"
