#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/ci/release_all.sh" "$@"
