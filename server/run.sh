#!/bin/bash
set -euo pipefail
exec uvicorn server.joligen_api:app "$@"
