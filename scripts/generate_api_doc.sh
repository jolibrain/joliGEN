#!/bin/bash

set -x

cd "$(dirname "${BASH_SOURCE[0]}")"/..

PORT=8000
DOC_PATH="docs/source/_static"
mkdir -p ${DOC_PATH}
uvicorn server.joligen_api:app --host localhost --port ${PORT} &
pid=$!

sleep 3

for i in {1..30}
do
    http_code=$(curl -o /dev/null -w "%{http_code}" localhost:${PORT}/redoc)

    # if code == 200 get doc and exit
    if [[ ${http_code} -eq 200 ]]
    then
        wget localhost:${PORT}/redoc -O "${DOC_PATH}/api.html"
        wget localhost:${PORT}/openapi.json -O "${DOC_PATH}/openapi.json"
        # XXX: Change to real path in the documentation
        sed -i 's/openapi.json/doc\/_static\/openapi.json/g' "${DOC_PATH}/api.html"
        kill $pid
        exit 0
    fi

    # exit if server failed
    ps -a | grep $pid
    serv_status=$?
    if [[ ${serv_status} -ne 0 ]]
    then
        exit 1
    fi

    sleep 1
done

kill $pid
