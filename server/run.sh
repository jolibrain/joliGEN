#!/bin/bash

uvicorn server.joligen_api:app --reload $*
