#!/bin/bash

uvicorn server.joligan_api:app --reload "$*"
