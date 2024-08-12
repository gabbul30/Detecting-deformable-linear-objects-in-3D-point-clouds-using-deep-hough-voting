#!/bin/bash
#
# This script runs docker build to create the pytorch-gpu.
#

set -exu
docker build --tag ynyg/torch-gpu-ubuntu20 -f Dockerfile ./
