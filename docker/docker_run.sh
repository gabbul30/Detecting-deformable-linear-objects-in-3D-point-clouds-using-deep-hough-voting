#!/bin/bash
#
# Usage:  ./docker_run.sh [/path/to/data]
#
# This script calls `nvidia-docker run` to start the mask-rcnn
# container with an interactive bash session.  This script sets
# the required environment variables and mounts the labelfusion
# source directory as a volume in the docker container.  If the
# path to a data directory is given then the data directory is
# also mounted as a volume.
# 
# -v mount folder into container

image_name=ynyg/torch-gpu-ubuntu20
xhost +local:docker

docker run --name pytorch_jupyter_ubuntu20 -it --rm\
    -v /home/glbw/:/repository \
    -e DISPLAY \
    --ipc=host \
    --net=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA-VISIBLE_DEVICES=all \
    -e QT_X11_NO_MITSHM=1 \
    -e NB_UID=1000 -e NB_GID=1000 -e VNC_SERVER_PASSWORD=password \
    --runtime=nvidia \
    --gpus 'all,"capabilities=compute,utility,graphics"'\
    --privileged\
    $image_name 

