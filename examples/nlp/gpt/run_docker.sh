#!/bin/bash

docker run --gpus all -it --rm --shm-size 16G -v -p 8989:8989 /disk1:/data nvcr.io/nvidia/nemo:24.05
