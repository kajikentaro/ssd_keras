#!/bin/bash
# ------------------------------------------------------------------
# [Masaya Ogushi] DOCKER EXEC TRAIN
#
#          library for Unix shell scripts.
#          Description
#              It take a lot of times to make a full feature files
#          Reference
#              http://dev.classmethod.jp/tool/jq-manual-japanese-translation-roughly/
#
#          Usage:
#               sh  docker_gpu_run.sh
# ------------------------------------------------------------------
# --- Function --------------------------------------------
# -- Body ---------------------------------------------------------

SRC=$(shell dirname `pwd`)
TRAIN_SCRIPT="/src/run_ssd_trainer.py"
DOCKER_IMAGE="ssd_keras"
CUDA=/usr/local/cuda
LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64/

nvidia-docker run --privileged --name="ssd_model" -p 8888:8888 -p 6006:6006 \
                  -e LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
                  -e CUDA_HOME=${CUDA} \
                  -v /etc/localtime:/etc/localtime:ro \
                  -v /usr/local/cuda:/usr/local/cuda \
                  -d -v $(SRC):/src  \
                  -it ${DOCKER_IMAGE} bash -c "python ${TRAIN_SCRIPT} "
