help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER_FILE_GPU=Dockerfile-gpu
DOCKER=docker
DOCKER_GPU=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)
CUDA=/usr/local/cuda
LD_LIBRARY=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# CPU environment

build:
	mkdir -p ssd_keras
	cp ../*.py ssd_keras/
	cp -r ../cython_train ssd_keras/
	cp -r ../train ssd_keras/
	cp ../*.ipynb ssd_keras/
	cp -r ../PASCAL_VOC ssd_keras/
	cp -r ../pics ssd_keras/
	cp -r ../testing_utils ssd_keras/
	cp ../weights_SSD300.hdf5 ssd_keras/
	$(DOCKER) build -t ssd_keras --build-arg python_version=3.5 -f $(DOCKER_FILE) .
	rm -rf ssd_keras

bash: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data ssd_keras bash

ipython: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data ssd_keras ipython

notebook: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data ssd_keras

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data ssd_keras pytest $(TEST)

# GPU environment

build-gpu:
	mkdir -p ssd_keras
	cp ../*.py ssd_keras/
	cp -r ../cython_train ssd_keras/
	cp -r ../train ssd_keras/
	cp ../*.ipynb ssd_keras/
	cp -r ../PASCAL_VOC ssd_keras/
	cp -r ../pics ssd_keras/
	cp -r ../testing_utils ssd_keras/
	cp ../weights_SSD300.hdf5 ssd_keras/
	$(DOCKER_GPU) build -t ssd_keras-gpu --build-arg python_version=3.5 -f $(DOCKER_FILE_GPU) .
	rm -rf ssd_keras

bash-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  ssd_keras-gpu bash

ipython-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006  -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  ssd_keras-gpu ipython

notebook-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  ssd_keras-gpu

test-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  ssd_keras-gpu pytest $(TEST)