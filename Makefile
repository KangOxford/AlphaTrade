NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
GPUS=--gpus '"device=5"'
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
SERVER_NAME = $(shell hostname)
# If using flair12 server, set data directory to /homes/80/sascha/data, otherwise assume data is on same level as the repo
ifeq ($(SERVER_NAME),flair-node-12)
DATADIR=/homes/80/sascha/data
else
DATADIR=~/data
endif
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) -v $(DATADIR):/home/$(MYUSER)/data --shm-size 20G -p 8030:80 -p 8031:6006
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = jaxmarl_lob
IMAGE = $(DOCKER_IMAGE_NAME):latest #  for working image: IMAGE = $(DOCKER_IMAGE_NAME):working or 
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile_JAXMARL  --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/. 
	
# --no-cache
	

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"