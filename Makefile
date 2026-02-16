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
ifneq (,$(filter $(SERVER_NAME),flair-node-12 flair-node-06))
DATADIR=/homes/80/sascha/data
else
DATADIR=~/data_local/data
endif
SCRATCH_DIR=~/scratch_LOB
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) -v $(DATADIR):/home/$(MYUSER)/data -v $(SCRATCH_DIR):/home/$(MYUSER)/scratch --shm-size 20G
PORT_FLAGS= -p 8077:80 -p 8076:6006
RUN_FLAGS=$(GPUS) $(BASE_FLAGS) $(PORT_FLAGS)
BASIC_FLAGS=$(GPUS) $(BASE_FLAGS)


DOCKER_IMAGE_NAME = jaxmarl_sascha
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_BASIC=docker run --gpus "device=$(gpu)" $(BASE_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile_JAXMARL --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/. 
	

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

ppo_2player:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name='ippo_rnn_JAXMARL_2player'"
ppo_exec_FQC:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name='ippo_rnn_JAXMARL_exec_FQC'"
ppo_exec_FP:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name='ippo_rnn_JAXMARL_exec_FP'"
ppo_mm_BOB:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name='ippo_rnn_JAXMARL_mm_BOB'"
ppo_mm_FQ:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py --config-name='ippo_rnn_JAXMARL_mm_FQ'"
baseline:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/baseline_eval/baseline_JAXMARL.py --config-name='2player_config'"
baseline_only_avst:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/baseline_eval/baseline_only_JAXMARL.py --config-name='baseline_mm_config_AvSt'"
baseline_only_bob:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/baseline_eval/baseline_only_JAXMARL.py --config-name='baseline_mm_config_bobStrategy'"

plot_trajectories:
	$(DOCKER_RUN_BASIC) /bin/bash -c "python3 ./gymnax_exchange/jaxrl/MARL/baseline_eval/plotting_episodes.py --combo "CURIOUS" "LACED" --directory=trajectories"
workflow-test:
	# without -it flag
	docker run --rm -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"