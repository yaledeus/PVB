#!/bin/zsh

########## adjust configs according to your needs ##########
CONFIG=./config/train.yaml

######### end of adjust ##########

########## Instruction ##########
# This script takes three optional environment variables:
# MODEL / GPU / ADDR / PORT
# e.g. Use gpu 0, 1 and 4 for training, set distributed training
# master address and port to localhost:9901, the command is as follows:
#
# MODEL=MEAN GPU="0,1,4" ADDR=localhost PORT=9901 bash train.sh
#
# Default value: GPU=-1 (use cpu only), ADDR=localhost, PORT=9901
# Note that if your want to run multiple distributed training tasks,
# either the addresses or ports should be different between
# each pair of tasks.
######### end of instruction ##########

# set master address and port e.g. ADDR=localhost PORT=9901 bash train.sh
MASTER_ADDR=localhost
MASTER_PORT=9901
if [ $ADDR ]; then MASTER_ADDR=$ADDR; fi
if [ $PORT ]; then MASTER_PORT=$PORT; fi
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

# set gpu, e.g. GPU="0,1,2,3" bash train.sh
if [ -z "$GPU" ]; then
    GPU="-1"  # use CPU
fi
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPUs: $GPU"
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	  PREFIX="python3 -m torch.distributed.run --nproc_per_node=${#GPU_ARR[@]} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --standalone --nnodes=1"
else
    PREFIX="python"
fi

${PREFIX} train.py \
    --config $CONFIG \
    --gpu "${!GPU_ARR[@]}"
