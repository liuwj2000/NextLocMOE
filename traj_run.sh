#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export NCCL_P2P_LEVEL=NVL                      # ?? NVLink ??
export TORCH_NCCL_BLOCKING_WAIT=1                 # ?? hang,????
export NCCL_ASYNC_ERROR_HANDLING=1            # ?? NCCL ????
accelerate launch --num_machines 1 \
                            --mixed_precision bf16 \
                            --dynamo_backend no\
                            --num_processes 7 run_model_NextlocLLM_MER.py \
                            --config_file user_config/Shanghai_MOE\
                            --max_epoch 150\
                            --learning_rate 0.001 \
                            --batch_size 12\
                            --if_train 1 \
                            --save_dict ./acce_file/Shanghai.pth 2>&1 | tee Shanghai_MOE.log
                            