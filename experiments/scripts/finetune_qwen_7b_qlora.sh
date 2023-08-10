#! /bin/bash

ROOT_PATH=/home/aiteam/work2/NLP
MODEL_NAME_OR_PATH=/home/aiteam/work/pretrained_models/Qwen-7B-Chat
MODEL_TYPE=qwen-7b-chat #bloom,llama

DATA_DIR=${ROOT_PATH}/datas/firefly
OUTPUT_DIR=${ROOT_PATH}/datas/output_dir/${MODEL_TYPE}/instruct_qlora
mkdir -p ${OUTPUT_DIR}

CACHE_DIR=${ROOT_PATH}/datas/hf_cache_dir/${MODEL_TYPE}/instruct_qlora
mkdir -p ${CACHE_DIR}

CUTOFF_LEN=1024
SYSTEM_PROMPT=""

# V100不支持lora+fp16
CUDA_VISIBLE_DEVICES="6" torchrun --nproc_per_node 1 ${ROOT_PATH}/experiments/qa_and_txt_generation/finetune_llm_chat.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_type ${MODEL_TYPE} \
    --use_qlora True \
    --bits 4 \
    --lora_config ${ROOT_PATH}/config/lora_config_llama.json \
    --train_file ${DATA_DIR}/train.json \
    --validation_file ${DATA_DIR}/dev.json \
    --chat_format 'chatml' \
    --source_prefix "human" \
    --target_prefix "assistant" \
    --system_prompt $SYSTEM_PROMPT \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --model_max_length ${CUTOFF_LEN} \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 8e-6 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${CACHE_DIR} \
    --report_to "all" \
    --output_dir ${OUTPUT_DIR}
#    --save_total_limit 5 \
#    --metric_for_best_model "rouge-l" \
#    --predict_with_generate True
#    --optim paged_adamw_32bit
