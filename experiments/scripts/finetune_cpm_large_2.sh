PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/腾讯对话NaturalConv_Release_20210318
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/CPM-natural_conv/
LOGGING_DIR=$OUTPUT_DIR/logs
MODEL_PATH="/root/work2/work2/chenzhihao/pretrained_models/CPM-generate"

PROJECT_NAME='nlp'
EXPERIMENT_NAME='cpm1-natural_conv'
GROUP_NAME='cpm1_generate'
SPEAKER1="用户："
SPEAKER2="\n机器人："

#export CUDA_VISIBLE_DEVICES=2,3
#accelerate config
accelerate launch $PROJECT_PATH/experiments/qa_and_text_generation/finetune_cpm_large_2.py \
  --pretrained \
  --model_checkpoint=$MODEL_PATH \
  --config_path=$MODEL_PATH/config.json \
  --tokenizer_path=$MODEL_PATH \
  --data_path=$DATA_PATH \
  --data_type="natural_conv" \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOGGING_DIR \
  --project_name=$PROJECT_NAME \
  --experiment_name=$EXPERIMENT_NAME \
  --group_name=$GROUP_NAME \
  --speaker1=$SPEAKER1 \
  --speaker2=$SPEAKER2 \
  --scheduler="linear" \
  --num_epochs=15 \
  --do_train \
  --do_valid \
  --do_test \
  --train_path="train.txt" \
  --valid_path="dev.txt" \
  --test_path="test.txt" \
  --train_batch_size=4 \
  --valid_batch_size=4 \
  --lr=2e-5 \
  --warmup_steps=2000 \
  --valid_steps=500 \
  --gradient_accumulation_steps=32 \
  --local_rank=0 \
  --mixed_precision='fp16' \
  --seed=2333 \
  --with_tracking \
  --max_seq_length=512 \
  --max_history=10 \
  --do_sample \
  --top_k=0 \
  --top_p=0.9 \
  --temperature=0.75 \
  --output_max_length=256 \
  --output_min_length=2