PROJECT_PATH=/data2/work2/chenzhihao/NLP
DATA_PATH=$PROJECT_PATH/datas/raw_datas/
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/CPM-large2/
LOGGING_DIR=$OUTPUT_DIR/logs
MODEL_PATH="/data2/work2/chenzhihao/pretrained_models/CPM-generate"

#export CUDA_VISIBLE_DEVICES=2,3
#accelerate configs
accelerate launch $PROJECT_PATH/examples/qa_and_text_generation/finetune_cpm_large_accelerate.py \
  --pretrained \
  --model_checkpoint=$MODEL_PATH \
  --config_path=$MODEL_PATH/config.json \
  --tokenizer_path=$MODEL_PATH \
  --data_path=$DATA_PATH \
  --output_dir=$OUTPUT_DIR \
  --logging_dir=$LOGGING_DIR \
  --scheduler="linear" \
  --num_epochs=15 \
  --train_batch_size=4 \
  --valid_batch_size=4 \
  --lr=2e-5 \
  --warmup_steps=2000 \
  --valid_steps=2000 \
  --gradient_accumulation_steps=32 \
  --local_rank=0 \
  --mixed_precision='fp16' \
  --seed=2333 \
  --do_train \
  --do_valid \
  --do_test \
  --with_tracking \
  --max_seq_length=512 \
  --do_sample \
  --top_k=0 \
  --top_p=0.0 \
  --temperature=1.0 \
  --output_max_length=256 \
  --output_min_length=5