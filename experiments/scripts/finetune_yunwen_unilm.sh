PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/腾讯对话NaturalConv_Release_20210318
MODEL_PATH=/root/work2/work2/chenzhihao/pretrained_models/torch_unilm_model
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/unilm/yunwen_unilm/seq2seq_on_natural_conv
LOGGING_DIR=$OUTPUT_DIR/logs

MODEL_TYPE="unilm"
SOURCE_NAME="source"
TARGET_NAME="target"

export CUDA_VISIBLE_DEVICES=7
python $PROJECT_PATH/experiments/qa_and_text_generation/finetune_unilm_for_seq2seq_yunwen.py \
  --data_dir $DATA_PATH \
  --model_type=$MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --log_dir $LOGGING_DIR \
  --src_file="source.json" \
  --source=$SOURCE_NAME \
  --target=$TARGET_NAME \
  --max_seq_length=512 \
  --max_position_embeddings=512 \
  --do_train \
  --do_lower_case \
  --train_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=10 \
  --scheduler="linear" \
  --local_rank=-1 \
  --gradient_accumulation_steps=1 \
  --seed=2333 \
  --fp16 \
  --fp16_opt_level='O1'

