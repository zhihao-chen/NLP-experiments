PROJECT_PATH=/data/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=$PROJECT_PATH/datas
OUTPUR_DIR=$CURRENT_DIR/output_file_dir
LOG_DIR=$CURRENT_DIR/logs
BERT_BASE_DIR=/data/chenzhihao/chinese-roberta-ext
DATA_FORMAT='duee1'
TASK_NAME="news2"
MODEL_TYPE="bert"  # bert, nezha, roformer
CUDA_NUMBERS='1' # '0,1,2,3'
SCHEDULER_TYPE='linear'  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]

python $CURRENT_DIR/relation_extraction/train_ee.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$BERT_BASE_DIR \
  --data_format=$DATA_FORMAT \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_eval_per_epoch \
  --do_predict_tag \
  --do_eval_per_epoch \
  --use_lstm \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=3e-5 \
  --crf_learning_rate=2e-3 \
  --num_train_epochs=30.0 \
  --fp16 \
  --local_rank -1 \
  --gradient_accumulation_steps=1 \
  --logging_steps=500 \
  --save_steps=500 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --logging_dir=$LOG_DIR/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=2333 \
  --cuda_number=$CUDA_NUMBERS \
  --scheduler_type=$SCHEDULER_TYPE