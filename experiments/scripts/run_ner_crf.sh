PROJECT_PATH=/data/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=$PROJECT_PATH/datas
OUTPUR_DIR=$CURRENT_DIR/output_file_dir
LOG_DIR=$CURRENT_DIR/logs
BERT_BASE_DIR=/data/chenzhihao/chinese-roberta-ext
TASK_NAME="ner"
MODEL_TYPE="bert"  # bert,nezha,albert,roformer

python /data/chenzhihao/NLP/experiments/ner/run_ner_crf.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=256 \
  --eval_max_seq_length=256 \
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
  --seed=42 \
  --cuda=1