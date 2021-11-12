PROJECT_PATH=/data/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=$PROJECT_PATH/datas
OUTPUR_DIR=$CURRENT_DIR/output_file_dir
LOG_DIR=$CURRENT_DIR/logs
BERT_BASE_DIR=/data/chenzhihao/chinese-roberta-ext
TASK_NAME="cluener"
MODEL_TYPE="bert"

python /data/chenzhihao/NLP/experiments/others/child_tuning_on_ner.py \
  --model_name_or_path=$BERT_BASE_DIR \
  --model_type=$MODEL_TYPE \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict_no_tag \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=256 \
  --per_gpu_train_batch_size=32 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=3e-5 \
  --crf_learning_rate=3e-3 \
  --num_train_epochs=30.0 \
  --fp16 \
  --fp16_backend=amp \
  --gradient_accumulation_steps=1 \
  --warmup_ratio=0.1 \
  --logging_steps=500 \
  --save_steps=500 \
  --eval_steps=1000 \
  --save_total_limit=10 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --logging_dir=$LOG_DIR/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=42 \
  --cuda_number=0 \
  --markup=bios \
  --reserve_p=0.3 \
  --mode=ChildTuning-D \
  --metric_for_best_model=f1 \
  --greater_is_better