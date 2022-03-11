PROJECT_PATH=/data/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=$PROJECT_PATH/datas
OUTPUR_DIR=$DATA_DIR/output_file_dir
LOG_DIR=$DATA_DIR/logs
BERT_BASE_DIR=/data/chenzhihao/chinese-roberta-ext
TASK_NAME="cluener"
MODEL_TYPE="bert"

python $CURRENT_DIR/ner/train_globalpointer.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict_no_tag \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=256 \
  --sliding_len=100 \
  --per_gpu_train_batch_size=64 \
  --per_gpu_eval_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=100 \
  --fp16 \
  --fp16_backend=amp \
  --warmup_ratio=0.1 \
  --local_rank -1 \
  --gradient_accumulation_steps=1 \
  --logging_steps=500 \
  --save_steps=500 \
  --eval_steps=500 \
  --save_total_limit=5 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}/ \
  --logging_dir=$LOG_DIR/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=2333 \
  --cuda_number=0 \
  --dataloader_num_workers=2 \
  --scheduler_type=linear \
  --metric_for_best_model=f1 \
  --greater_is_better \
  --rope \
  --reserve_p=0.3 \
  --mode=ChildTuning-D