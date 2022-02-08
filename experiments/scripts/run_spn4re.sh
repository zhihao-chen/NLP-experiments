PROJECT_PATH=/data/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=$PROJECT_PATH/datas
OUTPUR_DIR=$CURRENT_DIR/output_file_dir
LOG_DIR=$CURRENT_DIR/logs
BERT_BASE_DIR=/data/chenzhihao/chinese-roberta-ext
TASK_NAME="spnre_business_chance"
MODEL_TYPE="bert"  # bert,nezha,albert,roformer

python /data/chenzhihao/NLP/experiments/relation_extraction/train_spn4re.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict_no_tag \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=512 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=3e-5 \
  --crf_learning_rate=2e-3 \
  --num_train_epochs=100.0 \
  --fp16 \
  --fp16_backend=amp \
  --warmup_ratio=0.1 \
  --local_rank -1 \
  --gradient_accumulation_steps=1 \
  --logging_steps=500 \
  --save_steps=500 \
  --eval_steps=500 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}/ \
  --logging_dir=$LOG_DIR/ \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed=2333 \
  --cuda_number=0 \
  --markup=bios \
  --metric_for_best_model=f1 \
  --greater_is_better \
  --sliding_len=20 \
  --relation_labels="BUSEXP,ORGFIN,PERUP,STRCOO" \
  --num_generated_tuples=10 \
  --num_entities_in_tuple=8 \
  --allow_null_entities_in_tuple="0,0,1,1,1,1,1,1" \
  --entity_loss_weight="2,2,2,2,2,2,2,2"
