PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH/
CURRENT_DIR=$PROJECT_PATH/experiments
DATA_DIR=/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/
OUTPUT_DIR=$CURRENT_DIR/output_file_dir/semantic_match
LOG_PATH=$OUTPUT_DIR/dy_logs/

MODLE_TYPE='roberta-wwm'
MODLE_NAME_OR_PATH=/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext
DATA_TYPE='BQ'
OBJECT_TYPE='classification'
TASK_TYPE='match'
POOLING_STRATEGY='first-last-avg'
PROJECT_NAME='sup-sbert'
EXPERIMENT_NAME='sbert-training-dynamics'
GROUP_NAME='semantic_match'

python $CURRENT_DIR/sentence_embedding/train_sentence_bert_training_dynamics.py \
  --model_type $MODLE_TYPE \
  --model_name_or_path $MODLE_NAME_OR_PATH \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --dy_log_path $LOG_PATH \
  --data_type $DATA_TYPE \
  --task_type $TASK_TYPE \
  --object_type $OBJECT_TYPE \
  --pooling_strategy $POOLING_STRATEGY \
  --project_name $PROJECT_NAME \
  --experiment_name $EXPERIMENT_NAME \
  --group_name $GROUP_NAME \
  --do_train \
  --do_valid \
  --do_test \
  --do_recording \
  --max_seq_length=128 \
  --num_train_epochs=32 \
  --valid_batch_size=32 \
  --test_batch_size=32 \
  --valid_steps=500 \
  --num_labels=2 \
  --lr_rate=2e-5 \
  --gradient_accumulation_steps=1 \
  --scheduler_type='linear' \
  --num_workers=0 \
  --cuda_number=7
