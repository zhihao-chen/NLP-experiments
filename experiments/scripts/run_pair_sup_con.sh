PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/chinese-semantics-match-dataset/
OUTPUT_DIR=$PROJECT_PATH/experiments/output_file_dir/semantic_match
MODEL_PATH="/root/work2/work2/chenzhihao/pretrained_models/chinese-roberta-wwm-ext"

PROJECT_NAME='semantic_match'
EXPERIMENT_NAME='bq-pairsupcon-roberta-wwm-ext'
GROUP_NAME='nlp'
MODEL_TYPE='roberta-wwm-ext'
DATA_TYPE='BQ'

python $PROJECT_PATH/experiments/sentence_embedding/run_pair_sup_con.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path=$MODEL_PATH \
  --output_dir=$OUTPUT_DIR \
  --project_name=$PROJECT_NAME \
  --group_name=$GROUP_NAME \
  --experiment_name=$EXPERIMENT_NAME \
  --data_dir=$DATA_PATH \
  --data_type=$DATA_TYPE \
  --do_train \
  --do_valid \
  --do_test \
  --max_seq_length=64 \
  --lr_rate=5e-05 \
  --lr_scale 100 \
  --gradient_accumulation_steps=1 \
  --scheduler_type='linear' \
  --train_batch_size=256 \
  --valid_batch_size=128 \
  --num_train_epochs=200 \
  --gpuid=3 \
  --seed=2333 \
  --num_worker=0 \
  --num_labels=2 \
  --temperature=0.05 \
  --task_type='pairsupcon' \
  --contrast_type="HardNeg" \
  --beta=1
