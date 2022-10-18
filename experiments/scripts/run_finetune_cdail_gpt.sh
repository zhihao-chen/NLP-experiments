PROJECT_PATH=/data2/work2/chenzhihao/NLP-experiments
DATA_PATH=$PROJECT_PATH/datas/raw_datas/
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/CDail-GPT-QA
MODEL_CHECKPOINT=$PROJECT_PATH/datas/output_dir/CDail-GPT-QA

python $PROJECT_PATH/examples/qa_and_text_generation/finetune_cdail_gpt.py \
  --pretrained \
  --model_checkpoint=$MODEL_CHECKPOINT \
  --data_path=$DATA_PATH \
  --output_dir=$OUTPUT_DIR \
  --scheduler="linear" \
  --n_epochs=30 \
  --train_batch_size=12 \
  --valid_batch_size=12 \
  --lr=5e-5 \
  --warmup_steps=5000 \
  --valid_steps=5000 \
  --gradient_accumulation_steps=64 \
  --local_rank=-1 \
  --fp16='01' \
  --fp16_backend='amp' \
  --device='cuda:0' \
  --do_train \
  --do_valid