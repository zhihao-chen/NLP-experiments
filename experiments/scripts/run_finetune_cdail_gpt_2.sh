PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/腾讯对话NaturalConv_Release_20210318
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/CDail-GPT
MODEL_CHECKPOINT=/root/work2/work2/chenzhihao/pretrained_models/CDial-GPT_LCCC-large

python $PROJECT_PATH/experiments/qa_and_text_generation/finetune_cdail_gpt_2.py \
  --pretrained \
  --model_checkpoint=$MODEL_CHECKPOINT \
  --data_path=$DATA_PATH \
  --data_type="natural_conv" \
  --output_dir=$OUTPUT_DIR \
  --scheduler="linear" \
  --n_epochs=30 \
  --max_history=10 \
  --do_train \
  --do_valid \
  --do_test \
  --train_path="train.txt" \
  --valid_path="dev.txt" \
  --test_path="test.txt" \
  --train_batch_size=16 \
  --valid_batch_size=16 \
  --lr=5e-5 \
  --warmup_steps=5000 \
  --valid_steps=500 \
  --gradient_accumulation_steps=1 \
  --local_rank=-1 \
  --fp16='01' \
  --fp16_backend='amp' \
  --device='cuda:1'