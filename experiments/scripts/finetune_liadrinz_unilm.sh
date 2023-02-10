PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/腾讯对话NaturalConv_Release_20210318/corpus.txt
MODEL_TYPE=unilm
MODEL_NAME=/root/work2/work2/chenzhihao/pretrained_models/unilm-chinese-base
#MODEL_NAME=peterchou/unilm-chinese-base
OUTPUT_DIR=$PROJECT_PATH/datas/output_dir/unilm/liadrinz_unilm/seq2seq_on_natural_conv

export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=1
python3 -u $PROJECT_PATH/experiments/qa_and_text_generation/finetune_unilm_for_seq2seq_liadrinz.py \
    train \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size 32 \
    --corpus_file $DATA_PATH \
    --max_seq_len 512 \
    --seed 42 \
    --output_dir ${OUTPUT_DIR} \
    --gradient_accumulation_steps 2 \
    --lr=2e-5 \
    --num_train_epochs 5 \
    --mask_prob 0.2 \
    --local_rank=-1 \
    --fp16