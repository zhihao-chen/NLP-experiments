PROJECT_PATH=/root/work2/work2/chenzhihao/NLP
DATA_PATH=/root/work2/work2/chenzhihao/datasets/腾讯对话NaturalConv_Release_20210318
MODEL_PATH=/root/work2/work2/chenzhihao/pretrained_models/torch_unilm_model
MODEL_RECOVER_PATH=$PROJECT_PATH/datas/output_dir/unilm/yunwen_unilm/seq2seq_on_natural_conv/
OUTPUT_FILE=$PROJECT_PATH/datas/output_dir/unilm/yunwen_unilm/seq2seq_on_natural_conv/predict_.json

MODEL_TYPE="unilm"

export CUDA_VISIBLE_DEVICES=7
python $PROJECT_PATH/experiments/single_test/decode_yunwen_unilm_for_seq2seq.py \
  --model_type=$MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --model_recover_path=$MODEL_RECOVER_PATH \
  --input_file=$DATA_PATH/test.json \
  --split="test" \
  --max_seq_length=512 \
  --do_lower_case \
  --batch_size=32 \
  --beam_size=5 \
  --max_tgt_length=128