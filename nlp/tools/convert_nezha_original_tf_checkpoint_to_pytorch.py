# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: convert_nezha_original_tf_checkpoint_to_pytorch
    Author: czh
    Create Date: 2021/8/18
--------------------------------------
    Change Activity: 
======================================
"""
# Convert ALBERT checkpoint.
import argparse
import logging
import torch
from nlp.models.nezha import NeZhaConfig, NeZhaForPreTraining, load_tf_weights_in_nezha

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, nezha_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = NeZhaConfig.from_json_file(nezha_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = NeZhaForPreTraining(config)
    # Load weights from tf checkpoint
    load_tf_weights_in_nezha(model, tf_checkpoint_path)
    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    state_dict = {k: v for k, v in model.state_dict().items() if 'relative_positions' not in k}
    torch.save(state_dict, pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--nezha_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained ALBERT model. \n"
             "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.nezha_config_file, args.pytorch_dump_path)


'''
python convert_nezha_original_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./pretrained_models/nezha-large-www \
    --nezha_config_file=./pretrained_models/nezha-large-www/config.json \
    --pytorch_dump_path=./pretrained_models/nezha-large-www/pytorch_model.bin
'''
