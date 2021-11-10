# -*- coding: utf8 -*-
"""
======================================
    Project Name: EventExtraction
    File Name: arguments
    Author: czh
    Create Date: 2021/9/15
--------------------------------------
    Change Activity: 
======================================
"""
from pathlib import Path
from typing import Union

from pydantic import dataclasses, Field


@dataclasses.dataclass
class DataAndTrainArguments:
    task_name: str = Field(
        default='ee',
        description="Task Name. This will be used for directory name to distinguish different task."
    )
    data_dir: Union[str, Path] = Field(
        default="../data/normal_data",
        description="The input data dir. Should contain the training files"
    )
    model_type: str = Field(
        default='bert',
        description="Model type selected in the list: [bert, nezha]"
    )
    model_name_or_path: Union[str, Path] = Field(
        default="hfl/chinese-roberta-wwm-ext",
        description="pretrained bert directory (absolute path) or transformer bert model name. "
                    "bert-base-cased， chinese-bert-wwm-ext-hit"
    )
    model_sate_dict_path: Union[str, Path] = Field(default="",
                                                   description="Save path of have trained model for "
                                                               "continue train or predict")
    output_dir: Union[str, Path] = Field(
        default="../data/output/",
        description="The output directory where the model predictions and checkpoints will be written."
    )
    # Other parameters
    markup: str = Field(default='', description="Annotation method for sequence that choice from the list: "
                                                "['bios', 'bio', 'bieos', '']")
    config_name: Union[str, Path] = Field(
        default="",
        description="Pretrained config name or path if not the same as model_name"
    )
    tokenizer_name: Union[str, Path] = Field(
        default="",
        description="Pretrained tokenizer name or path if not the same as model_name"
    )
    cache_dir: Union[str, Path] = Field(
        default="",
        description="Where do you want to store the pre-trained models downloaded from s3"
    )
    evaluate_during_training: bool = Field(default=True, description="Whether to run evaluation during training", )
    do_eval_per_epoch: bool = Field(default=True, description="Whether to run eval after each epoch")
    do_predict_no_tag: bool = Field(default=False, description="Whether to run eval on the test set.")
    do_predict_tag: bool = Field(default=True, description="Whether to run predictions on the test set.")
    eval_all_checkpoints: bool = Field(default=False,
                                       description="Evaluate all checkpoints starting with the same prefix "
                                                   "as model_name ending and ending with step number", )
    do_lower_case: bool = Field(default=False, description="Set this flag if you are using an uncased model.")
    use_lstm: bool = False
    from_scratch: bool = True
    from_last_checkpoint: bool = Field(default=False, description="Only if 'from_scratch' was set 'False'")
    early_stop: bool = Field(default=False, description="Whether early stop the training")
    overwrite_output_dir: bool = Field(default=True, description="Overwrite the content of the output directory")
    overwrite_cache: bool = Field(default=True, description="Overwrite the cached training and evaluation sets")
    no_cuda: bool = Field(default=True, description="Avoid using CUDA when available")
    fp16: bool = Field(default=True, description="Whether to use 16-bit (mixed) precision instead of 32-bit", )

    train_max_seq_length: int = Field(default=128,
                                      description="The maximum total input sequence length after tokenization. "
                                                  "Sequences longer than this will be truncated,"
                                                  "sequences shorter will be padded.", )
    eval_max_seq_length: int = Field(default=512,
                                     description="The maximum total input sequence length after tokenization."
                                                 "Sequences longer than this will be truncated,"
                                                 "sequences shorter will be padded.", )
    per_gpu_train_batch_size: int = Field(default=8, description="Batch size per GPU/CPU for training.")
    per_gpu_eval_batch_size: int = Field(default=8, description="Batch size per GPU/CPU for evaluation.")
    gradient_accumulation_steps: int = Field(default=1,
                                             description="Number of updates steps to accumulate before performing "
                                                         "a backward/update pass.", )
    learning_rate: float = Field(default=5e-5, description="The initial learning rate for Adam.")
    crf_learning_rate: float = Field(default=5e-5, description="The initial learning rate for crf and linear layer.")
    weight_decay: float = Field(default=0.01, description="Weight decay if we apply some.")
    adam_epsilon: float = Field(default=1e-8, description="Epsilon for Adam optimizer.")
    warmup_proportion: float = Field(default=0.1,
                                     description="Proportion of training to perform linear learning rate warmup for,"
                                                 "E.g., 0.1 = 10% of training.")
    num_train_epochs: float = Field(default=3.0, description="Total number of training epochs to perform.")
    max_steps: int = Field(default=-1,
                           description="If > 0: set total number of training steps to perform. "
                                       "Override num_train_epochs.", )
    tolerance: int = Field(default=5, description="What number of steps for early stop.")
    logging_steps: int = Field(default=500, description="Log every X updates steps.")
    save_steps: int = Field(default=500, description="Save checkpoint every X updates steps.")

    scheduler_type: str = Field(default="linear",
                                description="The scheduler type to use. "
                                            "['linear','cosine','cosine_with_restarts',"
                                            "'polynomial','constant','constant_with_warmup']")
    cuda_number: str = "0"
    seed: int = Field(default=2333, description="random seed for initialization")
    local_rank: int = Field(default=-1, description="For distributed training: local_rank")
    dropout_rate: float = Field(default=0.3, description="Dropout rate")
