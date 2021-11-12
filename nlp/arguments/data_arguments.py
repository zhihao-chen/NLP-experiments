# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: data_arguments
    Author: czh
    Create Date: 2021/11/11
--------------------------------------
    Change Activity: 
======================================
"""
from dataclasses import dataclass, field
from typing import Optional


task_to_keys = {
    "ner": ("sentence", None),
    "duee1": ("sentence", None),
    "re": ("sentence", None),
    "cluener": ("sentence", None),
    "kg": ("sentence", None),
    "news": ("sentence", None)
}


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": f"The name of the task to train on: {', '.join(task_to_keys.keys())}"},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Dataset path"}
    )
    data_format: str = field(
        default=None,
        metadata={"help": "Specify the data format when use TPLinker。eg: duee1"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None,
                                     metadata={"help": "A csv or a json file containing the test data."})
    labels_file: Optional[str] = field(default=None,
                                       metadata={"help": "A txt or a json file containing the label data"})
    do_lower_case: bool = field(default=False,
                                metadata={"help": "Do you want to convert uppercase letters to lowercase letters"})
    markup: str = field(
        default=None,
        metadata={
            "help": "The label type of sequential label tasks",
            "choices": ['bios', 'bio', 'bieos', '']
        }
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
