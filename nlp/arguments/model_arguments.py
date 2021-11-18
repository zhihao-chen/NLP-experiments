# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: model_arguments
    Author: czh
    Create Date: 2021/11/11
--------------------------------------
    Change Activity: 
======================================
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="hfl/chinese-roberta-wwm-ext",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    use_lstm: bool = field(
        default=False,
        metadata={"help": "Whether or not to use lstm behind lm model"}
    )
    dropout_rate: float = field(default=0.5)
    crf_learning_rate: float = field(default=3e-3)
    model_type: str = field(
        default="bert",
        metadata={"help": "Specify the encoder type.", "choices": ["bert", "nezha", "roformer", "albert"]}
    )
    do_adv: bool = field(
        default=False,
        metadata={"help": "Whether to adversarial training."}
    )
    adv_epsilon: float = field(
        default=1.0,
        metadata= {"help": "Epsilon for adversarial."}
    )
    adv_name: str = field(default='word_embeddings', metadata={"help": "name for adversarial layer."})
    soft_label: bool = field(default=False)
    loss_type: str = field(default="ce", metadata={"help": "Loss function", "choices": ['lsr', 'focal', 'ce']})

    # myparams
    reserve_p: float = field(
        default=1.0,
        metadata={"help": "Will use when use child-tuning"}
    )
    mode: str = field(
        default=None,
        metadata={"help": "Specify what mode will be used for Child-Tuning. eg:'ChildTuning-D', 'ChildTuning-F'"}
    )
    rdrop_alpha: int = field(default=5, metadata={"help": "Rdrop alpha value, only when use rdrop"})
    rope: bool = field(default=False, metadata={"help": "Whether use RoPositionEmbedding or not"})
