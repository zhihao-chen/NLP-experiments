#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
--------------------------------
Author：czh
date：2023/8/2
--------------------------------
"""
import os

# os.environ["WANDB_DISABLED"] = "true"
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

import logging
import json
import torch
import transformers
import copy

import numpy as np
import pandas as pd
import jieba
import torch
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType
)
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import get_last_checkpoint
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.utils import add_start_docstrings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default='llama', metadata={"help": "Pretrained model type"}
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    # llama: bool = field(
    #     default=False,
    #     metadata={"help": "Llama model"}
    # )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    target_prefix: Optional[str] = field(
        default="",
        metadata={"help": "A prefix to add before every target text (useful for T5 models)."}
    )
    system_prompt: Optional[str] = field(
        default="",
        metadata={"help": "A prefix to add before every prompt"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    chat_format: Optional[str] = field(
        default="belle",
        metadata={"help": "Format of the chat dataset to use (belle, baichuan_chat, chatml)"},
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA."}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA."}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training."}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "LoRA config file."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "ddp_find_unused_parameters"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "gradient_checkpointing"}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    max_memory_MB: int = field(
        default=32000,
        metadata={"help": "Free memory per gpu."}
    )

    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )


# save peft at train end
class SavePeftModelAtEndCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    n_gpus = torch.cuda.device_count()
    max_memory = f'{training_args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = "auto"
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank}
        training_args.gradient_accumulation_steps = max(training_args.gradient_accumulation_steps // world_size, 1)
        training_args.ddp_find_unused_parameters = False
        max_memory = {'': max_memory[local_rank]}

    global_rank = torch.distributed.get_rank()
    log_file = os.path.join(training_args.output_dir, 'print_log.txt')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if 'llama' in model_args.model_type:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        print_rank_0("Set the eos_token_id and bos_token_id of LLama model tokenizer", log_file, global_rank)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
    elif "qwen" in model_args.model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,
                                                  pad_token='<|endoftext|>', bos_token='<|im_start|>',
                                                  eos_token='<|im_end|>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # int8 is not compatible with DeepSpeed (require not to pass device_map)
    if training_args.use_int8_training:
        print_rank_0("int8 is not compatible with DeepSpeed. ", log_file, global_rank)
        # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=True,  # xxx: int8 load in
            device_map=device_map,  # xxx: int8 requires passing device_map
            max_memory=max_memory,
            torch_dtype=torch_dtype,
        )
    elif training_args.use_qlora:
        # tokenizer.bos_token_id = 1

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,  # xxx: int8 load in
            device_map=device_map,  # xxx: int8 requires passing device_map
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        )
        if compute_dtype == torch.float16 and training_args.bits == 4:
            if torch.cuda.is_bf16_supported():
                print('=' * 80)
                print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
                print('=' * 80)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
        )
    model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16
                                                                          else torch.float32))

    print_rank_0("tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id), log_file, global_rank)
    print_rank_0("tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id), log_file, global_rank)
    print_rank_0("tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id), log_file, global_rank)

    # peft model
    if training_args.use_lora or training_args.use_qlora:
        print_rank_0("Loading lora config from {}".format(training_args.lora_config), log_file, global_rank)
        lora_config = json.load(open(training_args.lora_config))
        print_rank_0("Lora config: {}".format(lora_config), log_file, global_rank)

        modules = lora_config['lora_target_modules']
        if training_args.use_int8_training:
            print_rank_0("training_args.use_int8_training!!! (int8 is not compatible with DeepSpeed)", log_file,
                         global_rank)
            model = prepare_model_for_int8_training(model)
        elif training_args.use_qlora:
            model = prepare_model_for_kbit_training(model,
                                                    use_gradient_checkpointing=training_args.gradient_checkpointing)
            modules = find_all_linear_names(training_args, model)
        config = LoraConfig(
            r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=modules,
            lora_dropout=lora_config['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    print_trainable_parameters(training_args, model)

    source_prefix = data_args.source_prefix if data_args.source_prefix else "human"
    target_prefix = data_args.target_prefix if data_args.target_prefix else "assistant"

    def generate_and_tokenize_prompt(data_point):
        input_ids = []
        labels = []
        if data_args.chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [tokenizer.im_start_id]
            im_end_tokens = [tokenizer.im_end_id]
            nl_tokens = tokenizer.encode("\n")

            def _tokenize_str(role, content):
                return f"{role}\n{content}", tokenizer.encode(
                    role, allowed_special=set()
                ) + nl_tokens + tokenizer.encode(content, allowed_special=set())
            if data_args.system_prompt:
                system_text, system_tokens_part = _tokenize_str("system", data_args.system_prompt)
                system_tokens = im_start_tokens + system_tokens_part + im_end_tokens + nl_tokens
                input_ids += system_tokens
                label = [IGNORE_INDEX] * len(system_tokens)
                labels += label
            source = data_point["conversations"]
            for sentence in source:
                sentence_from = sentence["from"].lower()
                if sentence_from == source_prefix:
                    query_text, query_token_part = _tokenize_str("user", sentence['value'])
                    sentence_tokens = im_start_tokens + query_token_part + im_end_tokens + nl_tokens
                    label = [IGNORE_INDEX] * len(sentence_tokens)
                elif sentence_from == target_prefix:
                    response_text, response_token_part = _tokenize_str("assistant", sentence['value'])
                    sentence_tokens = im_start_tokens + response_token_part + im_end_tokens
                    label = copy.deepcopy(sentence_tokens)
                input_ids += sentence_tokens
                labels += label
        elif data_args.chat_format == 'baichuan_chat':
            user = "<reserved_102>"
            assistant = "<reserved_10>3"
            user_id = tokenizer.encode(user)
            assistant_id = tokenizer.encode(assistant)
            if data_args.system_prompt:
                system_prompt = data_args.system_prompt + "\n"
                system_prompt_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
                label = [IGNORE_INDEX] * len(system_prompt_ids)
                input_ids += system_prompt_ids
                labels += label
            source = data_point["conversations"]
            for sentence in source:
                sentence_from = sentence["from"].lower()
                if sentence_from == source_prefix:
                    sentence_tokens = user_id + tokenizer.encode(sentence['value'])
                    label = [IGNORE_INDEX] * len(sentence_tokens)
                elif sentence_from == target_prefix:
                    sentence_tokens = assistant_id + tokenizer.encode(sentence['value']) + [tokenizer.eos_token_id]
                    label = copy.deepcopy(sentence_tokens)
                input_ids += sentence_tokens
                labels += label
        else:
            if data_args.system_prompt:
                system_prompt = data_args.system_prompt + "\n"
                system_prompt_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
                label = [IGNORE_INDEX] * len(system_prompt_ids)
                input_ids += system_prompt_ids
                labels += label
            source = data_point["conversations"]
            for sentence in source:
                sentence_from = sentence["from"].lower()
                # 需要根据不同的模型进行更改，最好与base LLM的格式一致
                sentence_value = f'{source_prefix}:' + sentence["value"] + f'\n{target_prefix}:' \
                    if sentence_from == source_prefix else sentence["value"]
                # conversation += sentence_value
                sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)  # do not add bos_token_id
                label = copy.deepcopy(sentence_ids) if sentence_from != source_prefix else [IGNORE_INDEX] * len(sentence_ids)
                input_ids += sentence_ids
                labels += label
                # add eos at every end of assistant sentence
                if sentence_from != source_prefix:
                    input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
                    labels += [tokenizer.eos_token_id]

        input_ids = input_ids[:training_args.model_max_length - 1]
        labels = labels[:training_args.model_max_length - 1]
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[
                            18:24]  # labels can not have all values being -100. 18 and 24 are just random numbers

        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt

    assert os.path.exists(data_args.train_file), "{} file not exists".format(data_args.train_file)
    if data_args.train_file.endswith(".json") or data_args.train_file.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.train_file, cache_dir=model_args.cache_dir)
    else:
        data = load_dataset(data_args.train_file, cache_dir=model_args.cache_dir)

    column_names = data["train"].column_names
    data.cleanup_cache_files()
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt,
                                             remove_columns=column_names,
                                             # num_proc=8,
                                             load_from_cache_file=not data_args.overwrite_cache)
    val_data = load_dataset("json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir)
    val_data = val_data["train"].shuffle().map(generate_and_tokenize_prompt,
                                               remove_columns=column_names,
                                               # num_proc=8,
                                               load_from_cache_file=not data_args.overwrite_cache)

    for i in range(2):
        print_rank_0("Eval tokenized example: {}".format(val_data[i]), log_file, global_rank)
    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(train_data[i]), log_file, global_rank)

    def compute_metrics(eval_preds):
        preds, labels_ = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != IGNORE_INDEX, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels_ = np.where(labels_ != -100, labels_, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    training_nums = len(data['train'])
    num_gpus = torch.cuda.device_count()

    batch_size = training_args.per_device_train_batch_size * training_args.world_size * \
                 training_args.gradient_accumulation_steps
    t_total = math.ceil(training_nums / batch_size) * training_args.num_train_epochs
    training_args.eval_steps = max(t_total // 5, 5)
    training_args.save_steps = training_args.eval_steps
    training_args.warmup_steps = int(
        t_total * training_args.warmup_ratio) if training_args.warmup_ratio > 0.0 else training_args.warmup_steps
    print_rank_0(
        "num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus, training_nums, t_total, training_args.warmup_steps, training_args.eval_steps,
            training_args.save_steps), log_file, global_rank)
    print_rank_0(
        "val data nums = {}, training_nums = {}, batch_size = {}".format(len(val_data), training_nums, batch_size),
        log_file, global_rank)

    # Trainer
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py
    # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    # https://www.deepspeed.ai/docs/config-json/
    # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    # https://huggingface.co/transformers/v4.10.1/main_classes/deepspeed.html
    # https://github.com/tatsu-lab/stanford_alpaca/issues/176
    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt",
                                                        padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    print_rank_0(f"Using {training_args.half_precision_backend} half precision backend", log_file, global_rank)
    # Train!
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps

    total_train_batch_size = training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print_rank_0("***** Running training *****", log_file, global_rank)
    print_rank_0(f"  Num examples = {num_examples}", log_file, global_rank)
    print_rank_0(f"  Num train samples = {num_train_samples}", log_file, global_rank)
    print_rank_0(f"  world_size = {world_size}", log_file, global_rank)
    print_rank_0(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}",
                 log_file, global_rank)
    print_rank_0(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}", log_file, global_rank)
    print_rank_0(f"  Total optimization steps = {max_steps}", log_file, global_rank)
    print_rank_0(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}", log_file,
                 global_rank)

    model.config.use_cache = False
    if training_args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=last_checkpoint)

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
                 log_file, global_rank)


if __name__ == "__main__":
    main()

