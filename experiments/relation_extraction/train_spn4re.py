# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_spn4re
    Author: czh
    Create Date: 2021/11/15
--------------------------------------
    Change Activity: 
======================================
"""

# TODO: 目前还有问题，模型没收敛，评价指标没没打印出来
import os
import glob
import time
from tqdm import tqdm

from dataclasses import dataclass, field
from transformers import HfArgumentParser, BertTokenizerFast, BertConfig, BertModel, get_scheduler
from transformers.optimization import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nlp.arguments import TrainingArguments, ModelArguments, DataArguments
from nlp.models.bert_spn4re import SetPredPlusBert
from nlp.metrics.spn4re_metric import metric
from nlp.utils.enums import MatcherType, OptimizerEnum, DataType, RunMode
from nlp.utils.functions import formulate_gold
from nlp.processors.spn4ner_processor import RelDataset, PreProcessor
from nlp.tools.common import init_logger, init_summary_writer, rotate_checkpoints, seed_everything, prepare_device


@dataclass
class MyArguments:
    relation_labels: str = field(
        metadata={"help": "relation labels"}
    )

    allow_null_entities_in_tuple: str = field(default=None)
    num_generated_tuples: int = 10
    num_entities_in_tuple: int = 2
    entity_loss_weight: str = field(default=None)
    num_decoder_layers: int = 3
    relation_loss_weight: float = 1.0
    na_rel_coef: float = 1.0  # 无关的relation的权重，正常relation的weight默认为1，需要综合考虑num_generated_tuples设置该值
    matcher: MatcherType = MatcherType.AVG

    optimizer_type: OptimizerEnum = field(
        default=OptimizerEnum.AdamW,
        metadata={"help":"Which optimizer used for training. Selected in ['Adamw', 'LAMB', 'Adafactor']."}
    )

    encoder_lr: float = 1e-5
    decoder_lr: float = 2e-5
    n_best_size: int = 100
    max_span_length: int = 30


parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments, MyArguments))
args = parser.parse_args()
args.relation_labels = list(args.relation_labels.split(','))
args.allow_null_entities_in_tuple = [int(i) for i in args.allow_null_entities_in_tuple.split(',')]
args.entity_loss_weight = [float(i) for i in args.entity_loss_weight.split(',')]

args.output_dir = args.output_dir + '{}'.format(args.model_type)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
LOGGER = init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')

if args.fp16 and args.fp16_backend == 'amp':
    from torch.cuda.amp import autocast, GradScaler
    SCALER = GradScaler()

WRITER = init_summary_writer(log_dir=args.logging_dir)


def init_env():
    cuda_number = str(args.cuda_number)
    n_gpu = len(cuda_number.split(','))

    device, list_ids = prepare_device(cuda_number)
    args.device = device
    args.n_gpu = n_gpu
    LOGGER.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    seed_everything(args.seed)


def init_model(num_labels, model_name_or_path=None):
    if model_name_or_path is None:
        model_name_or_path = args.model_name_or_path
    config = BertConfig.from_pretrained(args.config_name if args.config_name else model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name_or_path,
                                                  do_lower_case=args.do_lower_case,
                                                  cache_dir=args.cache_dir if args.cache_dir else None, )
    encoder = BertModel.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    if args.bert_frozen:
        encoder.embeddings.word_embeddings.weight.requires_grad = False
        encoder.embeddings.position_embeddings.weight.requires_grad = False
        encoder.embeddings.token_type_embeddings.weight.requires_grad = False
    if not args.allow_null_entities_in_tuple:
        # 默认不允许空实体存在
        args.allow_null_entities_in_tuple = [0] * args.num_entities_in_tuple
    model = SetPredPlusBert(
        encoder=encoder,
        num_relation_classes=num_labels,  # 模型中会自动映射多一个unknown类别
        num_generated_tuples=args.num_generated_tuples,
        num_entities_in_tuple=args.num_entities_in_tuple,
        num_decoder_layers=args.num_decoder_layers,
        entity_loss_weight=args.entity_loss_weight,
        relation_loss_weight=args.relation_loss_weight,
        na_rel_coef=args.na_rel_coef,
        matcher=args.matcher
    )
    return config, tokenizer, model


def load_model(num_labels, model_name_or_path):
    encoder = BertModel.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model = SetPredPlusBert(
        encoder=encoder,
        num_relation_classes=num_labels,  # 模型中会自动映射多一个unknown类别
        num_generated_tuples=args.num_generated_tuples,
        num_entities_in_tuple=args.num_entities_in_tuple,
        num_decoder_layers=args.num_decoder_layers,
        entity_loss_weight=args.entity_loss_weight,
        relation_loss_weight=args.relation_loss_weight,
        na_rel_coef=args.na_rel_coef,
        matcher=args.matcher
    )
    model.load_state_dict(torch.load(os.path.join(model_name_or_path, 'pytorch.bin'), map_location=args.device))
    return model


def init_optimizer(total, parameters):
    optimizer = AdamW(parameters, eps=args.adam_epsilon, weight_decay=args.weight_decay,)

    scheduler = get_scheduler(args.scheduler_type, optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                              num_training_steps=total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    return optimizer, scheduler


def get_parameters(model):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    component = ['encoder', 'decoder']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and component[0] in n],
            'weight_decay': args.weight_decay,
            'lr': args.encoder_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and component[0] in n],
            'weight_decay': 0.0,
            'lr': args.encoder_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and component[1] in n],
            'weight_decay': args.weight_decay,
            'lr': args.decoder_lr
        },
        {
            'params': [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and component[1] in n],
            'weight_decay': 0.0,
            'lr': args.decoder_lr
        }
    ]
    return optimizer_grouped_parameters


def data_generator(processor: PreProcessor, data_type: DataType = DataType.TRAIN):
    if data_type == DataType.TRAIN:
        samples = processor.parse_from_pos_json_files(paths=os.path.join(args.data_dir, "train.json"),
                                                    data_type=data_type)
        num = len(samples)
        print(samples[0])
        data_sets = RelDataset(samples)
        data_loader = DataLoader(
            dataset=data_sets,
            batch_size=args.per_gpu_train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            collate_fn=RelDataset.collate_fn,
            pin_memory=True
        )
        return data_loader, num
    elif data_type == DataType.VAL:
        samples = processor.parse_from_pos_json_files(paths=os.path.join(args.data_dir, "dev.json"),
                                                      data_type=data_type)
        print(samples[0])
        num = len(samples)
        data_sets = RelDataset(samples)
        data_loader = DataLoader(
            dataset=data_sets,
            batch_size=args.per_gpu_eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=RelDataset.collate_fn,
            pin_memory=True
        )
        return data_loader, num
    else:
        raise ValueError("'data_type' must be 'DataType.TRAIN' or 'DataType.VAL'")


def train(model, train_dataloader, eval_dataloader, tokenizer):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model)
    args.warmup_steps = int(t_total * args.warmup_ratio)
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters)

    model.to(args.device)
    # Train!
    LOGGER.info("***** Running training *****")
    LOGGER.info("  Num examples = %d", args.train_sample_num)
    LOGGER.info("  Num Epochs = %d", args.num_train_epochs)
    LOGGER.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    LOGGER.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    LOGGER.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        LOGGER.info("  Continuing training from checkpoint, will skip to saved global_step")
        LOGGER.info("  Continuing training from epoch %d", epochs_trained)
        LOGGER.info("  Continuing training from global step %d", global_step)
        LOGGER.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    model.train()
    model.zero_grad()
    best_score = 0.0
    best_epoch = 0
    for epoch in range(int(args.num_train_epochs)):
        tr_loss, logging_loss = 0.0, 0.0
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch {epoch} "):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs = {"input_ids": batch["input_ids"].to(args.device),
                      "attention_mask": batch["attention_mask"].to(args.device),
                      "token_type_ids": batch["token_type_ids"].to(args.device),
                      "targets": batch["targets"]}
            if args.fp16 and args.fp16_backend == 'amp':
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            loss_dict = outputs["loss"]
            loss = loss_dict["total"]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16 and args.fp16_backend == 'amp':
                SCALER.scale(loss).backward()
            else:
                loss.backward()
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                WRITER.add_scalar("Loss/Train", loss.item(), global_step=global_step)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16 and args.fp16_backend == 'amp':
                    SCALER.step(optimizer)
                    SCALER.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    # Log metrics
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        result = evaluate(model, eval_dataloader)
                        if args.greater_is_better:
                            score = result[args.metric_for_best_model]
                            if score > best_score:
                                best_score = score
                                best_epoch = epoch
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                output_dir = os.path.join(args.output_dir, "best_model")
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                LOGGER.info(f"Save best model to {output_dir}")
                                torch.save(model_to_save, output_dir+'/pytorch.bin')
                                tokenizer.save_vocabulary(output_dir)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    torch.save(model_to_save, output_dir+'/pytorch.bin')
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    LOGGER.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    LOGGER.info("Saving optimizer and scheduler states to %s", output_dir)
                    if args.save_total_limit:
                        rotate_checkpoints(save_total_limit=args.save_total_limit, output_dir=args.output_dir)
        LOGGER.info(f"total loss: {tr_loss / global_step}")
        LOGGER.info("\n")
    return best_score, best_epoch


def evaluate(model, eval_dataloader, prefix=""):
    LOGGER.info("***** Running evaluation %s *****" % prefix)
    LOGGER.info("  Num examples = %d", args.eval_sample_num)
    LOGGER.info("  Batch size = %d", args.per_gpu_eval_batch_size)

    prediction, gold = {}, {}
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluation"):
        gen_tuples = model.gen_tuples(
            input_ids=batch["input_ids"].to(args.device),
            token_type_ids=batch["token_type_ids"].to(args.device),
            attention_mask=batch["attention_mask"].to(args.device),
            sent_lens=batch["sent_lens"],
            sent_idx=batch["sent_idx"],
            n_best_size=args.n_best_size,
            max_span_length=args.max_span_length,
            allow_null_entities_in_tuple=args.allow_null_entities_in_tuple,
            run_mode=RunMode.EVAL
        )

        output = {
            "gold": formulate_gold(batch["targets"], sent_indices=batch["sent_idx"]),
            "pred": gen_tuples
        }
        gold.update(output["gold"])
        prediction.update(output["pred"])

    res = metric(prediction, gold, verbose=True)
    return res


def main():
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    init_env()
    relation_labels = args.relation_labels

    config, tokenizer, model = init_model(len(relation_labels))
    model.to(args.device)

    preprocessor = PreProcessor(
        tokenizer=tokenizer,
        relation_labels=relation_labels,
        num_entities_in_tuple=args.num_entities_in_tuple,
        max_seq_len=args.max_seq_length,
        sliding_len=args.sliding_len
    )

    train_dataloader, train_num = data_generator(preprocessor, DataType.TRAIN)
    args.train_sample_num = train_num
    eval_dataloader, eval_num = data_generator(preprocessor, DataType.VAL)
    args.eval_sample_num = eval_num
    LOGGER.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        best_score, best_epoch = train(model, train_dataloader, eval_dataloader, tokenizer)
        LOGGER.info(f"best_epoch = %s, best_{args.metric_for_best_model} = %s", best_epoch, best_score)
        # Evaluation
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            best_model_path = os.path.join(args.output_dir, "best_model")
            checkpoints = [best_model_path]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in
                    sorted(glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True))
                )
            LOGGER.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
                eval_model = load_model(num_labels=len(relation_labels), model_name_or_path=checkpoint)
                eval_model.to(args.device)
                result = evaluate(eval_model, eval_dataloader, prefix)
                if global_step:
                    result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
                results.update(result)

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as fw:
                for key in sorted(results.keys()):
                    fw.write("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    main()
