# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_globalpointer
    Author: czh
    Create Date: 2022/2/9
--------------------------------------
    Change Activity: 
======================================
"""
import os
import regex
import glob
import json
import time
from tqdm import tqdm
import typing
import codecs
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, HfArgumentParser, get_scheduler

from nlp.models.bert_for_ner import GlobalPointerForNER
from nlp.metrics.metric import MetricsCalculator
from nlp.tools.common import (init_logger, prepare_device, seed_everything, rotate_checkpoints, init_summary_writer)
from nlp.processors.dataset import NerDataset
from nlp.utils.enums import DataType
from nlp.processors.global_pointer_processor import CluenerProcessor, global_pointer_entity_extract, SPE
from nlp.callback.optimizers.child_tuning_optimizer import ChildTuningAdamW
from nlp.callback.child_tuning_fisher import calculate_fisher
from nlp.arguments import TrainingArguments, DataArguments, ModelArguments

parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
args = parser.parse_args()
args.output_dir = args.output_dir + '{}'.format(args.model_type)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
LOGGER = init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')

if args.fp16 and args.fp16_backend == 'amp':
    from torch.cuda.amp import autocast, GradScaler
    SCALER = GradScaler()

if not os.path.exists(args.logging_dir):
    os.mkdir(args.logging_dir)
WRITER = init_summary_writer(log_dir=args.logging_dir)
METRIC = MetricsCalculator()


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


def init_model(num_labels, encoder_name_or_path, model_name_or_path=None):
    config = BertConfig.from_pretrained(args.config_name if args.config_name else encoder_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name if args.tokenizer_name else encoder_name_or_path,
                                                  do_lower_case=args.do_lower_case,
                                                  cache_dir=args.cache_dir if args.cache_dir else None)
    inner_dim = config.hidden_size // config.num_attention_heads
    if not model_name_or_path:
        model = GlobalPointerForNER(config=config, encoder_model_path=encoder_name_or_path, num_labels=num_labels,
                                    head_size=inner_dim, rope=args.rope, efficient=False)
    else:
        model = GlobalPointerForNER(config=config, num_labels=num_labels, head_size=inner_dim, rope=args.rope,
                                    efficient=False)
        model.load_state_dict(
            torch.load(os.path.join(model_name_or_path, 'pytorch_model.bin'),
                       map_location=args.device)
        )
    return tokenizer, model


def init_optimizer(total, parameters):
    if not args.mode:
        optimizer = AdamW(parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                          betas=(args.adam_beta1, args.adam_beta2))
    else:
        optimizer_cls = ChildTuningAdamW
        optimizer_kwargs = {"betas": (args.adam_beta1, args.adam_beta2),
                            "eps": args.adam_epsilon, "lr": args.learning_rate}
        optimizer = optimizer_cls(parameters, mode=args.mode, **optimizer_kwargs)
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
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.encoder.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    # optimizer_grouped_parameters = model.parameters()
    return optimizer_grouped_parameters


def load_features(processor: CluenerProcessor, tokenizer, data_type: DataType = DataType.TRAIN):
    if data_type == DataType.TRAIN:
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        samples = processor.parser_data_from_json_file(tokenizer, data_type)
        train_data = NerDataset(samples, args.num_labels, 'train')
        data_loader = DataLoader(train_data, shuffle=True, batch_size=args.per_gpu_train_batch_size,
                                 collate_fn=train_data.collate_fn, num_workers=args.dataloader_num_workers)
        LOGGER.info(f"  Num examples of {data_type.value} = %d", len(samples))
        return data_loader
    elif data_type == DataType.EVAL:
        samples = processor.parser_data_from_json_file(tokenizer, data_type)
        eval_data = NerDataset(samples, args.num_labels, 'eval')
        data_loader = DataLoader(eval_data, shuffle=False, batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=eval_data.collate_fn, num_workers=args.dataloader_num_workers)
        LOGGER.info(f"  Num examples of {data_type.value} = %d", len(samples))
        return data_loader
    else:
        samples = processor.parser_data_from_json_file(tokenizer, data_type)
        return samples


def evaluate(model, eval_dataloader, prefix='') -> typing.Tuple[dict, dict]:
    METRIC.reset()
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # Eval!
    LOGGER.info("***** Running evaluation %s *****", prefix)

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    f1, precision, recall = 0.0, 0.0, 0.0
    total_num = len(eval_dataloader)
    tr_loss = 0.0
    global_steps = 0
    for i, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
        with torch.no_grad():
            inputs = {"input_ids": None, "attention_mask": None,
                      "token_type_ids": None, "labels": None}
            for k, v in batch.items():
                if k in inputs:
                    inputs[k] = v.to(args.device)
            output = model(**inputs)
        logits = output['logits']
        loss = output['loss']
        tr_loss += loss.item()
        global_steps += 1
        METRIC.update(inputs['labels'], logits)
        sample_f1, sample_p, sample_r = METRIC.get_evaluate_fpr(inputs['labels'], logits)
        f1 += sample_f1
        precision += sample_p
        recall += sample_r

    avg_loss = tr_loss / global_steps
    avg_f1 = f1 / total_num
    avg_precision = precision / total_num
    avg_recall = recall / total_num

    result = {"loss": avg_loss, "f1": avg_f1, "precision": avg_precision, "recall": avg_recall}
    LOGGER.info("*********** Evaluation result *************")
    LOGGER.info(json.dumps(result, ensure_ascii=False, indent=2))
    eval_info, entity_info = METRIC.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    info = "-".join([f' {key}: {value} ' for key, value in results.items()])
    LOGGER.info(info)

    return result, entity_info


def train(model, train_dataloader, eval_dataloader):
    """ Train the model """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model)
    args.warmup_steps = int(t_total * args.warmup_ratio)
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters)
    if args.mode == 'ChildTuning-D':
        gradient_mask = calculate_fisher(model, train_dataloader, device=args.device, reserve_p=args.reserve_p)
        optimizer.set_gradient_mask(gradient_mask)

    # Train!
    LOGGER.info("***** Running training *****")
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
    LOGGER.info("******************************** Training *********************************")
    for epoch in range(int(args.num_train_epochs)):
        tr_loss = 0.0
        epoch_steps = 0.0
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch {epoch} "):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs = {"input_ids": None, "attention_mask": None,
                      "token_type_ids": None, "labels": None}
            for k, v in batch.items():
                if k in inputs:
                    inputs[k] = v.to(args.device)
            if args.fp16 and args.fp16_backend == 'amp':
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            loss = outputs['loss']
            logits = outputs['logits']
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16 and args.fp16_backend == 'amp':
                SCALER.scale(loss).backward()
            else:
                loss.backward()
            sample_f1 = METRIC.get_sample_f1(logits, inputs['labels'])
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                LOGGER.info(f"Training epoch: {epoch}\tsteps: {step + 1}\tloss: {loss.item()}\tf1: {sample_f1}")
                WRITER.add_scalar("Loss/Train", loss.item(), global_step=global_step)
                WRITER.add_scalar("f1/Train", sample_f1, global_step=global_step)
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
            epoch_steps += 1
            if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                # Log metrics
                if args.local_rank == -1:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    result, _ = evaluate(model, eval_dataloader)
                    WRITER.add_scalar("Loss/Eval", result['loss'], global_step=global_step)
                    WRITER.add_scalar("f1/Eval", result['f1'], global_step=global_step)
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

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                LOGGER.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                LOGGER.info("Saving optimizer and scheduler states to %s", output_dir)

                if args.save_total_limit:
                    rotate_checkpoints(save_total_limit=args.save_total_limit, output_dir=args.output_dir)
        LOGGER.info(f"total loss: {tr_loss / epoch_steps}")
        LOGGER.info("\n")
    return best_epoch, best_score


def predict(model, samples, id2entity, entity_type_name_dict):
    LOGGER.info("****** Testing ******")
    LOGGER.info("****** Num test datas %s ******" % len(samples))

    test_dataset = NerDataset(samples, len(id2entity), 'test')
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_gpu_eval_batch_size,
                            collate_fn=test_dataset.collate_fn)

    pred_results = []

    model.eval()
    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
        with torch.no_grad():
            inputs = {"input_ids": None, "attention_mask": None,
                      "token_type_ids": None, "labels": None}
            sample_list = batch['sample_list']
            for k, v in batch.items():
                if k in inputs and k != 'labels':
                    inputs[k] = v.to(args.device)

            output = model(**inputs)
        logits = output['logits'][:, :, 1:-1, 1:-1]
        res = global_pointer_entity_extract(pred_logits=logits, id2entity=id2entity,
                                            entity_type_names=entity_type_name_dict)
        assert len(res) == len(sample_list)
        pred_results.extend(res)
    assert len(pred_results) == len(samples)
    results = []
    i = 0
    for sample, res_lst in zip(samples, pred_results):
        text = sample.text
        text = regex.sub(rf'{SPE}', ' ', text)
        offset_mapping = sample.offset_mapping
        nl = len(offset_mapping)
        entity_set = defaultdict()
        entity_list = []
        if res_lst:
            res_lst.sort(key=lambda x: x['start'])
            for j, item in enumerate(res_lst):
                label = item['label']
                token_start = item['start']
                token_end = item['end']
                # print(token_start, token_end, nl, offset_mapping)
                # print("*"*20)
                char_start = offset_mapping[token_start][0]
                if token_end + 1 < nl:
                    char_end = offset_mapping[token_end + 1][1]
                else:
                    char_end = offset_mapping[token_end][1]
                entity = text[char_start: char_end]

                if entity not in entity_set:
                    entity_set[entity] = entity_set.get(entity, 0) + 1
                    adict = {
                        'id': str(i) + '_' + str(j),
                        'entity': entity,
                        'label': label,
                        'label_name': item['label_name'],
                        'start': char_start,
                        'end': char_end
                    }
                    entity_list.append(adict)
        results.append({'text': text, 'entity_list': entity_list})
        i += 1

    file_path = os.path.join(args.output_dir, "test_predicted.json")
    with codecs.open(file_path, 'w', encoding='utf8') as fw:
        for res in results:
            fw.write(json.dumps(res, ensure_ascii=False) + '\n')
        if os.path.exists(file_path):
            print(f"save predict results into {file_path}")
        else:
            print(f"{file_path} not exist")


def main():
    init_env()
    preprocessor = CluenerProcessor(
                                 data_dir=args.data_dir,
                                 add_special_tokens=False)
    entity_types, entity_types_to_name_dict, entity2id, id2entity = preprocessor.get_labels()
    args.num_labels = len(entity_types)
    METRIC.set_id2labels(id2entity)

    tokenizer, model = init_model(num_labels=args.num_labels, encoder_name_or_path=args.model_name_or_path)
    model.to(args.device)

    if args.do_train:
        train_dataloader = load_features(preprocessor, tokenizer, DataType.TRAIN)
        eval_dataloader = load_features(preprocessor, tokenizer, DataType.EVAL)

        best_epoch, best_score = train(model, train_dataloader, eval_dataloader)
        LOGGER.info(f"best_epoch = %s, best_{args.metric_for_best_model} = %s", best_epoch, best_score)

    results = {}
    entities_info = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        eval_dataloader = load_features(preprocessor, tokenizer, DataType.EVAL)
        best_model_path = os.path.join(args.output_dir, "best_model")
        checkpoints = [best_model_path]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True))
            )
        LOGGER.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            tokenizer, model = init_model(num_labels=args.num_labels,
                                          encoder_name_or_path=args.model_name_or_path,
                                          model_name_or_path=checkpoint)
            model.to(args.device)

            result, entity_info = evaluate(model, eval_dataloader, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
            entities_info.update(entity_info)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as fw:
            for key in sorted(results.keys()):
                fw.write("{} = {}\n".format(key, str(results[key])))
            fw.write('\n')
            for key in sorted(entities_info.keys()):
                infos = []
                for k, v in entities_info[key].items():
                    infos.append(f"{k}: {v}")
                info = f"{key}\t{' '.join(infos)}"
                fw.write(info + '\n')
    if args.do_predict_no_tag and args.local_rank in [-1, 0]:
        samples = load_features(preprocessor, tokenizer, DataType.TEST)
        best_model_path = os.path.join(args.output_dir, "best_model")
        tokenizer, model = init_model(num_labels=args.num_labels,
                                      encoder_name_or_path=args.model_name_or_path,
                                      model_name_or_path=best_model_path)
        model.to(args.device)
        predict(model, samples, id2entity, entity_types_to_name_dict)


if __name__ == "__main__":
    main()

