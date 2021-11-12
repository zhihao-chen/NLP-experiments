# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: train_ee
    Author: czh
    Create Date: 2021/9/8
--------------------------------------
    Change Activity: 
======================================
"""
import os
import glob
import json
import re
import time
import codecs
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
from accelerate import Accelerator

from nlp.processors.ee_seq import convert_examples_to_features, EEProcessor, collate_fn, tokenize
from nlp.processors.utils_ee import get_argument_for_seq
from nlp.models.bert_for_ee import MODEL_TYPE_CLASSES
from nlp.metrics.metric import get_prf_scores
from nlp.tools.common import init_logger, seed_everything, prepare_device
from nlp.processors.preprocess import is_all_alpha
from experiments.train_args import get_argparse


def init_env():
    parser = get_argparse()
    args_ = parser.parse_args()
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    accelerator_ = Accelerator(fp16=args_.fp16)

    if not os.path.exists(args_.logging_dir):
        os.mkdir(args_.logging_dir)

    seed_everything(args_.seed)

    if not os.path.exists(args_.output_dir):
        os.mkdir(args_.output_dir)
    args_.output_dir = args_.output_dir + '{}'.format(args_.model_type)
    if not os.path.exists(args_.output_dir):
        os.mkdir(args_.output_dir)
    if os.path.exists(args_.output_dir) and os.listdir(
            args_.output_dir) and args_.do_train and not args_.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args_.output_dir))
    logger_ = init_logger(log_file=args_.output_dir + f'/{args_.model_type}-{args_.task_name}-{time_}.log')

    cuda_number = str(args_.cuda_number)
    n_gpu = len(cuda_number.split(','))
    if n_gpu > 1 and not args_.no_cuda:
        device = accelerator_.device
    else:
        device, list_ids = prepare_device(cuda_number)
    args_.device = device
    args_.n_gpu = n_gpu
    logger_.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args_.local_rank, args_.device, args_.n_gpu, bool(args_.local_rank != -1), args_.fp16, )

    args_.task_name = args_.task_name.lower()

    summary_writer = SummaryWriter(log_dir=args_.logging_dir)

    ee_processor = EEProcessor(args_.data_dir)
    id2label, label2id, num_labels, event_type_dict = ee_processor.get_labels()
    args_.id2label = id2label
    args_.label2id = label2id
    args_.num_labels = num_labels

    logger_.info("label2id: ", label2id)

    return summary_writer, ee_processor, logger_, args_, accelerator_


writer, processor, logger, args, accelerator = init_env()


def init_ee_model(model_name_or_path=None):
    if model_name_or_path is None:
        model_name_or_path = args.model_name_or_path
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_class = MODEL_TYPE_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else model_name_or_path,
                                          num_labels=args.num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config, train_config=args,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    return config, tokenizer, model


def init_optimizer(total, parameters):
    optimizer = AdamW(parameters, lr=args.learning_rate, eps=args.adam_epsilon)
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
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    return optimizer_grouped_parameters


def load_examples(data_type="train"):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    if data_type == 'train':
        examples = processor.get_train_examples()
    elif data_type == 'dev':
        examples = processor.get_dev_examples()
    else:
        examples = processor.get_test_examples()
    return examples


def load_and_cache_examples(tokenizer, examples, data_type='train', max_seq_length=512):
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label2id=args.label2id,
                                                max_seq_length=max_seq_length,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                data_type=data_type
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


def train(model, tokenizer):
    """ Train the model """
    train_examples = load_examples(data_type="train")
    train_dataset = load_and_cache_examples(tokenizer, train_examples, data_type='train')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_grouped_parameters = get_parameters(model)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer, scheduler = init_optimizer(t_total, optimizer_grouped_parameters)
    if args.n_gpu > 1:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    else:
        model.to(args.device)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    model.zero_grad()
    best_f1 = 0.0
    best_epoch = 0
    tolerance_num = 0
    total_loss = 0
    for epoch in range(int(args.num_train_epochs)):
        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch {epoch}-loss {total_loss} "):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)

            outputs = model(**inputs)
            loss, logits = outputs  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step=global_step)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.local_rank in [-1, 0] and args.evaluate_during_training and args.do_eval_per_epoch:
            # Log metrics
            # Only evaluate when single GPU otherwise metrics may not average well
            logger.info(f"Evaluating model at {epoch+1} time")
            result = evaluate(model, tokenizer)
            f1 = result["argument_hard"]["argument_hard_f1"]
            writer.add_scalar("Loss/eval", result['loss'], global_step=global_step)
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                tolerance_num = 0
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_vocabulary(args.output_dir)
            else:
                tolerance_num += 1

        total_loss = tr_loss/global_step
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if args.early_stop and tolerance_num > args.tolerance:
            break
    return best_f1, best_epoch


def evaluate(model, tokenizer, data_type="dev", prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_examples = load_examples(data_type)
    eval_dataset = load_and_cache_examples(tokenizer, eval_examples, data_type=data_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    event_type_results = [0, 0, 0]  # [correct_num, pred_num, gold_num]
    argument_hard_results = [0, 0, 0]  # 严格匹配event_type, role, argument
    argument_soft_result = [0, 0, 0]  # 部分匹配role, argument
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluation"):
        batch_eval_example = eval_examples[step*args.eval_batch_size: (step+1)*args.eval_batch_size]
        assert len(batch_eval_example) <= args.eval_batch_size

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            mask = inputs['attention_mask']
            tags = model.crf.decode(logits, mask)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        tags = tags.squeeze(0).cpu().numpy().tolist()
        for i, example in enumerate(batch_eval_example):
            gold_arguments_hard = set()
            gold_arguments_soft = set()
            gold_event_type = set()
            for k, v in example.arguments.items():
                gold_event_type.add(v[0])
                gold_arguments_hard.add('-'.join(v+(k,)))
                gold_arguments_soft.add('-'.join([v[1], k]))

            text = example.text_a
            pred_entities = get_argument_for_seq(tags[i][1:-1], args.id2label)

            pred_arguments_hard = set()
            pred_arguments_soft = set()
            pred_event_type = set()
            for tag, s, e in pred_entities:
                t = text[s: e+1]
                if is_all_alpha(t):
                    if re.search(r"[A-Z]", text[s-1]):
                        t = text[s-1: e+1]
                pred_event_type.add(tag[0])
                pred_arguments_hard.add('-'.join(tag+(t,)))
                pred_arguments_soft.add('-'.join([tag[1], t]))

            correct_arg_num_hard = len(pred_arguments_hard.intersection(gold_arguments_hard))
            correct_arg_num_soft = len(pred_arguments_soft.intersection(gold_arguments_soft))
            correct_event_type_num = len(pred_event_type.intersection(gold_event_type))

            event_type_results[0] += correct_event_type_num
            event_type_results[1] += len(pred_event_type)
            event_type_results[2] += len(gold_event_type)

            argument_hard_results[0] += correct_arg_num_hard
            argument_hard_results[1] += len(pred_arguments_hard)
            argument_hard_results[2] += len(gold_arguments_hard)

            argument_soft_result[0] += correct_arg_num_soft
            argument_soft_result[1] += len(pred_arguments_soft)
            argument_soft_result[2] += len(gold_arguments_soft)

    eval_loss = eval_loss / nb_eval_steps
    results = {"loss": eval_loss,
               "event_type": get_prf_scores(*event_type_results, eval_type="event_type"),
               "argument_hard": get_prf_scores(*argument_hard_results, eval_type="argument_hard"),
               "argument_soft": get_prf_scores(*argument_soft_result, eval_type="argument_soft")}

    logger.info("***** Eval results for %s datasets*****", data_type)
    info = json.dumps(results, ensure_ascii=False, indent=2)
    logger.info(info)
    return results


def predict(model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_examples = load_examples(data_type='test')
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", 1)

    if isinstance(model, nn.DataParallel):
        model = model.module

    output_predict_file = os.path.join(pred_output_dir, prefix, "test_crf_prediction.json")
    with codecs.open(output_predict_file, 'w', encoding='utf8') as fw:
        for step, example in tqdm(enumerate(test_examples), desc="Predicting"):
            text = example.text_a
            tokens = tokenize(text, tokenizer.vocab)
            if len(tokens) > args.eval_max_seq_length - 2:
                tokens = tokens[: (args.eval_max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            input_ids = torch.LongTensor([input_ids]).to(args.device)
            input_mask = torch.LongTensor([input_mask]).to(args.device)
            segment_ids = torch.LongTensor([segment_ids]).to(args.device)
            model.eval()
            with torch.no_grad():
                inputs = {"input_ids": input_ids, "attention_mask": input_mask, "labels": None}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
                outputs = model(**inputs)
                logits = outputs[0]
                tags = model.crf.decode(logits, inputs['attention_mask'])
                tags = tags.squeeze(0).cpu().numpy().tolist()
            preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
            label_entities = get_argument_for_seq(preds, args.id2label)
            pred_arguments = {text[s: e+1]: (tag, s, e) for tag, s, e in label_entities}
            result = {"text": text, "entity_list": []}
            for k, lst in pred_arguments.items():
                v = lst[0]
                result["entity_list"].append({
                    'event_type': v[0],
                    'arguments': [{
                        'role': v[1],
                        'argument': k,
                        "start": lst[1],
                        "end": lst[2]
                    }]
                })
            fw.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    config, tokenizer, model = init_ee_model()

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        best_f1, best_epoch = train(model, tokenizer)
        logger.info("best_epoch = %s, best_f1 = %s", best_epoch, best_f1)

    if args.do_eval and args.local_rank in [-1, 0]:
        # Evaluation
        results = {}
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True)
            )
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            config, tokenizer, model = init_ee_model(checkpoint)
            model.to(args.device)
            result = evaluate(model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as fw:
            for key in sorted(results.keys()):
                fw.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict_no_tag and args.local_rank in [-1, 0]:
        config, tokenizer, model = init_ee_model(args.output_dir)
        model.to(args.device)
        predict(model, tokenizer)
    elif args.do_predict_tag and args.local_rank in [-1, 0]:
        config, tokenizer, model = init_ee_model(args.output_dir)
        model.to(args.device)
        evaluate(model, tokenizer, data_type='test')


if __name__ == "__main__":
    main()
