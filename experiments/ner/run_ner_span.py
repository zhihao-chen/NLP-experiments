# -*- coding: utf8 -*-
"""
======================================
    Project Name: NLP
    File Name: run_ner_span
    Author: czh
    Create Date: 2021/8/18
--------------------------------------
    Change Activity: 
======================================
"""
import os
import glob
import json
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (get_scheduler, BertTokenizer, set_seed, BertConfig,
                          AlbertConfig, AlbertTokenizer, HfArgumentParser)

from nlp.callback.optimizers.adamw import AdamW
from nlp.callback.adversarial import FGM
from nlp.models.nezha import NeZhaConfig
from nlp.models.bert_for_ner import BertSpanForNer, AlbertSpanForNer
from nlp.processors.utils_ner import bert_extract_item
from nlp.processors.ner_span import convert_examples_to_features, ner_processors, collate_fn
from nlp.metrics.metric import SpanEntityScore
from nlp.tools.common import init_logger, json_to_text, prepare_device, init_summary_writer, rotate_checkpoints
from nlp.arguments import DataArguments, ModelArguments, TrainingArguments

MODEL_CLASSES = {
    # bert ernie bert_wwm bert_wwwm_ext
    'nezha': (NeZhaConfig, BertTokenizer, BertSpanForNer),
    'bert': (BertConfig, BertTokenizer, BertSpanForNer),
    'albert': (AlbertConfig, AlbertTokenizer, AlbertSpanForNer)
}
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


def init_model(num_labels, model_name_or_path=None):
    if model_name_or_path is None:
        model_name_or_path = args.model_name_or_path
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path),
                                        config=config, train_args=args,
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
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = model.bert.named_parameters()
    start_parameters = model.start_fc.named_parameters()
    end_parameters = model.end_fc.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         'lr': args.learning_rate},

        {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         'lr': 0.001},

        {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': 0.001},
        {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         'lr': 0.001}
    ]
    return optimizer_grouped_parameters


def load_and_cache_examples(task, tokenizer, data_type='train'):
    processor = ner_processors[args.task_name](args.data_dir)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_span-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        LOGGER.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        LOGGER.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples()
        elif data_type == 'dev':
            examples = processor.get_dev_examples()
        else:
            examples = processor.get_test_examples()
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            LOGGER.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    if data_type in ['dev', 'test']:
        return features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_input_lens)
    return dataset


def train(model, tokenizer):
    train_dataset = load_and_cache_examples(args.task_name, tokenizer, data_type='train')
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
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
    LOGGER.info("  Num examples = %d", len(train_dataset))
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
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
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
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3], "end_positions": batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            if args.fp16 and args.fp16_backend == 'amp':
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            loss = outputs["loss"]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16 and args.fp16_backend == 'amp':
                SCALER.scale(loss).backward()
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()  # noqa
                loss_adv = model(**inputs)["loss"]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                # loss_adv.backward()
                if args.fp16 and args.fp16_backend == 'amp':
                    SCALER.scale(loss_adv).backward()
                else:
                    loss_adv.backward()
                fgm.restore()
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
                        result = evaluate(model, tokenizer)
                        WRITER.add_scalar("Loss/Eval", result['loss'], global_step=global_step)
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
                                model_to_save.save_pretrained(output_dir)
                                tokenizer.save_vocabulary(output_dir)
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
                    tokenizer.save_vocabulary(output_dir)
                    LOGGER.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    LOGGER.info("Saving optimizer and scheduler states to %s", output_dir)
                    if args.save_total_limit:
                        rotate_checkpoints(save_total_limit=args.save_total_limit, output_dir=args.output_dir)
        LOGGER.info("\n")

    return best_epoch, best_score


def evaluate(model, tokenizer, data_type="dev", prefix=""):
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args.task_name, tokenizer, data_type=data_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Eval!
    LOGGER.info("***** Running evaluation %s *****", prefix)
    LOGGER.info("  Num examples = %d", len(eval_features))
    LOGGER.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    for step, f in tqdm(enumerate(eval_features), desc="Evaluating"):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        subjects = f.subjects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids, "end_positions": end_ids}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        tmp_eval_loss, start_logits, end_logits = outputs["loss"], outputs["start_logits"], outputs["end_logits"]
        r = bert_extract_item(start_logits, end_logits)
        t = subjects
        metric.update(true_subject=t, pred_subject=r)
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

    LOGGER.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    LOGGER.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    LOGGER.info(info)
    LOGGER.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        LOGGER.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        LOGGER.info(info)
    return results


def predict(model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args.task_name, tokenizer, data_type='test')
    print(len(test_dataset))
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    LOGGER.info("***** Running prediction %s *****", prefix)
    LOGGER.info("  Num examples = %d", len(test_dataset))
    LOGGER.info("  Batch size = %d", 1)

    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_predict.json")
    for step, f in tqdm(enumerate(test_dataloader), desc="Predicting"):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)

        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": None, "end_positions": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]
        r = bert_extract_item(start_logits, end_logits)
        if r:
            label_entities = [[args.id2label[x[0]], x[1], x[2]] for x in r]
        else:
            label_entities = []
        json_d = {'id': step, 'entities': label_entities}
        results.append(json_d)

    LOGGER.info("\n")
    with open(output_predict_file, "w") as fw:
        for record in results:
            fw.write(json.dumps(record) + '\n')
    if args.task_name == "cluener":
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_span_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir, "test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {'id': x['id'], 'label': {}}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file, test_submit)


def main():
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    n_gpu = str(args.cuda_number)
    device, list_ids = prepare_device(n_gpu)
    args.device = device
    args.n_gpu = len(n_gpu.split(','))
    LOGGER.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    set_seed(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in ner_processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = ner_processors[args.task_name](args.data_dir)
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    args.num_labels = num_labels

    config, tokenizer, model = init_model(num_labels)
    LOGGER.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        best_epoch, best_f1 = train(model, tokenizer)
        LOGGER.info("best_epoch = %s, best_f1 = %s", best_epoch, best_f1)

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
            config, tokenizer, model = init_model(num_labels, checkpoint)
            model.to(args.device)
            if args.do_predict_tag:
                result = evaluate(model, tokenizer, data_type="test", prefix=prefix)
            else:
                result = evaluate(model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as fw:
            for key in sorted(results.keys()):
                fw.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict_no_tag and args.local_rank in [-1, 0]:
        best_model_path = os.path.join(args.output_dir, "best_model")
        config, tokenizer, model = init_model(num_labels, best_model_path)
        model.to(args.device)
        predict(model, tokenizer)


if __name__ == "__main__":
    main()
