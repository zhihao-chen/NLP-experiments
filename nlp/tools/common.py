import os
import re
import random
import json
import pickle
from collections import OrderedDict
from pathlib import Path
import logging
from typing import Union, List
import shutil

import torch
import numpy as np
import torch.nn as nn

LOGGER = logging.getLogger()


def init_summary_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=log_dir)
    return writer


def init_wandb_writer(project_name, train_args, group_name: str = "", experiment_name: str = ""):
    import wandb

    writer = wandb.init(project=project_name,
                        group=group_name,
                        name=experiment_name,
                        config=train_args)
    return writer, wandb.run


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit, use_mtime=False, output_dir=None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    # we don't do to allow resuming.
    save_total_limit = save_total_limit

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        LOGGER.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint)


def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")
    return


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    Example:
        init_logger(log_file)
        logger.info("abc'")
    """
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def enable_benchmark():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True
    enable_benchmark()


def seed_everything(seed=1029):
    """
    设置整个开发环境的seed
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    set_env()


def prepare_device(n_gpu_use: Union[List, str] = None):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if not n_gpu_use or not torch.cuda.is_available():
        device_type = 'cpu'
    else:
        if isinstance(n_gpu_use, str):
            n_gpu_use = n_gpu_use.split(",")
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        LOGGER.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are " \
              f"available on this machine."
        LOGGER.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(model, n_gpu=None):
    """
    判断环境 cpu还是gpu
    支持单机多卡
    :param n_gpu:
    :param model:
    :return:
    """
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        LOGGER.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device


def restore_checkpoint(resume_path, model=None):
    """
    加载模型
    :param resume_path:
    :param model:
    :return:
    注意： 如果是加载Bert模型的话，需要调整，不能使用该模式
    可以使用模块自带的Bert_model.from_pretrained(state_dict = your save state_dict)
    """
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model, best, start_epoch]


def save_pickle(data, file_path):
    """
    保存成pickle文件
    :param data:
    :param file_path:
    :return:
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    """
    读取pickle文件
    :param input_file:
    :return:
    """
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    """
    保存成json文件
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def save_numpy(data, file_path):
    """
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path), data)


def load_numpy(file_path):
    """
    加载.npy文件
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))


def load_json(file_path):
    """
    加载json文件
    :param file_path:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


def json_to_text(file_path, data):
    """
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def save_model(model, model_path):
    """ 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_path:
    :return:
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path):
    """
    加载模型
    :param model:
    :param model_path:
    :return:
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


class AverageMeter(object):
    """
    computes and stores the average and current value
    Example:
        loss = AverageMeter()
        for step,batch in enumerate(train_data):
             pred = self.model(batch)
             raw_loss = self.metrics(pred,target)
             loss.update(raw_loss.item(),n = 1)
        cur_loss = loss.avg
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def summary(model, *inputs, batch_size=-1, show_input=True):
    """
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        print("model summary info: ")
        for step,batch in enumerate(train_data):
            summary(self.model,*batch,show_input=True)
            break
    """

    def register_hook(module):
        def hook(module_, input_, output=None):
            class_name = str(module_.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary_)

            m_key = f"{class_name}-{module_idx + 1}"
            summary_[m_key] = OrderedDict()
            summary_[m_key]["input_shape"] = list(input_[0].size())
            summary_[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary_[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary_[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary_[m_key]["output_shape"] = list(output.size())
                    summary_[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module_, "weight") and hasattr(module_.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module_.weight.size())))
                summary_[m_key]["trainable"] = module_.weight.requires_grad
            if hasattr(module_, "bias") and hasattr(module_.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module_.bias.size())))
            summary_[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary_ = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary_:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary_[layer]["input_shape"]),
                "{0:,}".format(summary_[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary_[layer]["output_shape"]),
                "{0:,}".format(summary_[layer]["nb_params"]),
            )

        total_params += summary_[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary_[layer]["input_shape"])
        else:
            total_output += np.prod(summary_[layer]["output_shape"])
        if "trainable" in summary_[layer]:
            if summary_[layer]["trainable"]:
                trainable_params += summary_[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")
