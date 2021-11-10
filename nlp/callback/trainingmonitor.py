# encoding:utf-8
from typing import Dict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from nlp.tools.common import load_json
from nlp.tools.common import save_json

plt.switch_backend('agg')


class TrainingMonitor(object):
    def __init__(self, file_dir, arch, add_test=False):
        """
        重新开始训练的epoch点
        """
        if isinstance(file_dir, Path):
            pass
        else:
            file_dir = Path(file_dir)
        file_dir.mkdir(parents=True, exist_ok=True)

        self.arch = arch
        self.file_dir = file_dir
        self.H = {}
        self.add_test = add_test
        self.json_path = file_dir / (arch + "_training_monitor.json")
        self.paths = {}

    def reset(self, start_at):
        if start_at > 0:
            if self.json_path is not None:
                if self.json_path.exists():
                    self.H = load_json(self.json_path)
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:start_at]

    def epoch_step(self, logs: Dict):
        for (k, v) in logs.items():
            alist = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v, np.float):
                v = round(float(v), 4)
            alist.append(v)
            self.H[k] = alist

        # 写入文件
        if self.json_path is not None:
            save_json(data=self.H, file_path=self.json_path)

        # 保存train图像
        if len(self.H["loss"]) == 1:
            self.paths = {key: self.file_dir / (self.arch + f'_{key.upper()}') for key in self.H.keys()}

        if len(self.H["loss"]) > 1:
            # 指标变化
            # 曲线
            # 需要成对出现
            keys = [key for key, _ in self.H.items() if '_' not in key]
            for key in keys:
                array = np.arange(0, len(self.H[key]))
                plt.style.use("ggplot")
                plt.figure()
                plt.plot(array, self.H[key], label=f"train_{key}")
                plt.plot(array, self.H[f"valid_{key}"], label=f"valid_{key}")
                if self.add_test:
                    plt.plot(array, self.H[f"test_{key}"], label=f"test_{key}")
                plt.legend()
                plt.xlabel("Epoch #")
                plt.ylabel(key)
                plt.title(f"Training {key} [Epoch {len(self.H[key])}]")
                plt.savefig(str(self.paths[key]))
                plt.close()
