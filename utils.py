"""
工具函数模块
Utility Functions Module

包含:
1. 配置加载和保存
2. 检查点管理
3. 学习率调度
4. 日志记录
5. 其他辅助函数
"""

import os
import random
import yaml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """获取设备"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_directories(config: dict):
    """创建必要的目录"""
    dirs = [
        config['training']['checkpoint_dir'],
        config['visualization']['output_dir'],
        config['data']['data_dir']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


class CheckpointManager:
    """检查点管理器"""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        config: dict,
        is_best: bool = False
    ):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': config
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}_step_{step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)

        # 清理旧检查点
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """加载检查点"""
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pt')

        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return {'epoch': 0, 'step': 0}

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 加载模型状态字典，手动处理位置编码参数
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # 过滤掉位置编码参数和当前结构下 shape 不兼容的参数
        filtered_state_dict = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if 'pos_encoding.pe' not in key:
                if key not in model_state_dict:
                    skipped_keys.append((key, "missing_in_model"))
                    continue
                if model_state_dict[key].shape != value.shape:
                    skipped_keys.append((key, f"shape_mismatch {tuple(value.shape)} -> {tuple(model_state_dict[key].shape)}"))
                    continue
                filtered_state_dict[key] = value
        
        # 加载过滤后的状态字典
        try:
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print("Loaded model state with positional encodings reinitialized when needed.")
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} incompatible checkpoint key(s).")
                for key, reason in skipped_keys[:10]:
                    print(f"  - {key}: {reason}")
                if len(skipped_keys) > 10:
                    print(f"  ... and {len(skipped_keys) - 10} more")
            if missing_keys:
                print(f"Missing keys after load: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys after load: {len(unexpected_keys)}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，保留最新的几个"""
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith('checkpoint_epoch_') and f.endswith('.pt'):
                path = os.path.join(self.checkpoint_dir, f)
                checkpoints.append((path, os.path.getmtime(path)))

        # 按时间排序
        checkpoints.sort(key=lambda x: x[1], reverse=True)

        # 删除多余的检查点
        for path, _ in checkpoints[self.max_checkpoints:]:
            os.remove(path)


class WarmupCosineScheduler:
    """带预热的余弦学习率调度器"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self._get_lr()

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # 线性预热
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + np.cos(np.pi * progress))

    def state_dict(self) -> dict:
        return {'current_step': self.current_step}

    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict['current_step']


class AverageMeter:
    """平均值计算器"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.6f}"


class Logger:
    """日志记录器"""

    def __init__(self, log_dir: str, name: str = "training"):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def gradient_penalty(model: nn.Module, max_norm: float = 1.0) -> float:
    """梯度裁剪"""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算信噪比 (SNR)"""
    noise = original - reconstructed
    signal_power = (original ** 2).mean()
    noise_power = (noise ** 2).mean()

    if noise_power == 0:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算均方误差"""
    return ((original - reconstructed) ** 2).mean().item()


def compute_mae(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """计算平均绝对误差"""
    return (original - reconstructed).abs().mean().item()


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")

    # 测试配置加载
    config = load_config("config.yaml")
    print(f"Config loaded: {list(config.keys())}")

    # 测试随机种子
    set_seed(42)
    print(f"Random seed set to 42")

    # 测试设备获取
    device = get_device("cuda")
    print(f"Device: {device}")

    # 测试目录创建
    create_directories(config)
    print("Directories created")

    # 测试AverageMeter
    meter = AverageMeter("loss")
    for i in range(10):
        meter.update(i * 0.1)
    print(f"AverageMeter: {meter}")

    print("\nAll tests passed!")
