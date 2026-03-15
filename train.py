"""
🔧 修复后的训练脚本

关键修改：数据集不再返回masked_waveform和masked_spectrogram
模型接收原始数据和mask标记
"""

import os
import argparse
import time
import math
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from typing import Dict, Tuple

from model import create_model  # ✅ 使用修复后的模型
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from dataset import create_dataloaders
from utils import (
    load_config,
    save_config,
    set_seed,
    get_device,
    create_directories,
    CheckpointManager,
    WarmupCosineScheduler,
    AverageMeter,
    Logger,
    count_parameters,
    format_time,
    gradient_penalty,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dual-Branch Audio Pretrain Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    return parser.parse_args()


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _slice_batch(
    batch: Dict[str, torch.Tensor],
    start: int,
    end: int
) -> Dict[str, torch.Tensor]:
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _get_micro_batch_size(config: dict, batch_size: int) -> int:
    micro_batch_size = int(config['training'].get('micro_batch_size', batch_size))
    if micro_batch_size <= 0:
        raise ValueError("training.micro_batch_size must be a positive integer")
    return min(micro_batch_size, batch_size)


def _get_autocast_context(config: dict, device: torch.device):
    training_cfg = config['training']
    use_amp = bool(training_cfg.get('use_amp', False)) and device.type == 'cuda'
    if not use_amp:
        return nullcontext()

    amp_dtype_name = str(training_cfg.get('amp_dtype', 'float16')).lower()
    if amp_dtype_name == 'bfloat16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    return torch.autocast(device_type='cuda', dtype=amp_dtype)


def _create_grad_scaler(config: dict, device: torch.device):
    use_amp = bool(config['training'].get('use_amp', False)) and device.type == 'cuda'
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _compute_masked_snr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    region_mask: torch.Tensor
) -> float:
    if original.dim() != region_mask.dim():
        while region_mask.dim() < original.dim():
            region_mask = region_mask.unsqueeze(1)
        region_mask = region_mask.expand_as(original)

    if not region_mask.any():
        return float('nan')

    original_region = original[region_mask]
    reconstructed_region = reconstructed[region_mask]
    noise = original_region - reconstructed_region
    signal_power = (original_region ** 2).mean()
    noise_power = (noise ** 2).mean()

    if noise_power <= 0:
        return float('inf')

    snr = 10 * torch.log10(signal_power.clamp_min(1e-12) / noise_power.clamp_min(1e-12))
    return snr.item()


def _get_checkpoint_reconstruction_warning(checkpoint: Dict[str, torch.Tensor]) -> str:
    checkpoint_config = checkpoint.get('config', {})
    training_cfg = checkpoint_config.get('training', {})
    mask_cfg = checkpoint_config.get('mask', {})

    warning_reasons = []
    if bool(training_cfg.get('copy_unmasked_input', True)):
        warning_reasons.append("copy_unmasked_input=true")
    if float(training_cfg.get('waveform_stft_loss_weight', 0.0)) <= 0.0:
        warning_reasons.append("waveform_stft_loss_weight<=0")
    if float(training_cfg.get('waveform_mel_loss_weight', 0.0)) <= 0.0:
        warning_reasons.append("waveform_mel_loss_weight<=0")
    if float(training_cfg.get('waveform_unmask_loss_weight', training_cfg.get('unmask_loss_weight', 0.0))) <= 0.0:
        warning_reasons.append("waveform_unmask_loss_weight<=0")
    if float(training_cfg.get('spectrogram_unmask_loss_weight', training_cfg.get('unmask_loss_weight', 0.0))) <= 0.0:
        warning_reasons.append("spectrogram_unmask_loss_weight<=0")
    if mask_cfg.get('mask_type') == 'circular' and float(mask_cfg.get('mask_ratio', 0.0)) >= 0.2:
        warning_reasons.append(
            f"mask={mask_cfg.get('mask_type')}:{mask_cfg.get('mask_ratio')}"
        )

    if not warning_reasons:
        return ""

    return (
        "Loaded checkpoint was trained with older reconstruction settings "
        f"({', '.join(warning_reasons)}). Masked-region recovery may stay weak until "
        "you finetune it further with the current config."
    )


def _run_validation_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    config: dict
) -> Dict[str, float]:
    batch = _move_batch_to_device(batch, device)

    waveform = batch['waveform']
    spectrogram = batch['spectrogram']
    waveform_mask = batch['waveform_mask']
    spectrogram_mask = batch['spectrogram_mask']
    padding_mask_wave = batch['padding_mask_wave']
    padding_mask_spec = batch['padding_mask_spec']

    with _get_autocast_context(config, device):
        outputs = model(
            waveform=waveform,
            spectrogram=spectrogram,
            waveform_mask=waveform_mask,
            spectrogram_mask=spectrogram_mask
        )

        targets = {
            'waveform': waveform,
            'spectrogram': spectrogram
        }
        masks = {
            'waveform_mask': waveform_mask,
            'spectrogram_mask': spectrogram_mask,
            'padding_mask_wave': padding_mask_wave,
            'padding_mask_spec': padding_mask_spec,
        }

        losses = model.compute_loss(outputs, targets, masks)

    valid_wave_mask = waveform_mask & (~padding_mask_wave)
    valid_spec_mask = spectrogram_mask & (~padding_mask_spec)

    return {
        'batch_size': waveform.size(0),
        'total_loss': losses['total_loss'].item(),
        'waveform_loss': losses['waveform_loss'].item(),
        'spectrogram_loss': losses['spectrogram_loss'].item(),
        'waveform_mel_loss': losses['waveform_mel_loss'].item(),
        'waveform_masked_snr': _compute_masked_snr(
            waveform,
            outputs['reconstructed_waveform'],
            valid_wave_mask
        ),
        'spectrogram_masked_snr': _compute_masked_snr(
            spectrogram,
            outputs['reconstructed_spectrogram'],
            valid_spec_mask
        ),
    }


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    config: dict,
    epoch: int,
    logger: Logger,
    scaler
) -> Dict[str, float]:
    model.train()

    total_loss_meter = AverageMeter("total_loss")
    waveform_loss_meter = AverageMeter("waveform_loss")
    spectrogram_loss_meter = AverageMeter("spectrogram_loss")

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

    for step, batch in enumerate(pbar):
        full_batch_size = batch['waveform'].size(0)
        micro_batch_size = _get_micro_batch_size(config, full_batch_size)
        optimizer.zero_grad(set_to_none=True)

        last_losses = None
        grad_norm = float('nan')

        try:
            for start in range(0, full_batch_size, micro_batch_size):
                end = min(start + micro_batch_size, full_batch_size)
                micro_batch = _slice_batch(batch, start, end)
                micro_batch = _move_batch_to_device(micro_batch, device)

                waveform = micro_batch['waveform']
                spectrogram = micro_batch['spectrogram']
                waveform_mask = micro_batch['waveform_mask']
                spectrogram_mask = micro_batch['spectrogram_mask']
                padding_mask_wave = micro_batch['padding_mask_wave']
                padding_mask_spec = micro_batch['padding_mask_spec']

                with _get_autocast_context(config, device):
                    outputs = model(
                        waveform=waveform,
                        spectrogram=spectrogram,
                        waveform_mask=waveform_mask,
                        spectrogram_mask=spectrogram_mask
                    )

                    targets = {
                        'waveform': waveform,
                        'spectrogram': spectrogram
                    }
                    masks = {
                        'waveform_mask': waveform_mask,
                        'spectrogram_mask': spectrogram_mask,
                        'padding_mask_wave': padding_mask_wave,
                        'padding_mask_spec': padding_mask_spec,
                    }

                    losses = model.compute_loss(outputs, targets, masks)

                micro_bs = waveform.size(0)
                scale = micro_bs / full_batch_size
                scaler.scale(losses['total_loss'] * scale).backward()

                total_loss_meter.update(losses['total_loss'].item(), micro_bs)
                waveform_loss_meter.update(losses['waveform_loss'].item(), micro_bs)
                spectrogram_loss_meter.update(losses['spectrogram_loss'].item(), micro_bs)
                last_losses = losses

            scaler.unscale_(optimizer)
            grad_norm = gradient_penalty(model, config['training']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        except torch.cuda.OutOfMemoryError as e:
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during training. Current config is still too large for this GPU. "
                "Try lowering training.micro_batch_size to 1, training.batch_size to 1-2, "
                "or data.max_audio_seconds to 4-6 seconds."
            ) from e

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{total_loss_meter.avg:.4f}",
            'wave': f"{waveform_loss_meter.avg:.4f}",
            'spec': f"{spectrogram_loss_meter.avg:.4f}",
            'lr': f"{current_lr:.2e}"
        })

        # 日志记录
        global_step = (epoch - 1) * len(train_loader) + step + 1
        if global_step % config['training']['log_interval'] == 0:
            logger.info(
                f"Epoch {epoch}, Step {global_step}: "
                f"Total Loss = {total_loss_meter.avg:.6f}, "
                f"Waveform Loss = {waveform_loss_meter.avg:.6f}, "
                f"Spectrogram Loss = {spectrogram_loss_meter.avg:.6f}, "
                f"Wave Masked MSE = {last_losses['waveform_masked_loss']:.6f}, "
                f"Wave Coarse MSE = {last_losses['waveform_coarse_loss']:.6f}, "
                f"Spec Masked MSE = {last_losses['spectrogram_masked_loss']:.6f}, "
                f"Wave STFT = {last_losses['waveform_stft_loss']:.6f}, "
                f"Wave Diff = {last_losses['waveform_diff_loss']:.6f}, "
                f"Wave Mel = {last_losses['waveform_mel_loss']:.6f}, "
                f"LR = {current_lr:.2e}, "
                f"Grad Norm = {grad_norm:.4f}"
            )

    return {
        'total_loss': total_loss_meter.avg,
        'waveform_loss': waveform_loss_meter.avg,
        'spectrogram_loss': spectrogram_loss_meter.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict
) -> Dict[str, float]:
    model.eval()

    total_loss_meter = AverageMeter("total_loss")
    waveform_loss_meter = AverageMeter("waveform_loss")
    spectrogram_loss_meter = AverageMeter("spectrogram_loss")
    waveform_mel_loss_meter = AverageMeter("waveform_mel_loss")
    waveform_masked_snr_meter = AverageMeter("waveform_masked_snr")
    spectrogram_masked_snr_meter = AverageMeter("spectrogram_masked_snr")

    recovered_batches = 0
    skipped_samples = 0

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        try:
            metrics = _run_validation_batch(model, batch, device, config)
            batch_size = metrics['batch_size']
            total_loss_meter.update(metrics['total_loss'], batch_size)
            waveform_loss_meter.update(metrics['waveform_loss'], batch_size)
            spectrogram_loss_meter.update(metrics['spectrogram_loss'], batch_size)
            waveform_mel_loss_meter.update(metrics['waveform_mel_loss'], batch_size)
            waveform_masked_snr_meter.update(metrics['waveform_masked_snr'], batch_size)
            spectrogram_masked_snr_meter.update(metrics['spectrogram_masked_snr'], batch_size)
        except torch.cuda.OutOfMemoryError:
            recovered_batches += 1
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            batch_size = batch['waveform'].size(0)
            for idx in range(batch_size):
                single_batch = _slice_batch(batch, idx, idx + 1)
                try:
                    metrics = _run_validation_batch(model, single_batch, device, config)
                    total_loss_meter.update(metrics['total_loss'], 1)
                    waveform_loss_meter.update(metrics['waveform_loss'], 1)
                    spectrogram_loss_meter.update(metrics['spectrogram_loss'], 1)
                    waveform_mel_loss_meter.update(metrics['waveform_mel_loss'], 1)
                    waveform_masked_snr_meter.update(metrics['waveform_masked_snr'], 1)
                    spectrogram_masked_snr_meter.update(metrics['spectrogram_masked_snr'], 1)
                except torch.cuda.OutOfMemoryError:
                    skipped_samples += 1
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    if recovered_batches > 0 or skipped_samples > 0:
        print(
            "[Validation] Warning: "
            f"{recovered_batches} batch(es) hit OOM and fell back to per-sample validation; "
            f"{skipped_samples} sample(s) were still skipped."
        )

    if total_loss_meter.count == 0:
        print("[Validation] Error: no validation samples were evaluated.")
        return {
            'total_loss': float('inf'),
            'waveform_loss': float('inf'),
            'spectrogram_loss': float('inf'),
            'waveform_mel_loss': float('inf'),
            'waveform_masked_snr': float('nan'),
            'spectrogram_masked_snr': float('nan')
        }

    return {
        'total_loss': total_loss_meter.avg,
        'waveform_loss': waveform_loss_meter.avg,
        'spectrogram_loss': spectrogram_loss_meter.avg,
        'waveform_mel_loss': waveform_mel_loss_meter.avg,
        'waveform_masked_snr': waveform_masked_snr_meter.avg,
        'spectrogram_masked_snr': spectrogram_masked_snr_meter.avg
    }


def train(config: dict, resume_path: str = None):
    # 设置随机种子
    set_seed(config['seed'])

    # 获取设备
    device = get_device(config['device'])
    print(f"Using device: {device}")

    # 创建目录
    create_directories(config)

    # 初始化日志
    logger = Logger(config['training']['checkpoint_dir'], "training")
    logger.info(f"Starting training with config: {config}")

    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 创建模型
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # 统计参数
    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scaler = _create_grad_scaler(config, device)

    # 创建学习率调度器
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config['training']['warmup_steps'],
        total_steps=total_steps
    )

    # 检查点管理器
    checkpoint_manager = CheckpointManager(config['training']['checkpoint_dir'])

    # 恢复训练
    start_epoch = 1
    best_val_loss = float('inf')

    if resume_path:
        checkpoint = checkpoint_manager.load(model, optimizer, scheduler, resume_path)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        checkpoint_warning = _get_checkpoint_reconstruction_warning(checkpoint)
        if checkpoint_warning:
            logger.warning(checkpoint_warning)
            logger.warning(
                "Resumed checkpoint was trained under an older reconstruction objective; "
                "resetting best_val_loss to inf so new checkpoints are selected using the current loss."
            )
            best_val_loss = float('inf')
        if not math.isfinite(best_val_loss) or best_val_loss <= 0.0:
            logger.warning(
                "Loaded checkpoint carries a non-finite or zero validation loss. "
                "This usually means it was produced by the old broken validation path; "
                "resetting best_val_loss to inf."
            )
            best_val_loss = float('inf')
        if start_epoch > config['training']['num_epochs']:
            raise ValueError(
                f"Resume checkpoint is already at epoch {checkpoint.get('epoch', 0)}, "
                f"but config.training.num_epochs={config['training']['num_epochs']}. "
                "Increase num_epochs to continue finetuning, or start without --resume."
            )
        logger.info(f"Resumed training from epoch {start_epoch}")

    # 保存配置
    save_config(config, os.path.join(config['training']['checkpoint_dir'], 'config.yaml'))

    # 训练循环
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        epoch_start_time = time.time()

        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, epoch, logger, scaler
        )

        # 验证
        val_metrics = validate(model, val_loader, device, config)

        epoch_time = time.time() - epoch_start_time

        # 日志
        logger.info(
            f"Epoch {epoch} completed in {format_time(epoch_time)} - "
            f"Train Loss: {train_metrics['total_loss']:.6f}, "
            f"Val Loss: {val_metrics['total_loss']:.6f}, "
            f"Val Wave Mel: {val_metrics['waveform_mel_loss']:.4f}, "
            f"Val Waveform Masked SNR: {val_metrics['waveform_masked_snr']:.2f} dB, "
            f"Val Spectrogram Masked SNR: {val_metrics['spectrogram_masked_snr']:.2f} dB"
        )

        # 保存检查点
        is_best = math.isfinite(val_metrics['total_loss']) and val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
            logger.info(f"New best model! Val Loss: {best_val_loss:.6f}")
        elif not math.isfinite(val_metrics['total_loss']):
            logger.warning("Validation did not produce a finite loss; skipping best-model update for this epoch.")

        if epoch % config['training']['save_interval'] == 0 or is_best:
            checkpoint_manager.save(
                model, optimizer, scheduler,
                epoch, epoch * len(train_loader),
                val_metrics['total_loss'],
                config,
                is_best
            )

    # 训练完成
    total_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(total_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return model


def main():
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['checkpoint_dir'] = os.path.join(args.output_dir, 'checkpoints')
        config['visualization']['output_dir'] = os.path.join(args.output_dir, 'visualizations')

    # 训练
    train(config, args.resume)


if __name__ == "__main__":
    main()
