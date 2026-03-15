"""
🔧 修复后的可视化模块
适配新的数据格式：不再使用masked_waveform和masked_spectrogram
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import soundfile as sf
from contextlib import nullcontext
from typing import Dict, Optional, List
from tqdm import tqdm

from model import create_model
from dataset import AudioDataset, create_dataloaders
from utils import load_config, get_device, CheckpointManager


def _get_autocast_context(config: dict, device: torch.device):
    use_amp = bool(config.get('training', {}).get('use_amp', False)) and device.type == 'cuda'
    if not use_amp:
        return nullcontext()

    amp_dtype_name = str(config['training'].get('amp_dtype', 'float16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bfloat16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=amp_dtype)


def _is_cuda_memory_error(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    message = str(exc)
    memory_error_markers = (
        "out of memory",
        "CUBLAS_STATUS_NOT_INITIALIZED",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "CUDA out of memory",
    )
    return any(marker in message for marker in memory_error_markers)


def _expand_mask(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    expanded = mask.astype(bool, copy=False)
    while expanded.ndim < target.ndim:
        expanded = np.expand_dims(expanded, axis=0)
    return np.broadcast_to(expanded, target.shape)


def _compute_masked_mse_np(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray
) -> float:
    expanded_mask = _expand_mask(mask, original)
    diff = original[expanded_mask] - reconstructed[expanded_mask]
    if diff.size == 0:
        return float('nan')
    return float(np.mean(diff ** 2))


def _compute_masked_snr_np(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray
) -> float:
    expanded_mask = _expand_mask(mask, original)
    if not np.any(expanded_mask):
        return float('nan')

    original_region = original[expanded_mask]
    reconstructed_region = reconstructed[expanded_mask]
    noise = original_region - reconstructed_region
    signal_power = np.mean(original_region ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:
        return float('inf')

    return float(10.0 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12)))


def _warn_if_checkpoint_uses_stale_reconstruction_settings(checkpoint: Dict) -> None:
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

    if warning_reasons:
        print(
            "[Warning] Loaded checkpoint was trained with older reconstruction settings "
            f"({', '.join(warning_reasons)}). Masked-region recovery may still collapse; "
            "use the current config to retrain or continue finetuning."
        )


def plot_waveform_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    sample_rate: int,
    title: str = "Waveform Comparison",
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """绘制波形对比图"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    time = np.arange(len(original)) / sample_rate

    # 原始波形
    axes[0].plot(time, original, color='blue', linewidth=0.5)
    axes[0].set_title('Original Waveform', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim([0, time[-1]])
    axes[0].grid(True, alpha=0.3)

    # 掩码后波形
    axes[1].plot(time, masked, color='orange', linewidth=0.5)
    mask_regions = mask.astype(float)
    axes[1].fill_between(time, -1, 1, where=mask_regions > 0.5, alpha=0.3, color='red', label='Masked Region')
    axes[1].set_title('Masked Waveform', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim([0, time[-1]])
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 重建波形
    axes[2].plot(time, reconstructed, color='green', linewidth=0.5)
    axes[2].set_title('Reconstructed Waveform', fontsize=12)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlim([0, time[-1]])
    axes[2].grid(True, alpha=0.3)

    # 差异
    diff = original - reconstructed
    axes[3].plot(time, diff, color='red', linewidth=0.5)
    masked_mse = _compute_masked_mse_np(original, reconstructed, mask)
    axes[3].set_title(f'Difference (Masked MSE: {masked_mse:.6f})', fontsize=12)
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlim([0, time[-1]])
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved waveform comparison to {save_path}")

    plt.close()


def plot_spectrogram_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = "Spectrogram Comparison",
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """绘制频谱图对比"""
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 0.03])

    time_frames = original.shape[1]
    time_axis = np.arange(time_frames) * hop_length / sample_rate
    freq_axis = np.arange(original.shape[0])

    vmin = min(original.min(), reconstructed.min())
    vmax = max(original.max(), reconstructed.max())

    # 原始频谱图
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(original, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_axis[-1], 0, original.shape[0]], vmin=vmin, vmax=vmax)
    ax1.set_title('Original Spectrogram', fontsize=12)
    ax1.set_ylabel('Mel Bin')

    # 掩码后频谱图
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(masked, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_axis[-1], 0, masked.shape[0]], vmin=vmin, vmax=vmax)
    for i in range(len(mask)):
        if mask[i]:
            t = i * hop_length / sample_rate
            ax2.axvline(x=t, color='red', alpha=0.3, linewidth=0.5)
    ax2.set_title('Masked Spectrogram', fontsize=12)
    ax2.set_ylabel('Mel Bin')

    # 重建频谱图
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_axis[-1], 0, reconstructed.shape[0]], vmin=vmin, vmax=vmax)
    masked_mse = _compute_masked_mse_np(original, reconstructed, mask)
    ax3.set_title(f'Reconstructed Spectrogram (Masked MSE: {masked_mse:.6f})', fontsize=12)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Mel Bin')

    # 颜色条
    cax = fig.add_subplot(gs[:, 1])
    plt.colorbar(im3, cax=cax, label='Log Mel Magnitude')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved spectrogram comparison to {save_path}")

    plt.close()


def plot_combined_comparison(
    waveform_data: Dict[str, np.ndarray],
    spectrogram_data: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    sample_rate: int,
    hop_length: int,
    title: str = "Audio Reconstruction Comparison",
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """绘制综合对比图"""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 0.03])

    time_wave = np.arange(len(waveform_data['original'])) / sample_rate
    time_spec = np.arange(spectrogram_data['original'].shape[1]) * hop_length / sample_rate

    # 波形部分
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_wave, waveform_data['original'], color='blue', linewidth=0.5)
    ax1.set_title('Original Waveform', fontsize=11)
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_wave, waveform_data['masked'], color='orange', linewidth=0.5)
    mask_regions = masks['waveform'].astype(float)
    ax2.fill_between(time_wave, -1, 1, where=mask_regions > 0.5, alpha=0.3, color='red')
    ax2.set_title('Masked Waveform', fontsize=11)
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time_wave, waveform_data['reconstructed'], color='green', linewidth=0.5)
    ax3.set_title('Reconstructed Waveform', fontsize=11)
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[3, 0])
    wave_diff = waveform_data['original'] - waveform_data['reconstructed']
    ax4.plot(time_wave, wave_diff, color='red', linewidth=0.5)
    wave_snr = _compute_masked_snr_np(
        waveform_data['original'],
        waveform_data['reconstructed'],
        masks['waveform']
    )
    ax4.set_title(f'Waveform Difference (Masked SNR: {wave_snr:.2f} dB)', fontsize=11)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)

    # 频谱图部分
    vmin = min(spectrogram_data['original'].min(), spectrogram_data['reconstructed'].min())
    vmax = max(spectrogram_data['original'].max(), spectrogram_data['reconstructed'].max())

    ax5 = fig.add_subplot(gs[0, 1])
    im5 = ax5.imshow(spectrogram_data['original'], aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_spec[-1], 0, spectrogram_data['original'].shape[0]], vmin=vmin, vmax=vmax)
    ax5.set_title('Original Spectrogram', fontsize=11)
    ax5.set_ylabel('Mel Bin')

    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(spectrogram_data['masked'], aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_spec[-1], 0, spectrogram_data['masked'].shape[0]], vmin=vmin, vmax=vmax)
    ax6.set_title('Masked Spectrogram', fontsize=11)
    ax6.set_ylabel('Mel Bin')

    ax7 = fig.add_subplot(gs[2, 1])
    im7 = ax7.imshow(spectrogram_data['reconstructed'], aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, time_spec[-1], 0, spectrogram_data['reconstructed'].shape[0]], vmin=vmin, vmax=vmax)
    ax7.set_title('Reconstructed Spectrogram', fontsize=11)
    ax7.set_ylabel('Mel Bin')

    ax8 = fig.add_subplot(gs[3, 1])
    spec_diff = spectrogram_data['original'] - spectrogram_data['reconstructed']
    im8 = ax8.imshow(spec_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[0, time_spec[-1], 0, spec_diff.shape[0]])
    spec_snr = _compute_masked_snr_np(
        spectrogram_data['original'],
        spectrogram_data['reconstructed'],
        masks['spectrogram']
    )
    ax8.set_title(f'Spectrogram Difference (Masked SNR: {spec_snr:.2f} dB)', fontsize=11)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Mel Bin')

    cax = fig.add_subplot(gs[:, 2])
    plt.colorbar(im7, cax=cax, label='Log Mel Magnitude')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved combined comparison to {save_path}")

    plt.close()


def visualize_reconstruction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: dict,
    device: torch.device,
    output_dir: str,
    num_samples: int = 5
):
    """
    ✅ 变长 + OOM 保护的可视化函数

    关键改动：
    1. 逐样本单独推理（batch_size=1），彻底避免多条长音频叠加爆显存
    2. 推理前对超长音频截断，与 collate_fn 保持一致
    3. OOM 时跳过该样本并清理显存，继续下一条
    4. 每条推理后主动 empty_cache
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    sample_rate  = config['data']['sample_rate']
    hop_length   = config['data']['hop_length']
    dpi          = config['visualization']['dpi']
    save_audio   = config['visualization']['save_audio']
    # 与 collate_fn 保持一致的最大长度限制
    max_secs     = config['data'].get('max_audio_seconds', 10.0)
    max_wave_len = int(max_secs * sample_rate)
    max_spec_len = max_wave_len // hop_length + 1

    sample_count = 0
    skipped      = 0
    cpu_fallback = False

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating visualizations"):
            if sample_count >= num_samples:
                break

            wave_lengths = batch['waveform_lengths']     # [B]，CPU
            spec_lengths = batch['spectrogram_lengths']  # [B]，CPU
            batch_size   = batch['waveform'].size(0)

            # ✅ 逐样本推理，避免多条长音频同时占显存
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break

                try:
                    wl = wave_lengths[i].item()
                    sl = spec_lengths[i].item()

                    # ✅ 超长截断（从头截，可视化不需要随机性）
                    wl_clip = min(wl, max_wave_len)
                    sl_clip = min(sl, max_spec_len)

                    wave_cpu = batch['waveform'][i:i+1, :wl_clip]
                    spec_cpu = batch['spectrogram'][i:i+1, :, :, :sl_clip]
                    wmask_cpu = batch['waveform_mask'][i:i+1, :wl_clip]
                    smask_cpu = batch['spectrogram_mask'][i:i+1, :sl_clip]

                    while True:
                        wave_i = wave_cpu.to(device, non_blocking=True)          # [1, T]
                        spec_i = spec_cpu.to(device, non_blocking=True)          # [1,1,n_mels,T']
                        wmask_i = wmask_cpu.to(device, non_blocking=True)
                        smask_i = smask_cpu.to(device, non_blocking=True)

                        try:
                            with _get_autocast_context(config, device):
                                outputs = model(
                                    waveform=wave_i,
                                    spectrogram=spec_i,
                                    waveform_mask=wmask_i,
                                    spectrogram_mask=smask_i
                                )
                            break
                        except RuntimeError as e:
                            if device.type == 'cuda' and _is_cuda_memory_error(e):
                                print(
                                    "[Visualize] GPU memory issue detected during inference. "
                                    "Switching visualization to CPU fallback."
                                )
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                model = model.to(torch.device('cpu'))
                                device = torch.device('cpu')
                                cpu_fallback = True
                                continue
                            raise

                    # ---------- 转 numpy，脱离 GPU ----------
                    orig_wave    = wave_i[0].float().cpu().numpy().astype(np.float32, copy=False)  # [T]
                    recon_wave   = outputs['reconstructed_waveform'][0].float().cpu().numpy().astype(np.float32, copy=False)  # [T]
                    wave_mask_np = wmask_i[0].cpu().numpy()                              # [T]

                    orig_spec    = spec_i[0, 0].float().cpu().numpy().astype(np.float32, copy=False)  # [n_mels, T']
                    recon_spec   = outputs['reconstructed_spectrogram'][0, 0].float().cpu().numpy().astype(np.float32, copy=False)
                    spec_mask_np = smask_i[0].cpu().numpy()

                    # masked 版本仅用于可视化
                    mask_wave = orig_wave.copy();  mask_wave[wave_mask_np] = 0
                    mask_spec = orig_spec.copy();  mask_spec[:, spec_mask_np] = 0

                    duration_s = wl_clip / sample_rate
                    tag = f'sample_{sample_count + 1}'

                    # ---------- 绘图 ----------
                    waveform_data    = {'original': orig_wave,  'masked': mask_wave,  'reconstructed': recon_wave}
                    spectrogram_data = {'original': orig_spec,  'masked': mask_spec,  'reconstructed': recon_spec}
                    masks_dict       = {'waveform': wave_mask_np, 'spectrogram': spec_mask_np}

                    plot_combined_comparison(
                        waveform_data, spectrogram_data, masks_dict,
                        sample_rate, hop_length,
                        title=f'Sample {sample_count+1} - Audio Reconstruction (duration: {duration_s:.2f}s)',
                        save_path=os.path.join(output_dir, f'{tag}_combined.png'), dpi=dpi
                    )
                    plot_waveform_comparison(
                        orig_wave, mask_wave, recon_wave, wave_mask_np, sample_rate,
                        title=f'Sample {sample_count+1} - Waveform Comparison (duration: {duration_s:.2f}s)',
                        save_path=os.path.join(output_dir, f'{tag}_waveform.png'), dpi=dpi
                    )
                    plot_spectrogram_comparison(
                        orig_spec, mask_spec, recon_spec, spec_mask_np, sample_rate, hop_length,
                        title=f'Sample {sample_count+1} - Spectrogram Comparison (duration: {duration_s:.2f}s)',
                        save_path=os.path.join(output_dir, f'{tag}_spectrogram.png'), dpi=dpi
                    )

                    # ---------- 保存音频 ----------
                    if save_audio:
                        sf.write(os.path.join(output_dir, f'{tag}_original.wav'),      orig_wave,  sample_rate)
                        sf.write(os.path.join(output_dir, f'{tag}_reconstructed.wav'), recon_wave, sample_rate)
                        print(f"Saved audio for sample {sample_count+1} (duration: {duration_s:.2f}s)")

                    sample_count += 1

                except torch.cuda.OutOfMemoryError:
                    # ✅ OOM 时跳过该样本，继续下一条
                    skipped += 1
                    print(f"[Visualize] OOM on sample {sample_count+1}, skipping.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                finally:
                    # ✅ 每条推理后都清理显存
                    if torch.cuda.is_available() and device.type == 'cuda':
                        torch.cuda.empty_cache()

    if skipped > 0:
        print(f"[Visualize] Warning: skipped {skipped} sample(s) due to OOM.")
    if cpu_fallback:
        print("[Visualize] Note: visualization fell back to CPU to avoid GPU memory failures.")
    print(f"\nVisualization complete! {sample_count} samples saved to {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize Audio Reconstruction')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (default: best model)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置输出目录
    output_dir = args.output_dir or config['visualization']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 获取设备
    device = get_device(config['device'])
    print(f"Using device: {device}")

    # 创建模型
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # 加载检查点
    checkpoint_manager = CheckpointManager(config['training']['checkpoint_dir'])
    if args.checkpoint:
        checkpoint = checkpoint_manager.load(model, checkpoint_path=args.checkpoint)
    else:
        checkpoint = checkpoint_manager.load(model, load_best=True)
    _warn_if_checkpoint_uses_stale_reconstruction_settings(checkpoint)

    # 创建数据加载器
    print("Creating data loader...")
    _, val_loader = create_dataloaders(config)

    # 可视化
    visualize_reconstruction(
        model, val_loader, config, device,
        output_dir, args.num_samples
    )


if __name__ == "__main__":
    main()
