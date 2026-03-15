"""
🔧 音频数据集模块（支持变长音频 + 灵活数据源配置）

数据加载方式：
  支持三种模式，通过 config['data']['source_mode'] 控制：

  1. "simple"   —— 直接递归扫描 data_dir 下所有音频（原始行为）
  2. "dcase"    —— 按 DCASE2025 目录结构加载，自动发现 machine_type/train/ 子目录
                   可通过 machine_types 白名单过滤机器类型
  3. "explicit" —— 直接在 data_paths 中手动列出若干目录，逐一递归扫描

config.yaml 中对应的配置示例（三选一）：

  # ---- 模式 1: simple ----
  data:
    source_mode: simple
    data_dir: "./data"

  # ---- 模式 2: dcase ----
  data:
    source_mode: dcase
    dcase_root: "./data/dcase2025t2/eval_data/raw"  # 包含各 machine_type 的根目录
    machine_types:           # 留空列表 [] 表示加载全部机器类型
      - AutoTrash
      - fan
      - pump
    split: train             # train / test / val，对应子目录名

  # ---- 模式 3: explicit ----
  data:
    source_mode: explicit
    data_paths:
      - "./data/dcase2025t2/eval_data/raw/AutoTrash/train"
      - "./data/extra/pump/train"
"""

import os
import random
import functools
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from typing import Tuple, Dict, Optional, List

warnings.filterwarnings('ignore')

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Run: pip install soundfile")


# ──────────────────────────────────────────────
#  文件收集工具
# ──────────────────────────────────────────────

SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def _collect_files_from_dir(directory: str) -> List[str]:
    """递归扫描目录，收集所有支持格式的音频文件路径。"""
    files = []
    if not os.path.isdir(directory):
        print(f"[Warning] Directory not found, skipping: {directory}")
        return files
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.lower().endswith(SUPPORTED_FORMATS):
                files.append(os.path.join(root, fname))
    return files


def collect_audio_files(config: dict) -> List[str]:
    """
    根据 config['data']['source_mode'] 决定数据收集策略，返回音频文件路径列表。

    三种模式：
      simple   → 递归扫描 data['data_dir']
      dcase    → 按 DCASE 目录结构，自动发现 <dcase_root>/<machine_type>/<split>/ 子目录
      explicit → 直接扫描 data['data_paths'] 中列出的每个目录
    """
    data_cfg = config['data']
    mode = data_cfg.get('source_mode', 'simple')

    # ── 模式 1: simple ────────────────────────
    if mode == 'simple':
        data_dir = data_cfg.get('data_dir', './data')
        files = _collect_files_from_dir(data_dir)
        print(f"[DataLoader | simple] Scanned '{data_dir}' → {len(files)} files")
        return files

    # ── 模式 2: dcase ─────────────────────────
    elif mode == 'dcase':
        dcase_root   = data_cfg.get('dcase_root', './data/dcase2025t2/eval_data/raw')
        split        = data_cfg.get('split', 'train')           # 子目录名，如 train/test
        wanted_types = data_cfg.get('machine_types', [])        # 空列表 = 全部

        # 自动发现 dcase_root 下的机器类型目录
        if not os.path.isdir(dcase_root):
            print(f"[Warning] dcase_root not found: {dcase_root}")
            return []

        all_machine_types = sorted([
            d for d in os.listdir(dcase_root)
            if os.path.isdir(os.path.join(dcase_root, d))
        ])

        # 过滤（若 wanted_types 为空则使用全部）
        if wanted_types:
            selected = [t for t in wanted_types if t in all_machine_types]
            missing  = [t for t in wanted_types if t not in all_machine_types]
            if missing:
                print(f"[Warning] machine_types not found in dcase_root: {missing}")
        else:
            selected = all_machine_types

        print(f"[DataLoader | dcase] root='{dcase_root}', split='{split}'")
        print(f"  Available machine types : {all_machine_types}")
        print(f"  Selected machine types  : {selected}")

        all_files = []
        for mtype in selected:
            target_dir = os.path.join(dcase_root, mtype, split)
            files = _collect_files_from_dir(target_dir)
            print(f"  [{mtype}/{split}] → {len(files)} files")
            all_files.extend(files)

        print(f"[DataLoader | dcase] Total: {len(all_files)} files")
        return all_files

    # ── 模式 3: explicit ──────────────────────
    elif mode == 'explicit':
        data_paths = data_cfg.get('data_paths', [])
        if not data_paths:
            print("[Warning] source_mode='explicit' but data_paths is empty.")
            return []

        all_files = []
        for path in data_paths:
            files = _collect_files_from_dir(path)
            print(f"[DataLoader | explicit] '{path}' → {len(files)} files")
            all_files.extend(files)

        print(f"[DataLoader | explicit] Total: {len(all_files)} files")
        return all_files

    else:
        raise ValueError(
            f"Unknown source_mode='{mode}'. Choose one of: simple, dcase, explicit"
        )


# ──────────────────────────────────────────────
#  掩码生成器
# ──────────────────────────────────────────────

class MaskGenerator:
    """
    掩码生成器

    支持的 mask_type：
    ─────────────────────────────────────────────────────────
    circular  ★推荐★
        把频谱图时间轴首尾相连成一个"环"，随机选取起点，
        连续掩码 round(length × mask_ratio) 帧。
        若掩码区域超出末尾则从头部环绕补足，保证掩码长度
        精确等于 round(length × mask_ratio)，且始终连续。

        可控参数（config.yaml → mask）：
          mask_ratio : 掩码占比，如 0.3 表示掩掉 30%
          （min_masks / max_masks / mask_length 对该模式不生效）

    random     : 随机散布多个固定长度 mask block
    block      : 单个连续块，大小 = length × mask_ratio
    single     : 单个连续块，大小 = mask_length 帧
    structured : 均匀间隔分布多个固定长度 mask block
    ─────────────────────────────────────────────────────────
    """

    def __init__(
        self,
        mask_ratio: float = 0.3,
        mask_length: int = 10,
        min_masks: int = 1,
        max_masks: int = 10,
        mask_type: str = "circular"
    ):
        self.mask_ratio  = mask_ratio
        self.mask_length = mask_length
        self.min_masks   = min_masks
        self.max_masks   = max_masks
        self.mask_type   = mask_type

    # ── 公共入口 ──────────────────────────────

    def generate_mask(
        self,
        waveform_length: int,
        spectrogram_time_frames: int,
        hop_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (waveform_mask [T], spectrogram_mask [T'])
        True 表示该位置被掩码。

        掩码始终在频谱图时间轴上生成，再映射回波形轴，
        保证两者语义严格对齐。
        """
        if self.mask_type == "circular":
            spectrogram_mask = self._circular_mask(spectrogram_time_frames)
        elif self.mask_type == "random":
            spectrogram_mask = self._random_mask(spectrogram_time_frames)
        elif self.mask_type == "block":
            spectrogram_mask = self._block_mask(spectrogram_time_frames)
        elif self.mask_type in ("single", "random_block"):
            spectrogram_mask = self._single_block_mask(spectrogram_time_frames)
        elif self.mask_type == "structured":
            spectrogram_mask = self._structured_mask(spectrogram_time_frames)
        else:
            raise ValueError(
                f"Unknown mask_type='{self.mask_type}'. "
                "Choose one of: circular, random, block, single, structured"
            )

        waveform_mask = self._spectrogram_mask_to_waveform_mask(
            spectrogram_mask, waveform_length, hop_length
        )
        return waveform_mask, spectrogram_mask

    # ── circular（核心新增）──────────────────

    def _circular_mask(self, length: int) -> torch.Tensor:
        """
        环形连续掩码：
          1. mask_len = round(length × mask_ratio)，至少 1 帧
          2. 在 [0, length) 内随机选取起点 start
          3. 掩码帧 = [(start + i) % length  for i in range(mask_len)]
             → 超出末尾自动从 0 环绕

        等效于把频谱图时间轴首尾相接成圆环，
        随机截取 mask_ratio 比例的一段连续区域进行掩码。
        """
        mask_len = max(1, round(length * self.mask_ratio))
        mask_len = min(mask_len, length)        # 不能超过总帧数

        start = random.randint(0, length - 1)   # 任意起点（全范围）

        mask = torch.zeros(length, dtype=torch.bool)
        indices = torch.arange(mask_len)
        masked_indices = (start + indices) % length
        mask[masked_indices] = True

        return mask

    # ── 其他已有掩码类型 ──────────────────────

    def _random_mask(self, length: int) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        num_mask = max(self.min_masks, int(length * self.mask_ratio / max(self.mask_length, 1)))
        num_mask = min(num_mask, self.max_masks)
        possible_starts = list(range(0, length - self.mask_length + 1))
        if len(possible_starts) < num_mask:
            num_mask = len(possible_starts)
        if num_mask > 0:
            for start in random.sample(possible_starts, num_mask):
                mask[start:min(start + self.mask_length, length)] = True
        return mask

    def _block_mask(self, length: int) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        mask_size = int(length * self.mask_ratio)
        if mask_size > 0 and length > mask_size:
            start = random.randint(0, length - mask_size)
            mask[start:start + mask_size] = True
        return mask

    def _single_block_mask(self, length: int) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        block_len = max(1, min(int(self.mask_length), length))
        start = random.randint(0, length - block_len) if length > block_len else 0
        mask[start:start + block_len] = True
        return mask

    def _structured_mask(self, length: int) -> torch.Tensor:
        mask = torch.zeros(length, dtype=torch.bool)
        num_masks = max(self.min_masks, int(length * self.mask_ratio / max(self.mask_length, 1)))
        num_masks = min(num_masks, self.max_masks)
        if num_masks > 0:
            interval = length // num_masks
            for i in range(num_masks):
                start = i * interval
                mask[start:min(start + self.mask_length, length)] = True
        return mask

    # ── 频谱图 mask → 波形 mask ───────────────

    def _spectrogram_mask_to_waveform_mask(
        self, spectrogram_mask: torch.Tensor, waveform_length: int, hop_length: int
    ) -> torch.Tensor:
        """
        将频谱图时间帧 mask 映射到波形采样点 mask。
        每个被掩码的帧对应波形中 [frame×hop, frame×hop+hop) 区间。
        注意：环形掩码在频谱图轴上可能是"首尾各一段"，
        映射到波形轴后同样保持该结构，模型侧无需任何改动。
        """
        waveform_mask = torch.zeros(waveform_length, dtype=torch.bool)
        for frame_idx in range(len(spectrogram_mask)):
            if spectrogram_mask[frame_idx]:
                start = frame_idx * hop_length
                waveform_mask[start:min(start + hop_length, waveform_length)] = True
        return waveform_mask


# ──────────────────────────────────────────────
#  数据集
# ──────────────────────────────────────────────

class AudioDataset(Dataset):
    """
    音频数据集。
    接收一个文件路径列表（由 collect_audio_files 或外部生成）。
    返回原始数据 + mask 标记，不在数据端修改数据本身。
    """

    def __init__(
        self,
        config: dict,
        file_list: List[Optional[str]],
        mode: str = "train",
    ):
        self.config = config
        self.file_list = file_list
        self.mode = mode

        data_cfg = config['data']
        self.sample_rate   = data_cfg['sample_rate']
        self.audio_length  = data_cfg['audio_length']
        self.n_mels        = data_cfg['n_mels']
        self.n_fft         = data_cfg['n_fft']
        self.hop_length    = data_cfg['hop_length']
        self.win_length    = data_cfg['win_length']
        self.fmin          = data_cfg['fmin']
        self.fmax          = data_cfg['fmax']
        self.synthetic_length = int(self.sample_rate * self.audio_length)

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
        )

        mask_cfg = config['mask']
        self.mask_generator = MaskGenerator(
            mask_ratio=mask_cfg['mask_ratio'],
            mask_length=mask_cfg['mask_length'],
            min_masks=mask_cfg['min_masks'],
            max_masks=mask_cfg['max_masks'],
            mask_type=mask_cfg['mask_type'],
        )

        print(f"Dataset [{mode}]: {len(self.file_list)} files")

    # ── 音频加载 ──────────────────────────────

    def _load_audio(self, filepath: Optional[str]) -> torch.Tensor:
        if filepath is None or not os.path.exists(filepath):
            return self._generate_synthetic_audio()
        try:
            if HAS_SOUNDFILE:
                waveform_np, sr = sf.read(filepath, dtype='float32')
                if waveform_np.ndim == 2:
                    waveform_np = waveform_np.mean(axis=1)
                waveform = torch.from_numpy(waveform_np).unsqueeze(0)
                if sr != self.sample_rate:
                    waveform = T.Resample(sr, self.sample_rate)(waveform)
            else:
                return self._generate_synthetic_audio()
            return self._normalize(waveform)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return self._generate_synthetic_audio()

    def _generate_synthetic_audio(self) -> torch.Tensor:
        t = torch.linspace(0, self.audio_length, self.synthetic_length)
        freqs = [random.uniform(100, 2000) for _ in range(random.randint(2, 5))]
        amps  = [random.uniform(0.1, 0.3)  for _ in range(len(freqs))]
        waveform = torch.zeros(1, self.synthetic_length)
        for freq, amp in zip(freqs, amps):
            waveform += amp * torch.sin(2 * np.pi * freq * t).unsqueeze(0)
        waveform += 0.01 * torch.randn(1, self.synthetic_length)
        return self._normalize(waveform)

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        max_val = waveform.abs().max()
        return waveform / max_val if max_val > 0 else waveform

    def _compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return mel

    # ── Dataset 接口 ──────────────────────────

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        waveform     = self._load_audio(self.file_list[idx])        # [1, T]
        spectrogram  = self._compute_mel_spectrogram(waveform)      # [1, n_mels, T']

        waveform_mask, spectrogram_mask = self.mask_generator.generate_mask(
            waveform_length=waveform.size(-1),
            spectrogram_time_frames=spectrogram.size(-1),
            hop_length=self.hop_length,
        )

        return {
            'waveform':        waveform.squeeze(0),   # [T]
            'spectrogram':     spectrogram,            # [1, n_mels, T']
            'waveform_mask':   waveform_mask,          # [T]
            'spectrogram_mask': spectrogram_mask,      # [T']
        }


# ──────────────────────────────────────────────
#  变长 collate
# ──────────────────────────────────────────────

def variable_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    max_wave_seconds: float = 10.0,
    sample_rate: int = 16000,
    hop_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    变长音频动态 padding collate 函数。

    返回字段：
        waveform           : [B, T_max]
        spectrogram        : [B, 1, n_mels, T'_max]
        waveform_mask      : [B, T_max]
        spectrogram_mask   : [B, T'_max]
        waveform_lengths   : [B]
        spectrogram_lengths: [B]
        padding_mask_wave  : [B, T_max]   True = padding
        padding_mask_spec  : [B, T'_max]  True = padding
    """
    max_wave_samples = int(max_wave_seconds * sample_rate)
    max_spec_frames  = max_wave_samples // hop_length + 1

    # 超长截断
    clipped = []
    for item in batch:
        wl = item['waveform'].shape[0]
        if wl > max_wave_samples:
            start    = random.randint(0, wl - max_wave_samples)
            wave     = item['waveform'][start: start + max_wave_samples]
            wmask    = item['waveform_mask'][start: start + max_wave_samples]
            sl       = item['spectrogram'].shape[2]
            s_start  = min(start // hop_length, max(0, sl - max_spec_frames))
            spec     = item['spectrogram'][:, :, s_start: s_start + max_spec_frames]
            smask    = item['spectrogram_mask'][s_start: s_start + max_spec_frames]
            clipped.append({'waveform': wave, 'spectrogram': spec,
                            'waveform_mask': wmask, 'spectrogram_mask': smask})
        else:
            clipped.append(item)
    batch = clipped

    wave_lengths = [item['waveform'].shape[0]       for item in batch]
    spec_lengths = [item['spectrogram'].shape[2]    for item in batch]
    n_mels       = batch[0]['spectrogram'].shape[1]
    B            = len(batch)
    max_wl       = max(wave_lengths)
    max_sl       = max(spec_lengths)

    waveform_padded    = torch.zeros(B, max_wl)
    spectrogram_padded = torch.zeros(B, 1, n_mels, max_sl)
    wm_padded          = torch.zeros(B, max_wl,  dtype=torch.bool)
    sm_padded          = torch.zeros(B, max_sl,  dtype=torch.bool)
    pad_wave           = torch.ones(B, max_wl,   dtype=torch.bool)   # True = padding
    pad_spec           = torch.ones(B, max_sl,   dtype=torch.bool)

    for i, item in enumerate(batch):
        wl, sl = wave_lengths[i], spec_lengths[i]
        waveform_padded[i, :wl]          = item['waveform']
        spectrogram_padded[i, :, :, :sl] = item['spectrogram']
        wm_padded[i, :wl]                = item['waveform_mask']
        sm_padded[i, :sl]                = item['spectrogram_mask']
        pad_wave[i, :wl]                 = False
        pad_spec[i, :sl]                 = False

    return {
        'waveform':            waveform_padded,
        'spectrogram':         spectrogram_padded,
        'waveform_mask':       wm_padded,
        'spectrogram_mask':    sm_padded,
        'waveform_lengths':    torch.tensor(wave_lengths, dtype=torch.long),
        'spectrogram_lengths': torch.tensor(spec_lengths, dtype=torch.long),
        'padding_mask_wave':   pad_wave,
        'padding_mask_spec':   pad_spec,
    }


# ──────────────────────────────────────────────
#  DataLoader 工厂
# ──────────────────────────────────────────────

def create_dataloaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练 / 验证 DataLoader。

    文件收集由 collect_audio_files(config) 完成，
    支持 simple / dcase / explicit 三种模式。
    """
    all_files = collect_audio_files(config)

    if not all_files:
        print("No audio files found. Falling back to synthetic data (100 samples).")
        all_files = [None] * 100

    random.shuffle(all_files)
    split_idx   = int(len(all_files) * config['data']['train_split'])
    train_files = all_files[:split_idx]
    val_files   = all_files[split_idx:]

    train_dataset = AudioDataset(config, train_files, mode="train")
    val_dataset   = AudioDataset(config, val_files,   mode="val")

    sr       = config['data']['sample_rate']
    hop      = config['data']['hop_length']
    max_secs = config['data'].get('max_audio_seconds', 10.0)

    _collate = functools.partial(
        variable_length_collate_fn,
        max_wave_seconds=max_secs,
        sample_rate=sr,
        hop_length=hop,
    )

    train_bs = config['training']['batch_size']
    val_bs   = max(1, train_bs // 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate,
    )

    return train_loader, val_loader
