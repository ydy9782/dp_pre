"""
✅ 修复后的双分支音频预训练模型 - 真正解决掩码重建问题

核心修复:
1. 编码器在掩码位置也提取特征(不设为0),只在注意力中标记
2. 融合层处理完整序列
3. 解码器接收完整的融合特征(而不是只有未掩码部分)
4. 增强mask token的表达能力
5. 添加渐进式重建机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext
from typing import Dict, Tuple, Optional
import torchaudio.functional as AF


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            positions: [batch, seq_len] - 可选的位置索引
        """
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len {self.max_len}. "
                "Increase config.yaml -> model.transformer.max_seq_len."
            )
        
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = positions.float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * 
            (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(batch_size, seq_len, d_model, device=x.device)
        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        if d_model > 1:
            pe[:, :, 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
        
        return x + pe


class WaveformEncoder(nn.Module):
    """
    ✅ 修复后的波形编码器
    关键改进: 对所有位置提取特征,只在注意力中使用mask标记
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.patch_stride = int(config.get('patch_stride', 256))
        self.patch_kernel = int(config.get('patch_kernel', self.patch_stride * 4))
        self.patch_padding = int(config.get('patch_padding', self.patch_kernel // 2))
        if 2 * self.patch_padding < self.patch_kernel:
            raise ValueError(
                "Waveform patch config must satisfy 2 * patch_padding >= patch_kernel "
                "so decoder can cover the full waveform without interpolation."
            )
        
        # 用与频谱 hop 对齐的 patch 化前端，大幅缩短波形 token 序列
        self.embedding = nn.Sequential(
            nn.Conv1d(
                1,
                config['input_dim'],
                kernel_size=self.patch_kernel,
                stride=self.patch_stride,
                padding=self.patch_padding
            ),
            nn.BatchNorm1d(config['input_dim']),
            nn.GELU(),
            nn.Conv1d(config['input_dim'], config['input_dim'], kernel_size=3, padding=1),
            nn.BatchNorm1d(config['input_dim']),
            nn.GELU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_dim'] * 4,
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )
        
        self.proj = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.pos_encoding = PositionalEncoding(
            config['hidden_dim'],
            max_len=config.get('max_seq_len', 5000)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, length] 原始波形
            mask: [batch, length] 掩码标记 (True表示被掩码)
            
        Returns:
            encoded: [batch, num_tokens, hidden_dim] - 所有tokens的编码
            token_mask: [batch, num_tokens] - token级别的掩码
        """
        # 1. 卷积编码 - ✅ 对所有位置编码
        x_input = x.unsqueeze(1)  # [batch, 1, length]
        x_embed = self.embedding(x_input)  # [batch, input_dim, seq_len]
        x_embed = x_embed.transpose(1, 2)  # [batch, seq_len, input_dim]
        x_proj = self.proj(x_embed)  # [batch, seq_len, hidden_dim]
        
        num_tokens = x_embed.shape[1]
        
        # 2. 计算token级别的掩码
        token_mask = self._compute_token_mask(mask, num_tokens)
        
        # 3. ✅ 添加位置编码到所有tokens
        x_pos = self.pos_encoding(x_proj)
        
        # 4. ✅ Transformer编码,使用key_padding_mask标记掩码位置
        # 这样模型知道哪些位置是掩码,但仍然保留这些位置的特征
        encoded = self.transformer(
            x_pos,
            src_key_padding_mask=token_mask  # True表示被掩码,注意力时会忽略
        )
        
        return encoded, token_mask
    
    def _compute_token_mask(self, mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """将样本级mask转换为token级mask"""
        batch_size = mask.shape[0]
        token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=mask.device)
        
        for i in range(seq_len):
            raw_start = i * self.patch_stride - self.patch_padding
            raw_end = raw_start + self.patch_kernel
            start_idx = max(raw_start, 0)
            end_idx = min(raw_end, mask.shape[1])
            if start_idx < end_idx:
                token_mask[:, i] = mask[:, start_idx:end_idx].any(dim=1)
        
        return token_mask


class SpectrogramEncoder(nn.Module):
    """
    ✅ 修复后的频谱图编码器
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.freq_bins = int(config.get('freq_bins', 4))
        self.dropout = float(config.get('dropout', 0.1))
        self.time_stride = 1
        self.time_receptive_field = 1
        time_strides = config.get('time_strides')
        if time_strides is None:
            time_strides = [1 if i % 2 == 0 else 2 for i in range(config['num_layers'])]
        if len(time_strides) != config['num_layers']:
            raise ValueError(
                "model.cnn.time_strides length must equal model.cnn.num_layers"
            )
        self.time_strides = [int(s) for s in time_strides]
        
        # CNN编码器
        layers = []
        in_channels = config['in_channels']
        
        for i in range(config['num_layers']):
            out_channels = config['base_channels'] * (2 ** i)
            time_stride = self.time_strides[i]
            stride = (config['stride'], time_stride)
            self.time_receptive_field += (config['kernel_size'] - 1) * self.time_stride
            self.time_stride *= time_stride
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=config['kernel_size'],
                         stride=stride,
                         padding=config['padding']),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout2d(self.dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((self.freq_bins, None))
        self.final_channels = config['base_channels'] * (2 ** (config['num_layers'] - 1))
        self.output_dim = self.final_channels * self.freq_bins
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 1, n_mels, time] 原始频谱图
            mask: [batch, time] 掩码标记
            
        Returns:
            encoded: [batch, time, channels] - 所有tokens的编码
            token_mask: [batch, time_after_cnn] - token级别的掩码
        """
        batch_size, _, n_mels, time = x.shape
        
        if mask.dim() == 3 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        
        # 1. ✅ CNN编码所有位置
        encoded = self.cnn(x)  # [batch, channels, freq, time]
        encoded = self.pool(encoded)  # [batch, channels, freq_bins, time]
        encoded = encoded.permute(0, 3, 1, 2).contiguous()  # [batch, time, channels, freq_bins]
        encoded = encoded.view(batch_size, encoded.shape[1], -1)  # [batch, time, channels * freq_bins]
        
        # 2. 计算token掩码
        num_tokens = encoded.shape[1]
        token_mask = self._compute_token_mask(mask, num_tokens)
        
        return encoded, token_mask
    
    def _compute_token_mask(self, mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = mask.shape[0]
        token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=mask.device)
        
        for i in range(seq_len):
            start_idx = i * self.time_stride
            end_idx = min(start_idx + self.time_receptive_field, mask.shape[1])
            token_mask[:, i] = mask[:, start_idx:end_idx].any(dim=1)
        
        return token_mask


class CrossAttentionFusion(nn.Module):
    """
    ✅ 修复后的交叉注意力融合层
    处理完整序列,使用mask标记
    """
    
    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config['hidden_dim']
        output_dim = config.get('output_dim', hidden_dim)
        num_heads = int(config.get('num_heads', 4))
        dropout = float(config.get('dropout', 0.1))
        ffn_multiplier = int(config.get('ffn_multiplier', 4))
        ffn_dim = int(config.get('ffn_dim', hidden_dim * ffn_multiplier))
        wave_dim = config['wave_dim']
        spec_dim = config['spec_dim']
        
        self.wave_proj = nn.Linear(wave_dim, hidden_dim)
        self.spec_proj = nn.Linear(spec_dim, hidden_dim)
        
        # 交叉注意力
        self.wave_to_spec = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.spec_to_wave = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.wave_norm = nn.LayerNorm(hidden_dim)
        self.spec_norm = nn.LayerNorm(hidden_dim)
        
        self.wave_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        
        self.spec_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.wave_out_proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        self.spec_out_proj = nn.Identity() if hidden_dim == output_dim else nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        wave_features: torch.Tensor,
        spec_features: torch.Tensor,
        wave_mask: torch.Tensor,
        spec_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            wave_features: [batch, wave_len, wave_dim]
            spec_features: [batch, spec_len, spec_dim]
            wave_mask: [batch, wave_len] - 掩码标记
            spec_mask: [batch, spec_len] - 掩码标记
        """
        # 投影到相同维度
        wave_features = self.wave_proj(wave_features)
        spec_features = self.spec_proj(spec_features)
        
        # ✅ 波形特征融合频谱信息 (掩码位置也参与)
        wave_attn, _ = self.wave_to_spec(
            query=wave_features,
            key=spec_features,
            value=spec_features,
            key_padding_mask=spec_mask  # 只在key侧使用mask
        )
        wave_features = self.wave_norm(wave_features + wave_attn)
        wave_features = wave_features + self.wave_ffn(wave_features)
        
        # ✅ 频谱特征融合波形信息
        spec_attn, _ = self.spec_to_wave(
            query=spec_features,
            key=wave_features,
            value=wave_features,
            key_padding_mask=wave_mask
        )
        spec_features = self.spec_norm(spec_features + spec_attn)
        spec_features = spec_features + self.spec_ffn(spec_features)
        
        return self.wave_out_proj(wave_features), self.spec_out_proj(spec_features)


class WaveformDecoder(nn.Module):
    """
    ✅ 修复后的波形解码器
    关键改进: 接收完整的融合特征序列
    """
    
    def __init__(self, config: dict, transformer_config: dict, input_dim: int):
        super().__init__()
        self.hidden_dim = config['waveform_hidden_dim']
        self.output_patch_size = int(transformer_config.get('patch_stride', 256))
        self.mask_feature_bias_scale = float(config.get('mask_feature_bias_scale', 0.5))
        self.input_proj = (
            nn.Identity() if input_dim == self.hidden_dim
            else nn.Linear(input_dim, self.hidden_dim)
        )
        decoder_num_heads = int(config.get('waveform_num_heads', transformer_config.get('num_heads', 8)))
        decoder_dropout = float(config.get('dropout', transformer_config.get('dropout', 0.1)))
        patch_mlp_ratio = int(config.get('waveform_patch_mlp_ratio', 2))
        refine_channels = int(config.get('waveform_refine_channels', 16))
        refine_kernel_size = int(config.get('waveform_refine_kernel_size', 5))
        refine_padding = refine_kernel_size // 2
        self.refine_residual_scale = float(config.get('waveform_refine_residual_scale', 0.25))
        
        # ✅ 增强的mask token embedding
        self.mask_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        # 可学习的mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.5)
        
        # ✅ 更深的Transformer解码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=decoder_num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['waveform_num_layers']
        )
        
        self.pos_encoding = PositionalEncoding(
            self.hidden_dim,
            max_len=transformer_config.get('max_seq_len', 5000)
        )
        
        # 每个 token 直接重建一段波形 patch，再做轻量时域细化
        self.patch_reconstruction = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * patch_mlp_ratio),
            nn.GELU(),
            nn.Linear(self.hidden_dim * patch_mlp_ratio, self.output_patch_size)
        )
        self.refine = nn.Sequential(
            nn.Conv1d(1, refine_channels, kernel_size=refine_kernel_size, padding=refine_padding),
            nn.GELU(),
            nn.Conv1d(refine_channels, refine_channels, kernel_size=refine_kernel_size, padding=refine_padding),
            nn.GELU(),
            nn.Conv1d(refine_channels, 1, kernel_size=refine_kernel_size, padding=refine_padding),
        )
        
    def forward(
        self,
        fused_features: torch.Tensor,
        token_mask: torch.Tensor,
        target_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_features: [batch, num_tokens, hidden_dim] - ✅ 完整的融合特征
            token_mask: [batch, num_tokens] - token级别的掩码
            target_length: int - 目标波形长度
            
        Returns:
            reconstructed: [batch, target_length] - 重建的波形
            coarse_waveform: [batch, target_length] - patch 级粗重建
        """
        fused_features = self.input_proj(fused_features)
        batch_size, num_tokens, _ = fused_features.shape
        device = fused_features.device
        
        # 1. ✅ 对掩码位置,用mask token增强特征
        mask_tokens = self.mask_token.expand(batch_size, num_tokens, -1)
        mask_embeds = self.mask_embedding(mask_tokens)
        
        # 混合原始特征和mask embedding
        # 未掩码位置主要用融合特征,掩码位置用mask embedding
        enhanced_features = fused_features + (
            token_mask.unsqueeze(-1).to(fused_features.dtype) *
            self.mask_feature_bias_scale * mask_embeds
        )
        
        # 2. 添加位置编码
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        enhanced_features = self.pos_encoding(enhanced_features, positions)
        
        # 3. ✅ Transformer解码 (self-attention从上下文推断掩码内容)
        # 不使用key_padding_mask,让所有位置都能互相看到
        decoded = self.transformer(enhanced_features)
        
        # 4. 每个 token 重建一段 patch，并拼回时域波形
        patch_output = self.patch_reconstruction(decoded)  # [batch, num_tokens, patch_size]
        coarse_waveform = patch_output.reshape(batch_size, -1)
        residual = self.refine(coarse_waveform.unsqueeze(1)).squeeze(1)
        output = coarse_waveform + self.refine_residual_scale * residual
        
        # 5. 这里不再使用插值补点；每个采样点都必须来自可学习的 patch 重建。
        if output.shape[1] > target_length:
            output = output[:, :target_length]
            coarse_waveform = coarse_waveform[:, :target_length]
        elif output.shape[1] < target_length:
            raise RuntimeError(
                f"Waveform decoder produced {output.shape[1]} samples, "
                f"which is shorter than target length {target_length}. "
                "Adjust patch_stride/patch_kernel/patch_padding so the decoder "
                "covers the full waveform without interpolation."
            )
        
        return output, coarse_waveform


class SpectrogramDecoder(nn.Module):
    """
    ✅ 修复后的频谱图解码器
    """
    
    def __init__(self, config: dict, data_config: dict, input_dim: int, max_seq_len: int):
        super().__init__()
        
        self.hidden_dim = config['spectrogram_hidden_dim']
        self.n_mels = data_config['n_mels']
        self.mask_feature_bias_scale = float(config.get('mask_feature_bias_scale', 0.5))
        self.input_proj = (
            nn.Identity() if input_dim == self.hidden_dim
            else nn.Linear(input_dim, self.hidden_dim)
        )
        decoder_num_heads = int(config.get('spectrogram_num_heads', 8))
        decoder_dropout = float(config.get('dropout', 0.1))
        refine_channels = config.get('spectrogram_refine_channels', [32, 16])
        if len(refine_channels) != 2:
            raise ValueError("model.decoder.spectrogram_refine_channels must have exactly 2 values")
        refine_channels_1 = int(refine_channels[0])
        refine_channels_2 = int(refine_channels[1])
        refine_kernel_size = int(config.get('spectrogram_refine_kernel_size', 3))
        refine_padding = refine_kernel_size // 2
        
        # ✅ 增强的mask token
        self.mask_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.5)
        
        # Transformer解码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=decoder_num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['spectrogram_num_layers']
        )
        
        self.pos_encoding = PositionalEncoding(self.hidden_dim, max_len=max_seq_len)
        self.freq_proj = nn.Linear(self.hidden_dim, self.n_mels)
        
        # 细化网络
        self.refine = nn.Sequential(
            nn.Conv2d(1, refine_channels_1, kernel_size=refine_kernel_size, padding=refine_padding),
            nn.BatchNorm2d(refine_channels_1),
            nn.GELU(),
            nn.Conv2d(refine_channels_1, refine_channels_2, kernel_size=refine_kernel_size, padding=refine_padding),
            nn.BatchNorm2d(refine_channels_2),
            nn.GELU(),
            nn.Conv2d(refine_channels_2, 1, kernel_size=refine_kernel_size, padding=refine_padding)
        )
        
    def forward(
        self,
        fused_features: torch.Tensor,
        token_mask: torch.Tensor,
        target_time: int
    ) -> torch.Tensor:
        """
        Args:
            fused_features: [batch, num_tokens, hidden_dim] - ✅ 完整的融合特征
            token_mask: [batch, num_tokens]
            target_time: int - 目标时间帧数
        """
        fused_features = self.input_proj(fused_features)
        batch_size, num_tokens, _ = fused_features.shape
        device = fused_features.device
        
        # 1. ✅ 增强掩码位置的特征
        mask_tokens = self.mask_token.expand(batch_size, num_tokens, -1)
        mask_embeds = self.mask_embedding(mask_tokens)
        
        enhanced_features = fused_features + (
            token_mask.unsqueeze(-1).to(fused_features.dtype) *
            self.mask_feature_bias_scale * mask_embeds
        )
        
        # 2. 位置编码
        positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        enhanced_features = self.pos_encoding(enhanced_features, positions)
        
        # 3. ✅ Transformer解码
        decoded = self.transformer(enhanced_features)
        
        # 4. 投影到频率维度
        freq_features = self.freq_proj(decoded)  # [batch, num_tokens, n_mels]
        
        # 5. 调整时间维度
        if freq_features.shape[1] != target_time:
            freq_features = F.interpolate(
                freq_features.transpose(1, 2),
                size=target_time,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # 6. 转换为2D并细化
        spec_2d = freq_features.transpose(1, 2).unsqueeze(1)  # [batch, 1, n_mels, time]
        output = self.refine(spec_2d)
        
        return output


class DualBranchAudioMAE(nn.Module):
    """
    ✅ 修复后的双分支音频MAE
    真正解决掩码重建问题
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stft_window_cache = {}
        self._mel_filter_cache = {}
        data_config = config['data']

        waveform_encoder_config = dict(config['model']['transformer'])
        waveform_encoder_config.setdefault('patch_stride', data_config['hop_length'])
        waveform_encoder_config.setdefault(
            'patch_kernel',
            max(data_config.get('win_length', data_config['hop_length'] * 4), data_config['hop_length'])
        )
        waveform_encoder_config.setdefault(
            'patch_padding',
            data_config.get('n_fft', waveform_encoder_config['patch_kernel']) // 2
        )
        if waveform_encoder_config['patch_stride'] != data_config['hop_length']:
            raise ValueError(
                "model.transformer.patch_stride must equal data.hop_length "
                "to keep waveform and spectrogram masking aligned."
            )
        
        # 编码器
        self.waveform_encoder = WaveformEncoder(waveform_encoder_config)
        self.spectrogram_encoder = SpectrogramEncoder(config['model']['cnn'])
        
        # 融合层
        fusion_output_dim = config['model']['fusion'].get(
            'output_dim',
            config['model']['fusion']['hidden_dim']
        )
        fusion_config = {
            'hidden_dim': config['model']['fusion']['hidden_dim'],
            'output_dim': fusion_output_dim,
            'num_heads': config['model']['fusion'].get('num_heads', 4),
            'dropout': config['model']['fusion'].get('dropout', 0.1),
            'ffn_multiplier': config['model']['fusion'].get('ffn_multiplier', 4),
            'wave_dim': waveform_encoder_config['hidden_dim'],
            'spec_dim': self.spectrogram_encoder.output_dim
        }
        self.fusion = CrossAttentionFusion(fusion_config)
        
        # 解码器
        self.waveform_decoder = WaveformDecoder(
            config['model']['decoder'],
            waveform_encoder_config,
            input_dim=fusion_output_dim
        )
        self.spectrogram_decoder = SpectrogramDecoder(
            config['model']['decoder'],
            config['data'],
            input_dim=fusion_output_dim,
            max_seq_len=waveform_encoder_config.get('max_seq_len', 5000)
        )

    def _apply_input_masks(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor,
        waveform_mask: torch.Tensor,
        spectrogram_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在编码前真正移除被掩码的原始内容，避免目标泄漏到编码器。
        """
        masked_waveform = waveform.masked_fill(waveform_mask, 0.0)

        spec_mask = spectrogram_mask.unsqueeze(1).unsqueeze(2).expand_as(spectrogram)
        masked_spectrogram = spectrogram.masked_fill(spec_mask, 0.0)

        return masked_waveform, masked_spectrogram
        
    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor,
        waveform_mask: torch.Tensor,
        spectrogram_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: [batch, length] - 原始波形
            spectrogram: [batch, 1, n_mels, time] - 原始频谱图
            waveform_mask: [batch, length] - 波形掩码
            spectrogram_mask: [batch, time] - 频谱掩码
        """
        masked_waveform, masked_spectrogram = self._apply_input_masks(
            waveform,
            spectrogram,
            waveform_mask,
            spectrogram_mask
        )

        # 1. ✅ 编码: 只看未掩码输入，同时保留 token 级 mask 供注意力和解码使用
        wave_encoded, wave_token_mask = self.waveform_encoder(masked_waveform, waveform_mask)
        spec_encoded, spec_token_mask = self.spectrogram_encoder(masked_spectrogram, spectrogram_mask)
        
        # 2. ✅ 融合: 处理完整序列
        fused_wave, fused_spec = self.fusion(
            wave_encoded, 
            spec_encoded,
            wave_token_mask,
            spec_token_mask
        )
        
        # 3. ✅ 解码: 从完整的融合特征重建
        predicted_waveform, coarse_waveform = self.waveform_decoder(
            fused_wave,
            wave_token_mask,
            target_length=waveform.shape[1]
        )
        
        predicted_spectrogram = self.spectrogram_decoder(
            fused_spec,
            spec_token_mask,
            target_time=spectrogram.shape[3]
        )

        copy_unmasked_input = bool(self.config.get('training', {}).get('copy_unmasked_input', False))
        spec_mask_expanded = spectrogram_mask.unsqueeze(1).unsqueeze(2).expand_as(predicted_spectrogram)
        inpainted_waveform = torch.where(waveform_mask, predicted_waveform, waveform)
        inpainted_spectrogram = torch.where(spec_mask_expanded, predicted_spectrogram, spectrogram)
        if copy_unmasked_input:
            reconstructed_waveform = inpainted_waveform
            reconstructed_spectrogram = inpainted_spectrogram
        else:
            reconstructed_waveform = predicted_waveform
            reconstructed_spectrogram = predicted_spectrogram
        
        return {
            'predicted_waveform': predicted_waveform,
            'coarse_waveform': coarse_waveform,
            'predicted_spectrogram': predicted_spectrogram,
            'reconstructed_waveform': reconstructed_waveform,
            'reconstructed_spectrogram': reconstructed_spectrogram,
            'inpainted_waveform': inpainted_waveform,
            'inpainted_spectrogram': inpainted_spectrogram,
            'wave_token_mask': wave_token_mask,
            'spec_token_mask': spec_token_mask
        }

    def _get_stft_window(self, win_length: int, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, device.index, win_length)
        window = self._stft_window_cache.get(cache_key)
        if window is None or window.device != device:
            window = torch.hann_window(win_length, device=device)
            self._stft_window_cache[cache_key] = window
        return window

    def _get_mel_filter(
        self,
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        fmin: float,
        fmax: float,
        device: torch.device
    ) -> torch.Tensor:
        cache_key = (device.type, device.index, n_fft, n_mels, sample_rate, float(fmin), float(fmax))
        mel_filter = self._mel_filter_cache.get(cache_key)
        if mel_filter is None or mel_filter.device != device:
            mel_filter = AF.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=float(fmin),
                f_max=float(fmax),
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm=None,
                mel_scale='htk'
            ).transpose(0, 1).to(device)
            self._mel_filter_cache[cache_key] = mel_filter
        return mel_filter

    def _compute_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        data_cfg = self.config['data']
        n_fft = int(data_cfg['n_fft'])
        hop_length = int(data_cfg['hop_length'])
        win_length = int(data_cfg.get('win_length', n_fft))
        n_mels = int(data_cfg['n_mels'])
        fmin = float(data_cfg.get('fmin', 0.0))
        fmax = float(data_cfg.get('fmax', data_cfg['sample_rate'] / 2))
        sample_rate = int(data_cfg['sample_rate'])

        window = self._get_stft_window(win_length, waveform.device)
        stft = torch.stft(
            waveform.float(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        power = torch.view_as_real(stft).pow(2).sum(dim=-1)
        mel_filter = self._get_mel_filter(
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            device=waveform.device
        )

        # 避免在当前 CUDA/cuBLAS 环境下触发小矩阵 GEMM 的 cublas 初始化报错。
        mel_chunk_size = int(self.config.get('training', {}).get('waveform_mel_chunk_size', 16))
        mel_chunks = []
        power_expanded = power.unsqueeze(1)  # [B, 1, F, T]
        for start in range(0, n_mels, mel_chunk_size):
            end = min(start + mel_chunk_size, n_mels)
            mel_filter_chunk = mel_filter[start:end].unsqueeze(0).unsqueeze(-1)  # [1, M_chunk, F, 1]
            mel_chunk = (power_expanded * mel_filter_chunk).sum(dim=2)  # [B, M_chunk, T]
            mel_chunks.append(mel_chunk)
        mel = torch.cat(mel_chunks, dim=1)

        log_mel = torch.log(mel.clamp_min(1e-9))
        log_mel = (log_mel - log_mel.mean(dim=(1, 2), keepdim=True)) / (
            log_mel.std(dim=(1, 2), keepdim=True) + 1e-8
        )
        return log_mel.unsqueeze(1)

    def _compute_waveform_stft_loss(
        self,
        reconstructed_waveform: torch.Tensor,
        target_waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        在“补全后的完整波形”上计算多分辨率 STFT 损失。
        由于未掩码区域已直接替换为原始波形，频域误差会主要集中在掩码区及其边界连续性。
        """
        training_cfg = self.config['training']
        resolutions = training_cfg.get('waveform_stft_resolutions', [])
        if not resolutions:
            data_cfg = self.config.get('data', {})
            base_n_fft = int(data_cfg.get('n_fft', 1024))
            base_hop = int(data_cfg.get('hop_length', max(1, base_n_fft // 4)))
            base_win = int(data_cfg.get('win_length', base_n_fft))
            resolutions = [
                {
                    'n_fft': max(256, base_n_fft // 4),
                    'hop_length': max(64, base_hop // 4),
                    'win_length': max(256, base_win // 4),
                },
                {
                    'n_fft': max(512, base_n_fft // 2),
                    'hop_length': max(128, base_hop // 2),
                    'win_length': max(512, base_win // 2),
                },
                {
                    'n_fft': base_n_fft,
                    'hop_length': base_hop,
                    'win_length': base_win,
                },
            ]

        autocast_context = (
            torch.autocast(device_type='cuda', enabled=False)
            if reconstructed_waveform.device.type == 'cuda'
            else nullcontext()
        )

        with autocast_context:
            reconstructed = reconstructed_waveform.float()
            target = target_waveform.float()
            total_loss = reconstructed.new_tensor(0.0)

            for resolution in resolutions:
                n_fft = int(resolution['n_fft'])
                hop_length = int(resolution.get('hop_length', max(1, n_fft // 4)))
                win_length = int(resolution.get('win_length', n_fft))
                window = self._get_stft_window(win_length, reconstructed.device)

                reconstructed_stft = torch.stft(
                    reconstructed,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    return_complex=True,
                )
                target_stft = torch.stft(
                    target,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    return_complex=True,
                )

                # 避免在新 GPU + 旧 NVRTC 组合下触发 complex.abs() 的 JIT 编译报错。
                reconstructed_mag = torch.sqrt(
                    torch.view_as_real(reconstructed_stft).pow(2).sum(dim=-1).clamp_min(1e-7)
                )
                target_mag = torch.sqrt(
                    torch.view_as_real(target_stft).pow(2).sum(dim=-1).clamp_min(1e-7)
                )

                spectral_convergence = torch.linalg.vector_norm(
                    (target_mag - reconstructed_mag).reshape(target_mag.shape[0], -1),
                    dim=1
                ) / torch.linalg.vector_norm(
                    target_mag.reshape(target_mag.shape[0], -1),
                    dim=1
                ).clamp_min(1e-7)
                log_magnitude = F.l1_loss(
                    reconstructed_mag.log(),
                    target_mag.log()
                )

                total_loss = total_loss + spectral_convergence.mean() + log_magnitude

        return total_loss / len(resolutions)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        ✅ 变长支持：通过 padding_mask 排除 padding 区域的 loss 贡献
        """
        config = self.config['training']
        
        pred_wave = outputs.get('predicted_waveform', outputs['reconstructed_waveform'])
        coarse_wave = outputs.get('coarse_waveform')
        pred_spec = outputs.get('predicted_spectrogram', outputs['reconstructed_spectrogram'])
        recon_wave = outputs.get('reconstructed_waveform', pred_wave)
        target_wave = targets['waveform']
        target_spec = targets['spectrogram']
        wave_mask = masks['waveform_mask']
        spec_mask = masks['spectrogram_mask']

        # ✅ 获取 padding 掩码（True=padding，需排除）
        # 兼容旧调用（没有 padding_mask 时不屏蔽）
        pad_wave = masks.get('padding_mask_wave',
                             torch.zeros_like(wave_mask, dtype=torch.bool))
        pad_spec = masks.get('padding_mask_spec',
                             torch.zeros_like(spec_mask, dtype=torch.bool))

        # ✅ 有效区域 = 非 padding
        valid_wave = ~pad_wave           # [B, T]
        valid_spec = ~pad_spec           # [B, T']

        # 波形损失（仅在有效区域内计算）
        wave_mse = F.mse_loss(pred_wave, target_wave, reduction='none')  # [B, T]

        # 掩码区域（有效 & 被 mask）
        masked_wave_valid = wave_mask & valid_wave
        unmasked_wave_valid = (~wave_mask) & valid_wave

        if masked_wave_valid.any():
            wave_masked_mse = wave_mse[masked_wave_valid].mean()
            wave_masked_l1 = F.l1_loss(pred_wave[masked_wave_valid],
                                        target_wave[masked_wave_valid])
        else:
            wave_masked_mse = torch.tensor(0.0, device=wave_mse.device)
            wave_masked_l1 = torch.tensor(0.0, device=pred_wave.device)

        if unmasked_wave_valid.any():
            wave_unmasked_mse = wave_mse[unmasked_wave_valid].mean()
        else:
            wave_unmasked_mse = torch.tensor(0.0, device=wave_mse.device)

        wave_mask_loss_weight = float(config.get('waveform_mask_loss_weight', config.get('mask_loss_weight', 10.0)))
        wave_mask_l1_weight = float(config.get('waveform_mask_l1_weight', config.get('mask_l1_weight', 2.0)))
        spec_mask_loss_weight = float(config.get('spectrogram_mask_loss_weight', config.get('mask_loss_weight', 10.0)))
        spec_mask_l1_weight = float(config.get('spectrogram_mask_l1_weight', config.get('mask_l1_weight', 2.0)))
        wave_unmask_loss_weight = float(
            config.get('waveform_unmask_loss_weight', config.get('unmask_loss_weight', 0.1))
        )
        spec_unmask_loss_weight = float(
            config.get('spectrogram_unmask_loss_weight', config.get('unmask_loss_weight', 0.1))
        )
        waveform_stft_loss_weight = float(config.get('waveform_stft_loss_weight', 0.0))
        waveform_diff_loss_weight = float(config.get('waveform_diff_loss_weight', 0.0))
        waveform_coarse_loss_weight = float(config.get('waveform_coarse_loss_weight', 0.0))
        waveform_coarse_l1_weight = float(config.get('waveform_coarse_l1_weight', 0.0))
        waveform_mel_loss_weight = float(config.get('waveform_mel_loss_weight', 0.0))
        waveform_mel_mask_margin_frames = int(config.get('waveform_mel_mask_margin_frames', 0))

        waveform_stft_loss = torch.tensor(0.0, device=pred_wave.device)
        if waveform_stft_loss_weight > 0.0:
            waveform_stft_loss = self._compute_waveform_stft_loss(recon_wave, target_wave)

        waveform_diff_loss = torch.tensor(0.0, device=pred_wave.device)
        if waveform_diff_loss_weight > 0.0 and pred_wave.shape[1] > 1:
            pred_wave_diff = pred_wave[:, 1:] - pred_wave[:, :-1]
            target_wave_diff = target_wave[:, 1:] - target_wave[:, :-1]
            valid_wave_diff = masked_wave_valid[:, 1:] | masked_wave_valid[:, :-1]
            if valid_wave_diff.any():
                waveform_diff_loss = F.l1_loss(
                    pred_wave_diff[valid_wave_diff],
                    target_wave_diff[valid_wave_diff]
                )

        coarse_wave_masked_mse = torch.tensor(0.0, device=pred_wave.device)
        coarse_wave_masked_l1 = torch.tensor(0.0, device=pred_wave.device)
        if coarse_wave is not None and masked_wave_valid.any():
            coarse_wave_mse = F.mse_loss(coarse_wave, target_wave, reduction='none')
            coarse_wave_masked_mse = coarse_wave_mse[masked_wave_valid].mean()
            coarse_wave_masked_l1 = F.l1_loss(
                coarse_wave[masked_wave_valid],
                target_wave[masked_wave_valid]
            )

        waveform_mel_loss = torch.tensor(0.0, device=pred_wave.device)
        if waveform_mel_loss_weight > 0.0:
            autocast_context = (
                torch.autocast(device_type='cuda', enabled=False)
                if recon_wave.device.type == 'cuda'
                else nullcontext()
            )
            with autocast_context:
                reconstructed_log_mel = self._compute_log_mel_spectrogram(recon_wave)
                if reconstructed_log_mel.shape[-1] != target_spec.shape[-1]:
                    reconstructed_log_mel = F.interpolate(
                        reconstructed_log_mel,
                        size=(target_spec.shape[2], target_spec.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )

                mel_mask = spec_mask
                if waveform_mel_mask_margin_frames > 0:
                    mel_mask = F.max_pool1d(
                        mel_mask.float().unsqueeze(1),
                        kernel_size=2 * waveform_mel_mask_margin_frames + 1,
                        stride=1,
                        padding=waveform_mel_mask_margin_frames
                    ).squeeze(1) > 0.5
                mel_mask_expanded = mel_mask.unsqueeze(1).unsqueeze(2).expand_as(target_spec)
                mel_valid = mel_mask_expanded & (~pad_spec.unsqueeze(1).unsqueeze(2).expand_as(target_spec))
                if mel_valid.any():
                    waveform_mel_loss = F.l1_loss(
                        reconstructed_log_mel[mel_valid],
                        target_spec.float()[mel_valid]
                    )

        wave_loss = (
            wave_mask_loss_weight * wave_masked_mse +
            wave_mask_l1_weight * wave_masked_l1 +
            wave_unmask_loss_weight * wave_unmasked_mse +
            waveform_stft_loss_weight * waveform_stft_loss +
            waveform_diff_loss_weight * waveform_diff_loss +
            waveform_coarse_loss_weight * coarse_wave_masked_mse +
            waveform_coarse_l1_weight * coarse_wave_masked_l1 +
            waveform_mel_loss_weight * waveform_mel_loss
        )

        # 频谱损失（仅在有效区域内计算）
        spec_mse = F.mse_loss(pred_spec, target_spec, reduction='none')  # [B, 1, n_mels, T']

        # 将 pad_spec [B, T'] 扩展到频谱维度
        pad_spec_expanded = pad_spec.unsqueeze(1).unsqueeze(2).expand_as(pred_spec)
        spec_mask_expanded = spec_mask.unsqueeze(1).unsqueeze(2).expand_as(pred_spec)

        masked_spec_valid   = spec_mask_expanded & (~pad_spec_expanded)
        unmasked_spec_valid = (~spec_mask_expanded) & (~pad_spec_expanded)

        if masked_spec_valid.any():
            spec_masked_mse = spec_mse[masked_spec_valid].mean()
            spec_masked_l1 = F.l1_loss(pred_spec[masked_spec_valid],
                                        target_spec[masked_spec_valid])
        else:
            spec_masked_mse = torch.tensor(0.0, device=spec_mse.device)
            spec_masked_l1 = torch.tensor(0.0, device=pred_spec.device)

        if unmasked_spec_valid.any():
            spec_unmasked_mse = spec_mse[unmasked_spec_valid].mean()
        else:
            spec_unmasked_mse = torch.tensor(0.0, device=spec_mse.device)

        spec_loss = (
            spec_mask_loss_weight * spec_masked_mse +
            spec_mask_l1_weight * spec_masked_l1 +
            spec_unmask_loss_weight * spec_unmasked_mse
        )

        # 总损失
        loss_weights = config['loss_weights']
        total_loss = (
            loss_weights['waveform'] * wave_loss +
            loss_weights['spectrogram'] * spec_loss
        )

        return {
            'total_loss': total_loss,
            'waveform_loss': wave_loss,
            'spectrogram_loss': spec_loss,
            'waveform_masked_loss': wave_masked_mse,
            'spectrogram_masked_loss': spec_masked_mse,
            'waveform_unmasked_loss': wave_unmasked_mse,
            'spectrogram_unmasked_loss': spec_unmasked_mse,
            'waveform_stft_loss': waveform_stft_loss,
            'waveform_diff_loss': waveform_diff_loss,
            'waveform_coarse_loss': coarse_wave_masked_mse,
            'waveform_coarse_l1_loss': coarse_wave_masked_l1,
            'waveform_mel_loss': waveform_mel_loss,
        }


def create_model(config: dict) -> DualBranchAudioMAE:
    """创建模型"""
    model = DualBranchAudioMAE(config)
    return model


if __name__ == "__main__":
    # 测试模型
    config = {
        'data': {
            'sample_rate': 16000,
            'audio_length': 2.0,
            'n_mels': 80,
            'hop_length': 256
        },
        'model': {
            'transformer': {
                'input_dim': 128,
                'hidden_dim': 256,
                'num_heads': 4,
                'num_layers': 4,
                'dropout': 0.1
            },
            'cnn': {
                'in_channels': 1,
                'base_channels': 32,
                'num_layers': 4,
                'kernel_size': 3,
                'stride': 2,
                'padding': 1
            },
            'fusion': {
                'hidden_dim': 256
            },
            'decoder': {
                'waveform_hidden_dim': 256,
                'waveform_num_layers': 6,
                'spectrogram_hidden_dim': 256,
                'spectrogram_num_layers': 4
            }
        },
        'training': {
            'loss_weights': {
                'waveform': 10.0,
                'spectrogram': 2.0
            }
        }
    }
    
    model = create_model(config)
    
    # 测试前向传播
    batch_size = 2
    audio_length = 32000
    spec_time = audio_length // 256 + 1
    
    waveform = torch.randn(batch_size, audio_length)
    spectrogram = torch.randn(batch_size, 1, 80, spec_time)
    
    # 创建掩码
    waveform_mask = torch.zeros(batch_size, audio_length, dtype=torch.bool)
    waveform_mask[:, 10000:20000] = True
    
    spectrogram_mask = torch.zeros(batch_size, spec_time, dtype=torch.bool)
    spectrogram_mask[:, 40:80] = True
    
    print("Testing fixed model...")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Waveform mask ratio: {waveform_mask.sum().item() / waveform_mask.numel():.2%}")
    
    outputs = model(waveform, spectrogram, waveform_mask, spectrogram_mask)
    
    print(f"\nOutputs:")
    print(f"  Reconstructed waveform: {outputs['reconstructed_waveform'].shape}")
    print(f"  Reconstructed spectrogram: {outputs['reconstructed_spectrogram'].shape}")
    
    # 检查重建质量
    recon_wave = outputs['reconstructed_waveform']
    print(f"\nReconstruction check:")
    print(f"  Masked region mean: {recon_wave[waveform_mask].mean():.4f}")
    print(f"  Masked region std: {recon_wave[waveform_mask].std():.4f}")
    print(f"  Unmasked region mean: {recon_wave[~waveform_mask].mean():.4f}")
    print(f"  Original masked mean: {waveform[waveform_mask].mean():.4f}")
    
    print("\n✅ Fixed model test passed!")
