# -*- coding: utf-8 -*-
"""
音源质量评分模块

提供多维度的音频质量评估，用于筛选最佳样本
支持时长、音量稳定性、音高稳定性三个评估维度
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def duration_score(duration: float) -> float:
    """
    时长评分：适中时长得分最高
    
    参数:
        duration: 音频时长（秒）
    
    返回:
        0~1 的分数
    
    评分逻辑:
        - 过短（<0.2s）：发音不完整，低分
        - 最佳范围（0.3~0.8s）：满分
        - 过长（>1.0s）：可能包含多字或拖音，递减
    """
    if duration < 0.2:
        return duration / 0.2 * 0.5  # 0~0.5分
    elif duration <= 0.8:
        return 1.0  # 满分
    elif duration <= 1.2:
        return 1.0 - (duration - 0.8) / 0.4 * 0.3  # 0.7~1.0分
    else:
        return max(0.3, 0.7 - (duration - 1.2) * 0.2)  # 递减，最低0.3


def rms_variance_score(audio: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """
    音量稳定性评分：RMS 方差越小越好
    
    参数:
        audio: 音频数据（numpy 数组）
        sr: 采样率
        frame_ms: 帧长度（毫秒）
    
    返回:
        0~1 的分数
    
    计算步骤:
        1. 将音频分帧
        2. 计算每帧的 RMS 能量
        3. 计算 RMS 序列的方差
        4. 归一化到 0~1 分数
    """
    frame_size = int(sr * frame_ms / 1000)
    if frame_size <= 0:
        return 0.5
    
    frames = len(audio) // frame_size
    if frames < 2:
        return 0.5  # 太短无法评估
    
    rms_values = []
    for i in range(frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        rms_values.append(rms)
    
    if len(rms_values) < 2:
        return 0.5
    
    # 归一化 RMS 值（避免绝对值影响）
    rms_array = np.array(rms_values)
    mean_rms = np.mean(rms_array)
    if mean_rms > 0:
        rms_normalized = rms_array / mean_rms
        variance = np.var(rms_normalized)
    else:
        variance = 0
    
    # 归一化：方差越小分数越高
    # 经验阈值：方差 < 0.01 为优秀，> 0.5 为较差
    score = max(0, 1.0 - variance * 2)
    return min(1.0, score)


def f0_variance_score(audio: np.ndarray, sr: int) -> float:
    """
    音高稳定性评分：F0 方差越小越好
    
    参数:
        audio: 音频数据（numpy 数组）
        sr: 采样率
    
    返回:
        0~1 的分数
    
    计算步骤:
        1. 使用 librosa.pyin 提取 F0
        2. 过滤无声帧（F0=NaN）
        3. 转换为音分计算方差
        4. 归一化到 0~1 分数
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa 未安装，无法计算 F0 方差")
        return 0.5
    
    try:
        # 提取 F0（使用 pyin 算法）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio.astype(np.float32),
            fmin=librosa.note_to_hz('C2'),   # ~65Hz
            fmax=librosa.note_to_hz('C6'),   # ~1047Hz
            sr=sr
        )
        
        # 过滤无效值
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) < 3:
            return 0.5  # 无法评估
        
        # 转换为音分（cents）计算方差，避免频率绝对值影响
        # cents = 1200 * log2(f / f_ref)
        median_f0 = np.median(valid_f0)
        if median_f0 <= 0:
            return 0.5
        
        f0_cents = 1200 * np.log2(valid_f0 / median_f0)
        variance = np.var(f0_cents)
        
        # 归一化：方差 < 100 cents² 为优秀，> 10000 cents² 为较差
        # 100 cents ≈ 1个半音
        score = max(0, 1.0 - variance / 10000)
        return min(1.0, score)
        
    except Exception as e:
        logger.warning(f"F0 计算失败: {e}")
        return 0.5



class QualityScorer:
    """
    音频质量评分器
    
    支持多维度评估和加权综合评分
    """
    
    # 默认权重
    DEFAULT_WEIGHTS = {
        "duration": 0.3,
        "rms": 0.3,
        "f0": 0.4
    }
    
    def __init__(
        self,
        enabled_metrics: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化评分器
        
        参数:
            enabled_metrics: 启用的评分维度，如 ["duration", "rms", "f0"]
            weights: 各维度权重
        """
        self.enabled_metrics = enabled_metrics or ["duration"]
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def score(
        self,
        audio: np.ndarray,
        sr: int,
        duration: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算音频质量分数
        
        参数:
            audio: 音频数据
            sr: 采样率
            duration: 音频时长（秒），如不提供则自动计算
        
        返回:
            包含各维度分数和综合分数的字典
        """
        if duration is None:
            duration = len(audio) / sr
        
        scores = {}
        
        if "duration" in self.enabled_metrics:
            scores["duration"] = duration_score(duration)
        
        if "rms" in self.enabled_metrics:
            scores["rms"] = rms_variance_score(audio, sr)
        
        if "f0" in self.enabled_metrics:
            scores["f0"] = f0_variance_score(audio, sr)
        
        # 计算加权综合分数
        if scores:
            total_weight = sum(self.weights.get(k, 0) for k in scores.keys())
            if total_weight > 0:
                combined = sum(
                    scores[k] * self.weights.get(k, 0) 
                    for k in scores.keys()
                ) / total_weight
            else:
                combined = sum(scores.values()) / len(scores)
            scores["combined"] = combined
        else:
            scores["combined"] = 0.5
        
        return scores
    
    def score_from_file(self, wav_path: str) -> Dict[str, float]:
        """
        从文件计算质量分数
        
        参数:
            wav_path: 音频文件路径
        
        返回:
            包含各维度分数和综合分数的字典
        """
        try:
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            
            # 转换为单声道
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            return self.score(audio, sr)
            
        except Exception as e:
            logger.error(f"读取音频文件失败 {wav_path}: {e}")
            return {"combined": 0.5}


def calculate_quality_score(
    audio: np.ndarray,
    sr: int,
    weights: Optional[Dict[str, float]] = None,
    enabled_metrics: Optional[List[str]] = None
) -> float:
    """
    便捷函数：计算综合质量评分
    
    参数:
        audio: 音频数据
        sr: 采样率
        weights: 各维度权重
        enabled_metrics: 启用的评分维度
    
    返回:
        0~1 的综合分数
    """
    scorer = QualityScorer(enabled_metrics=enabled_metrics, weights=weights)
    scores = scorer.score(audio, sr)
    return scores.get("combined", 0.5)
