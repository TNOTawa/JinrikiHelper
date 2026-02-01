# -*- coding: utf-8 -*-
"""
导出插件基类

定义插件接口和配置选项类型
"""

import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """配置选项类型"""
    TEXT = "text"           # 文本输入框
    NUMBER = "number"       # 数字输入框
    SWITCH = "switch"       # 开关
    LABEL = "label"         # 纯文本标签（不可编辑）
    FILE = "file"           # 文件选择
    FOLDER = "folder"       # 文件夹选择
    COMBO = "combo"         # 下拉选择框
    MULTI_SELECT = "multi_select"  # 多选框


@dataclass
class PluginOption:
    """插件配置选项"""
    key: str                          # 选项键名
    label: str                        # 显示标签
    option_type: OptionType           # 选项类型
    default: Any = None               # 默认值
    description: str = ""             # 描述说明
    choices: List[Any] = field(default_factory=list)  # 下拉/多选选项
    min_value: Optional[float] = None # 最小值（仅NUMBER类型）
    max_value: Optional[float] = None # 最大值（仅NUMBER类型）
    step: Optional[float] = None      # 步进值（仅NUMBER类型）
    file_types: List[Tuple[str, str]] = field(default_factory=list)  # 文件类型过滤
    visible_when: Optional[Dict[str, Any]] = None  # 条件显示规则


class ExportPlugin(ABC):
    """导出插件基类"""
    
    # 插件元信息（子类必须覆盖）
    name: str = "未命名插件"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    
    def __init__(self):
        self._options: Dict[str, Any] = {}
        self._progress_callback: Optional[Callable[[str], None]] = None
        # 初始化默认值
        for opt in self.get_options():
            self._options[opt.key] = opt.default
    
    @abstractmethod
    def get_options(self) -> List[PluginOption]:
        """
        获取插件配置选项列表
        
        返回:
            配置选项列表
        """
        pass
    
    @abstractmethod
    def export(
        self,
        source_name: str,
        bank_dir: str,
        options: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        执行导出
        
        参数:
            source_name: 音源名称
            bank_dir: bank目录路径
            options: 用户配置的选项值
        
        返回:
            (成功标志, 消息)
        """
        pass
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """设置进度回调"""
        self._progress_callback = callback
    
    def _log(self, msg: str):
        """记录日志"""
        logger.info(msg)
        if self._progress_callback:
            self._progress_callback(msg)
    
    def get_option_value(self, key: str) -> Any:
        """获取选项值"""
        return self._options.get(key)
    
    def set_option_value(self, key: str, value: Any):
        """设置选项值"""
        self._options[key] = value
    
    def reset_to_defaults(self):
        """重置为默认值"""
        for opt in self.get_options():
            self._options[opt.key] = opt.default
    
    def get_export_dir(self, bank_dir: str, source_name: str, subdir: str) -> str:
        """
        获取导出目录路径
        
        参数:
            bank_dir: bank目录
            source_name: 音源名称
            subdir: 子目录名
        
        返回:
            export/[音源名称]/[subdir]/ 路径
        """
        from pathlib import Path
        base = Path(bank_dir).parent
        return os.path.join(base, "export", source_name, subdir)
    
    def get_source_paths(self, bank_dir: str, source_name: str) -> Dict[str, str]:
        """
        获取音源相关路径
        
        返回:
            {
                "source_dir": 音源目录,
                "slices_dir": 切片目录,
                "textgrid_dir": TextGrid目录
            }
        """
        source_dir = os.path.join(bank_dir, source_name)
        return {
            "source_dir": source_dir,
            "slices_dir": os.path.join(source_dir, "slices"),
            "textgrid_dir": os.path.join(source_dir, "textgrid")
        }
    
    def get_quality_scorer(
        self,
        enabled_metrics: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        获取质量评分器实例
        
        参数:
            enabled_metrics: 启用的评分维度，如 ["duration", "rms", "f0"]
            weights: 各维度权重
        
        返回:
            QualityScorer 实例
        """
        from ..quality_scorer import QualityScorer
        return QualityScorer(enabled_metrics=enabled_metrics, weights=weights)
    
    def score_audio_quality(
        self,
        wav_path: str,
        enabled_metrics: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        评估音频文件质量
        
        参数:
            wav_path: 音频文件路径
            enabled_metrics: 启用的评分维度
            weights: 各维度权重
        
        返回:
            包含各维度分数和综合分数的字典
        """
        scorer = self.get_quality_scorer(enabled_metrics, weights)
        return scorer.score_from_file(wav_path)
