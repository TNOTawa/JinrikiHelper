# -*- coding: utf-8 -*-
"""
音源制作流水线
将所有非GUI的业务逻辑集中管理
"""

import os
import glob
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流水线配置"""
    source_name: str  # 音源名称
    input_path: str  # 输入音频路径（文件或目录）
    output_base_dir: str  # 输出基础目录 (bank)
    models_dir: str  # 模型目录
    
    # 模型配置
    whisper_model: str = "openai/whisper-small"
    mfa_dict_path: Optional[str] = None
    mfa_model_path: Optional[str] = None
    
    # 处理参数
    language: str = "chinese"
    single_speaker: bool = True
    clean_mfa_cache: bool = True
    max_samples_per_word: int = 100
    
    @property
    def source_dir(self) -> str:
        """音源目录: bank/[音源名称]"""
        return os.path.join(self.output_base_dir, self.source_name)
    
    @property
    def slices_dir(self) -> str:
        """切片目录: bank/[音源名称]/slices"""
        return os.path.join(self.source_dir, "slices")
    
    @property
    def textgrid_dir(self) -> str:
        """TextGrid目录: bank/[音源名称]/textgrid"""
        return os.path.join(self.source_dir, "textgrid")
    
    @property
    def segments_dir(self) -> str:
        """分字片段临时目录（处理完成后可删除）"""
        return os.path.join(self.output_base_dir, ".temp_segments", self.source_name)
    
    @property
    def export_dir(self) -> str:
        """导出目录: export/[音源名称]/simple_export"""
        # 导出到项目根目录的 export 文件夹
        base = Path(self.output_base_dir).parent
        return os.path.join(base, "export", self.source_name, "simple_export")
    
    @property
    def meta_file(self) -> str:
        """元文件路径: bank/[音源名称]/meta.json"""
        return os.path.join(self.source_dir, "meta.json")


@dataclass
class VoiceBankMeta:
    """
    音源元信息
    
    存储制作音源时的设置和模型信息
    """
    # 基本信息
    source_name: str
    created_at: str  # ISO格式时间戳
    updated_at: str  # ISO格式时间戳
    
    # 模型信息
    whisper_model: str  # Whisper模型名称
    mfa_dict: str  # MFA字典文件名
    mfa_acoustic: str  # MFA声学模型文件名
    
    # 处理参数
    language: str  # 转录语言
    single_speaker: bool  # 单说话人模式
    
    # 统计信息
    slice_count: int = 0  # 切片数量
    textgrid_count: int = 0  # TextGrid文件数量
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "VoiceBankMeta":
        """从流水线配置创建元信息"""
        now = datetime.now().isoformat()
        
        # 提取模型文件名（不含路径）
        mfa_dict = ""
        if config.mfa_dict_path:
            mfa_dict = os.path.basename(config.mfa_dict_path)
        
        mfa_acoustic = ""
        if config.mfa_model_path:
            mfa_acoustic = os.path.basename(config.mfa_model_path)
        
        return cls(
            source_name=config.source_name,
            created_at=now,
            updated_at=now,
            whisper_model=config.whisper_model,
            mfa_dict=mfa_dict,
            mfa_acoustic=mfa_acoustic,
            language=config.language,
            single_speaker=config.single_speaker
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceBankMeta":
        """从字典创建"""
        return cls(**data)
    
    def save(self, path: str):
        """保存到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> Optional["VoiceBankMeta"]:
        """从文件加载"""
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"加载元文件失败: {e}")
            return None


class VoiceBankPipeline:
    """音源制作流水线"""
    
    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.progress_callback = progress_callback
        self._vad_model = None
        self._get_speech_timestamps = None
        self._whisper_model = None
        self._whisper_processor = None
        self._device = None
    
    def _log(self, msg: str):
        """记录日志"""
        logger.info(msg)
        if self.progress_callback:
            self.progress_callback(msg)
    
    def _ensure_dirs(self):
        """确保目录存在"""
        os.makedirs(self.config.source_dir, exist_ok=True)
        os.makedirs(self.config.slices_dir, exist_ok=True)
    
    def _save_meta(self, slice_count: int = 0):
        """
        保存元文件
        
        参数:
            slice_count: 切片数量
        """
        # 统计TextGrid文件数量
        textgrid_count = 0
        if os.path.exists(self.config.textgrid_dir):
            textgrid_count = len([
                f for f in os.listdir(self.config.textgrid_dir)
                if f.endswith('.TextGrid')
            ])
        
        # 检查是否已有元文件（更新而非覆盖）
        existing_meta = VoiceBankMeta.load(self.config.meta_file)
        
        if existing_meta:
            # 更新现有元文件
            existing_meta.updated_at = datetime.now().isoformat()
            existing_meta.whisper_model = self.config.whisper_model
            existing_meta.mfa_dict = os.path.basename(self.config.mfa_dict_path) if self.config.mfa_dict_path else ""
            existing_meta.mfa_acoustic = os.path.basename(self.config.mfa_model_path) if self.config.mfa_model_path else ""
            existing_meta.language = self.config.language
            existing_meta.single_speaker = self.config.single_speaker
            if slice_count > 0:
                existing_meta.slice_count = slice_count
            existing_meta.textgrid_count = textgrid_count
            meta = existing_meta
        else:
            # 创建新元文件
            meta = VoiceBankMeta.from_config(self.config)
            meta.slice_count = slice_count
            meta.textgrid_count = textgrid_count
        
        meta.save(self.config.meta_file)
        self._log(f"元文件已保存: {self.config.meta_file}")
    
    # ==================== 模型加载 ====================
    
    def _load_vad_model(self):
        """加载VAD模型"""
        if self._vad_model is not None:
            return
        
        self._log("正在加载 Silero VAD 模型...")
        from src.silero_vad_downloader import ensure_vad_model
        from silero_vad import load_silero_vad, get_speech_timestamps
        
        # 确保模型已下载
        model_path = ensure_vad_model(self.config.models_dir, self.progress_callback)
        
        # 使用 silero_vad 包加载本地 ONNX 模型
        self._vad_model = load_silero_vad(onnx=True)
        self._get_speech_timestamps = get_speech_timestamps
        self._log("VAD 模型加载完成")
    
    def _load_whisper_model(self):
        """加载Whisper模型"""
        if self._whisper_model is not None:
            return
        
        self._log(f"正在加载 Whisper 模型: {self.config.whisper_model}...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        
        cache_dir = os.path.join(self.config.models_dir, "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        self._whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model,
            cache_dir=cache_dir
        )
        self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.config.whisper_model,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 移动到GPU（如果可用）
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._whisper_model.to(self._device)
        
        self._log("Whisper 模型加载完成")
    
    # ==================== 步骤0: VAD切片 + 转录 ====================
    
    def step0_preprocess(self) -> Tuple[bool, str, List[str]]:
        """
        步骤0: VAD切片 + Whisper转录
        
        输入: 原始音频文件
        输出: bank/[音源名称]/slices/ 下的 .wav 和 .lab 文件
        """
        try:
            self._ensure_dirs()
            self._load_vad_model()
            self._load_whisper_model()
            
            # 收集输入文件
            input_files = self._collect_audio_files(self.config.input_path)
            if not input_files:
                return False, "未找到音频文件", []
            
            self._log(f"找到 {len(input_files)} 个音频文件")
            
            all_slices = []
            for idx, audio_file in enumerate(input_files):
                basename = Path(audio_file).stem
                self._log(f"处理 [{idx+1}/{len(input_files)}]: {basename}")
                
                # VAD切片
                slices = self._vad_split(audio_file, self.config.slices_dir, basename)
                
                # 转录每个切片
                for slice_path in slices:
                    text = self._transcribe(slice_path)
                    if text:
                        self._write_lab(slice_path, text)
                        all_slices.append(slice_path)
                        self._log(f"  {Path(slice_path).name} -> {text[:30]}...")
                    else:
                        self._log(f"  跳过空转录: {Path(slice_path).name}")
            
            # 保存元文件
            self._save_meta(slice_count=len(all_slices))
            
            return True, f"预处理完成，共 {len(all_slices)} 个切片", all_slices
            
        except Exception as e:
            logger.error(f"预处理失败: {e}", exc_info=True)
            return False, str(e), []
    
    def _collect_audio_files(self, path: str) -> List[str]:
        """收集音频文件"""
        extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        if os.path.isfile(path):
            return [path] if path.lower().endswith(extensions) else []
        elif os.path.isdir(path):
            return [
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(extensions)
            ]
        return []
    
    def _vad_split(self, audio_path: str, output_dir: str, prefix: str) -> List[str]:
        """
        VAD切片
        
        输出格式统一为: 16bit 44.1kHz 单声道 WAV
        """
        import torch
        import soundfile as sf
        import numpy as np
        
        # 标准输出格式
        TARGET_SR = 44100
        
        # 读取并转换为标准格式
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # 转换为单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 重采样到 44.1kHz（标准格式）
        if sr != TARGET_SR:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            audio = resampler(audio_tensor).numpy()
        
        # VAD 需要 16kHz，单独重采样用于检测
        import torchaudio
        audio_tensor = torch.from_numpy(audio).float()
        resampler_16k = torchaudio.transforms.Resample(TARGET_SR, 16000)
        wav_16k = resampler_16k(audio_tensor)
        
        # 获取语音时间戳（基于16kHz）
        timestamps = self._get_speech_timestamps(
            wav_16k, self._vad_model,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            sampling_rate=16000
        )
        
        self._log(f"  检测到 {len(timestamps)} 个语音片段")
        
        output_files = []
        for i, ts in enumerate(timestamps):
            # 将16kHz的时间戳转换为44.1kHz的采样点索引
            start = int(ts['start'] * TARGET_SR / 16000)
            end = int(ts['end'] * TARGET_SR / 16000)
            segment = audio[start:end]
            
            output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.wav")
            # 写入 16bit 44.1kHz 单声道 WAV
            sf.write(output_path, segment, TARGET_SR, subtype='PCM_16')
            output_files.append(output_path)
        
        return output_files
    
    def _transcribe(self, audio_path: str) -> str:
        """Whisper转录（输入已是44.1kHz，需转为16kHz）"""
        import soundfile as sf
        import numpy as np
        import torch
        import torchaudio
        
        # 读取音频（已是44.1kHz单声道）
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Whisper 需要 16kHz
        audio_tensor = torch.from_numpy(audio).float()
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_16k = resampler(audio_tensor).numpy()
        
        # 处理输入
        input_features = self._whisper_processor(
            audio_16k, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self._device)
        
        # 设置语言
        forced_decoder_ids = self._whisper_processor.get_decoder_prompt_ids(
            language=self.config.language, 
            task="transcribe"
        )
        
        # 生成
        with torch.no_grad():
            predicted_ids = self._whisper_model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids
            )
        
        # 解码
        transcription = self._whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    
    def _write_lab(self, audio_path: str, text: str):
        """写入.lab文件"""
        lab_path = os.path.splitext(audio_path)[0] + ".lab"
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    # ==================== 步骤1: MFA对齐 ====================
    
    def step1_mfa_align(self) -> Tuple[bool, str]:
        """
        步骤1: MFA语音对齐
        
        输入: bank/[音源名称]/slices/ 下的 .wav 和 .lab 文件
        输出: bank/[音源名称]/textgrid/ 下的 .TextGrid 文件
        
        注意: 直接使用中文文本，MFA字典为汉字到音素映射
        """
        try:
            os.makedirs(self.config.textgrid_dir, exist_ok=True)
            
            # 调用 MFA 对齐（直接使用中文文本，不转拼音）
            from src.mfa_runner import run_mfa_alignment
            
            success, message = run_mfa_alignment(
                corpus_dir=self.config.slices_dir,
                output_dir=self.config.textgrid_dir,
                dict_path=self.config.mfa_dict_path,
                model_path=self.config.mfa_model_path,
                single_speaker=self.config.single_speaker,
                clean=self.config.clean_mfa_cache,
                progress_callback=self.progress_callback
            )
            
            # 更新元文件（更新TextGrid数量）
            if success:
                self._save_meta()
            
            return success, message
            
        except Exception as e:
            logger.error(f"MFA对齐失败: {e}", exc_info=True)
            return False, str(e)
    
    # ==================== 制作流程（步骤0+1） ====================
    
    def run_make_pipeline(self) -> Tuple[bool, str]:
        """运行制作流水线（仅步骤0和步骤1）"""
        self._log("=" * 50)
        self._log(f"开始制作音源: {self.config.source_name}")
        self._log("=" * 50)
        
        # 步骤0
        self._log("\n【步骤0】音频预处理 (VAD切片 + Whisper转录)")
        success, msg, _ = self.step0_preprocess()
        if not success:
            return False, f"步骤0失败: {msg}"
        
        # 步骤1
        self._log("\n【步骤1】MFA语音对齐")
        success, msg = self.step1_mfa_align()
        if not success:
            return False, f"步骤1失败: {msg}"
        
        self._log("\n" + "=" * 50)
        self._log("✅ 音源制作完成!")
        self._log(f"输出目录: {self.config.source_dir}")
        self._log("提示: 请到「导出音源」页面进行导出")
        self._log("=" * 50)
        
        return True, "音源制作完成"


# ==================== 模型扫描工具 ====================

def scan_mfa_models(models_dir: str) -> Dict[str, List[str]]:
    """
    扫描MFA模型目录
    
    返回:
        {
            "acoustic": ["mandarin_mfa.zip", ...],
            "dictionary": ["mandarin_china_mfa.dict", ...]
        }
    """
    mfa_dir = os.path.join(models_dir, "mfa")
    result = {"acoustic": [], "dictionary": []}
    
    if not os.path.exists(mfa_dir):
        return result
    
    for f in os.listdir(mfa_dir):
        if f.endswith('.zip'):
            result["acoustic"].append(f)
        elif f.endswith('.dict') or f.endswith('.txt'):
            result["dictionary"].append(f)
    
    return result


def scan_whisper_models(models_dir: str) -> List[str]:
    """
    扫描已下载的Whisper模型
    
    返回模型名称列表
    """
    whisper_dir = os.path.join(models_dir, "whisper")
    models = []
    
    if not os.path.exists(whisper_dir):
        return models
    
    # 检查 HuggingFace 缓存目录结构
    for item in os.listdir(whisper_dir):
        if item.startswith("models--"):
            # 格式: models--openai--whisper-small
            parts = item.replace("models--", "").split("--")
            if len(parts) >= 2:
                models.append("/".join(parts))
    
    return models


def load_voice_bank_meta(bank_dir: str, source_name: str) -> Optional[VoiceBankMeta]:
    """
    加载音源元信息
    
    参数:
        bank_dir: bank目录路径
        source_name: 音源名称
    
    返回:
        VoiceBankMeta对象，如果不存在则返回None
    """
    meta_path = os.path.join(bank_dir, source_name, "meta.json")
    return VoiceBankMeta.load(meta_path)


def list_voice_banks_with_meta(bank_dir: str) -> List[Dict]:
    """
    列出所有音源及其元信息
    
    参数:
        bank_dir: bank目录路径
    
    返回:
        包含音源信息的字典列表
    """
    result = []
    
    if not os.path.exists(bank_dir):
        return result
    
    for name in os.listdir(bank_dir):
        source_dir = os.path.join(bank_dir, name)
        if not os.path.isdir(source_dir):
            continue
        
        # 检查是否为有效音源目录（包含slices子目录）
        slices_dir = os.path.join(source_dir, "slices")
        if not os.path.exists(slices_dir):
            continue
        
        info = {"name": name, "meta": None}
        
        # 尝试加载元信息
        meta = load_voice_bank_meta(bank_dir, name)
        if meta:
            info["meta"] = meta.to_dict()
        
        result.append(info)
    
    return result
