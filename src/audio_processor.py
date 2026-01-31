# -*- coding: utf-8 -*-
"""
音频处理模块
包含 Silero VAD 切片和 Whisper 转录功能
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器，整合VAD切片和Whisper转录"""
    
    def __init__(
        self,
        models_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        初始化音频处理器
        
        参数:
            models_dir: 模型目录
            progress_callback: 进度回调函数
        """
        self.models_dir = models_dir
        self.progress_callback = progress_callback
        self.vad_model = None
        self.whisper_pipe = None
    
    def _log(self, msg: str):
        """记录日志并回调"""
        logger.info(msg)
        if self.progress_callback:
            self.progress_callback(msg)
    
    def load_vad_model(self):
        """加载 Silero VAD 模型"""
        if self.vad_model is not None:
            return
        
        self._log("正在加载 Silero VAD 模型...")
        
        from src.silero_vad_downloader import ensure_vad_model
        import torch
        
        # 确保模型已下载
        model_path = ensure_vad_model(self.models_dir, self.progress_callback)
        
        # 加载模型
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        self.vad_utils = utils
        self._log("Silero VAD 模型加载完成")
    
    def load_whisper_model(self, model_name: str = "openai/whisper-small"):
        """
        加载 Whisper 模型
        
        参数:
            model_name: 模型名称
        """
        if self.whisper_pipe is not None:
            return
        
        self._log(f"正在加载 Whisper 模型: {model_name}...")
        
        from transformers import pipeline
        import torch
        
        cache_dir = os.path.join(self.models_dir, "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            model_kwargs={"cache_dir": cache_dir}
        )
        self._log("Whisper 模型加载完成")

    def vad_split(
        self,
        audio_path: str,
        output_dir: str,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        threshold: float = 0.5
    ) -> List[str]:
        """
        使用 VAD 对音频进行切片
        
        参数:
            audio_path: 输入音频路径
            output_dir: 输出目录
            min_speech_duration_ms: 最小语音时长(毫秒)
            min_silence_duration_ms: 最小静音时长(毫秒)
            threshold: VAD阈值
        
        返回:
            切片文件路径列表
        """
        import torch
        import torchaudio
        
        self.load_vad_model()
        
        basename = Path(audio_path).stem
        os.makedirs(output_dir, exist_ok=True)
        
        self._log(f"正在处理: {audio_path}")
        
        # 读取音频
        wav, sr = torchaudio.load(audio_path)
        
        # 转换为单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        
        # 重采样到16kHz (VAD要求)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav_16k = resampler(wav)
            sr_vad = 16000
        else:
            wav_16k = wav
            sr_vad = sr
        
        # 获取语音时间戳
        get_speech_timestamps = self.vad_utils[0]
        speech_timestamps = get_speech_timestamps(
            wav_16k,
            self.vad_model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=sr_vad
        )
        
        self._log(f"检测到 {len(speech_timestamps)} 个语音片段")
        
        # 切片并保存
        output_files = []
        for i, ts in enumerate(speech_timestamps):
            # 转换回原始采样率的索引
            start = int(ts['start'] * sr / sr_vad)
            end = int(ts['end'] * sr / sr_vad)
            
            segment = wav[start:end]
            
            output_path = os.path.join(output_dir, f"{basename}_{i:04d}.wav")
            torchaudio.save(output_path, segment.unsqueeze(0), sr)
            output_files.append(output_path)
        
        self._log(f"切片完成，共 {len(output_files)} 个文件")
        return output_files
    
    def transcribe(self, audio_path: str, language: str = "chinese") -> str:
        """
        使用 Whisper 转录音频
        
        参数:
            audio_path: 音频文件路径
            language: 语言
        
        返回:
            转录文本
        """
        if self.whisper_pipe is None:
            raise RuntimeError("Whisper 模型未加载")
        
        result = self.whisper_pipe(
            audio_path,
            generate_kwargs={"language": language}
        )
        return result["text"].strip()
    
    def generate_lab(self, audio_path: str, text: str) -> str:
        """
        生成 .lab 文件
        
        参数:
            audio_path: 音频文件路径
            text: 转录文本
        
        返回:
            lab文件路径
        """
        lab_path = os.path.splitext(audio_path)[0] + ".lab"
        with open(lab_path, "w", encoding="utf-8") as f:
            f.write(text)
        return lab_path

    def process_full_pipeline(
        self,
        input_path: str,
        output_dir: str,
        language: str = "chinese",
        whisper_model: str = "openai/whisper-small"
    ) -> Tuple[bool, str, List[str]]:
        """
        完整处理流程: VAD切片 → Whisper转录 → 生成.lab
        
        参数:
            input_path: 输入音频文件或目录
            output_dir: 输出目录
            language: 转录语言
            whisper_model: Whisper模型名称
        
        返回:
            (成功标志, 消息, 输出文件列表)
        """
        try:
            # 加载模型
            self.load_vad_model()
            self.load_whisper_model(whisper_model)
            
            # 收集输入文件
            input_files = []
            if os.path.isfile(input_path):
                input_files = [input_path]
            elif os.path.isdir(input_path):
                for f in os.listdir(input_path):
                    if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                        input_files.append(os.path.join(input_path, f))
            
            if not input_files:
                return False, "未找到音频文件", []
            
            self._log(f"找到 {len(input_files)} 个音频文件")
            
            # 创建输出目录
            slices_dir = os.path.join(output_dir, "slices")
            os.makedirs(slices_dir, exist_ok=True)
            
            all_output_files = []
            
            for idx, audio_file in enumerate(input_files):
                self._log(f"处理 [{idx+1}/{len(input_files)}]: {os.path.basename(audio_file)}")
                
                # VAD切片
                slice_files = self.vad_split(audio_file, slices_dir)
                
                # 转录每个切片
                for slice_file in slice_files:
                    self._log(f"转录: {os.path.basename(slice_file)}")
                    text = self.transcribe(slice_file, language)
                    
                    if text:
                        lab_path = self.generate_lab(slice_file, text)
                        self._log(f"生成: {os.path.basename(lab_path)} -> {text[:30]}...")
                        all_output_files.append(slice_file)
                    else:
                        self._log(f"跳过空转录: {os.path.basename(slice_file)}")
            
            return True, f"处理完成，共 {len(all_output_files)} 个切片", all_output_files
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            return False, str(e), []


def process_audio_pipeline(
    input_path: str,
    output_dir: str,
    models_dir: str,
    language: str = "chinese",
    whisper_model: str = "openai/whisper-small",
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str, List[str]]:
    """
    便捷函数：执行完整音频处理流程
    
    参数:
        input_path: 输入音频文件或目录
        output_dir: 输出目录
        models_dir: 模型目录
        language: 转录语言
        whisper_model: Whisper模型名称
        progress_callback: 进度回调
    
    返回:
        (成功标志, 消息, 输出文件列表)
    """
    processor = AudioProcessor(models_dir, progress_callback)
    return processor.process_full_pipeline(input_path, output_dir, language, whisper_model)
