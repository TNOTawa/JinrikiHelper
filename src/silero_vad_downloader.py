# -*- coding: utf-8 -*-
"""
Silero VAD 模型下载模块
支持自动下载 Silero VAD 模型到指定目录
"""

import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Silero VAD 模型配置
SILERO_VAD_CONFIG = {
    "repo": "snakers4/silero-vad",
    "model_name": "silero_vad",
    "version": "v5.1",
    "onnx_filename": "silero_vad.onnx",
    "jit_filename": "silero_vad.jit",
    "download_base": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data"
}


def _download_file(
    url: str,
    dest_path: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    下载文件
    
    参数:
        url: 下载地址
        dest_path: 保存路径
        progress_callback: 进度回调
    
    返回:
        是否成功
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    try:
        log(f"正在下载: {url}")
        
        # 创建目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 下载文件
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                log(f"文件大小: {total_size / 1024 / 1024:.2f} MB")
            
            # 分块下载
            block_size = 8192
            downloaded = 0
            
            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size and downloaded % (block_size * 100) == 0:
                        percent = downloaded / total_size * 100
                        log(f"下载进度: {percent:.1f}%")
        
        log(f"下载完成: {dest_path}")
        return True
        
    except urllib.error.HTTPError as e:
        log(f"HTTP 错误: {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        log(f"网络错误: {e.reason}")
        return False
    except Exception as e:
        log(f"下载失败: {e}")
        return False


def get_vad_model_path(models_dir: str) -> str:
    """
    获取 VAD 模型文件路径
    
    参数:
        models_dir: 模型根目录
    
    返回:
        ONNX 模型文件路径
    """
    return os.path.join(models_dir, "silero_vad", SILERO_VAD_CONFIG["onnx_filename"])


def is_vad_model_downloaded(models_dir: str) -> bool:
    """
    检查 VAD 模型是否已下载
    
    参数:
        models_dir: 模型根目录
    
    返回:
        是否已下载
    """
    model_path = get_vad_model_path(models_dir)
    return os.path.exists(model_path)


def download_silero_vad(
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    use_onnx: bool = True
) -> tuple[bool, str]:
    """
    下载 Silero VAD 模型
    
    参数:
        output_dir: 输出目录 (模型根目录)
        progress_callback: 进度回调
        use_onnx: 是否下载 ONNX 格式 (默认 True，否则下载 JIT 格式)
    
    返回:
        (成功标志, 文件路径或错误信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    # 确定文件名和 URL
    if use_onnx:
        filename = SILERO_VAD_CONFIG["onnx_filename"]
    else:
        filename = SILERO_VAD_CONFIG["jit_filename"]
    
    url = f"{SILERO_VAD_CONFIG['download_base']}/{filename}"
    vad_dir = os.path.join(output_dir, "silero_vad")
    dest_path = os.path.join(vad_dir, filename)
    
    # 检查是否已存在
    if os.path.exists(dest_path):
        log(f"Silero VAD 模型已存在: {dest_path}")
        return True, dest_path
    
    log("开始下载 Silero VAD 模型...")
    log(f"版本: {SILERO_VAD_CONFIG['version']}")
    log(f"格式: {'ONNX' if use_onnx else 'JIT'}")
    
    if _download_file(url, dest_path, progress_callback):
        log("Silero VAD 模型下载完成!")
        return True, dest_path
    else:
        return False, "Silero VAD 模型下载失败"


def ensure_vad_model(
    models_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    确保 VAD 模型已下载，如未下载则自动下载
    
    参数:
        models_dir: 模型根目录
        progress_callback: 进度回调
    
    返回:
        模型文件路径
    
    异常:
        RuntimeError: 下载失败时抛出
    """
    model_path = get_vad_model_path(models_dir)
    
    if os.path.exists(model_path):
        logger.info(f"Silero VAD 模型已就绪: {model_path}")
        return model_path
    
    success, result = download_silero_vad(models_dir, progress_callback)
    if success:
        return result
    else:
        raise RuntimeError(f"Silero VAD 模型下载失败: {result}")
