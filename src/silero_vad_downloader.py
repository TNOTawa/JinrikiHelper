# -*- coding: utf-8 -*-
"""
Silero VAD 模型下载模块
支持自动下载 Silero VAD 模型到指定目录
支持多镜像源，适配国内云环境
"""

import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

# Silero VAD 模型配置
SILERO_VAD_CONFIG = {
    "repo": "snakers4/silero-vad",
    "model_name": "silero_vad",
    "version": "v5.1",
    "onnx_filename": "silero_vad.onnx",
    "jit_filename": "silero_vad.jit",
}

# 下载镜像源列表（按优先级排序）
# 国内云环境优先使用 HuggingFace 镜像（魔搭创空间访问 HF 较快）
DOWNLOAD_MIRRORS = [
    # HuggingFace 镜像（国内云环境推荐）
    "https://huggingface.co/deepghs/silero-vad-onnx/resolve/main",
    # HuggingFace onnx-community 镜像
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx",
    # HuggingFace 镜像站（hf-mirror.com）
    "https://hf-mirror.com/deepghs/silero-vad-onnx/resolve/main",
    # ghproxy 镜像（GitHub 加速）
    "https://ghproxy.com/https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data",
    # mirror.ghproxy 镜像
    "https://mirror.ghproxy.com/https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data",
    # jsdelivr CDN（稳定但可能有延迟）
    "https://cdn.jsdelivr.net/gh/snakers4/silero-vad@master/src/silero_vad/data",
    # GitHub 原始地址（作为最后备选）
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data",
]

# HuggingFace 镜像的文件名映射（HF 上的文件名可能不同）
HF_FILENAME_MAP = {
    "silero_vad.onnx": ["silero_vad.onnx", "model.onnx"],
    "silero_vad.jit": ["silero_vad.jit"],
}


def _download_file_from_url(
    url: str,
    dest_path: str,
    timeout: int = 30,
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    从单个 URL 下载文件
    
    参数:
        url: 下载地址
        dest_path: 保存路径
        timeout: 超时时间（秒）
        progress_callback: 进度回调
    
    返回:
        是否成功
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    temp_path = dest_path + ".downloading"
    
    try:
        log(f"正在下载: {url}")
        
        # 创建目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 下载文件
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = response.headers.get("Content-Length")
            if total_size:
                total_size = int(total_size)
                log(f"文件大小: {total_size / 1024 / 1024:.2f} MB")
            
            # 分块下载
            block_size = 8192
            downloaded = 0
            
            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size and downloaded % (block_size * 100) == 0:
                        percent = downloaded / total_size * 100
                        log(f"下载进度: {percent:.1f}%")
        
        # 下载完成，重命名
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        
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
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def _download_file_with_mirrors(
    filename: str,
    dest_path: str,
    mirrors: List[str],
    progress_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    使用多镜像源下载文件，自动尝试下一个源
    
    参数:
        filename: 文件名
        dest_path: 保存路径
        mirrors: 镜像源列表
        progress_callback: 进度回调
    
    返回:
        是否成功
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    # 获取可能的文件名列表（用于 HuggingFace 镜像）
    possible_filenames = HF_FILENAME_MAP.get(filename, [filename])
    
    for i, base_url in enumerate(mirrors):
        # 提取域名用于日志显示
        try:
            domain = base_url.split('/')[2]
        except:
            domain = base_url[:30]
        
        log(f"尝试镜像源 {i + 1}/{len(mirrors)}: {domain}")
        
        # 镜像源使用较短超时，快速切换
        timeout = 30 if i < len(mirrors) - 1 else 120
        
        # 尝试不同的文件名（HuggingFace 镜像可能使用 model.onnx）
        for try_filename in possible_filenames:
            url = f"{base_url}/{try_filename}"
            if _download_file_from_url(url, dest_path, timeout, progress_callback):
                return True
        
        if i < len(mirrors) - 1:
            log("切换到下一个镜像源...")
    
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
    下载 Silero VAD 模型（支持多镜像源）
    
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
    
    # 确定文件名
    if use_onnx:
        filename = SILERO_VAD_CONFIG["onnx_filename"]
    else:
        filename = SILERO_VAD_CONFIG["jit_filename"]
    
    vad_dir = os.path.join(output_dir, "silero_vad")
    dest_path = os.path.join(vad_dir, filename)
    
    # 检查是否已存在
    if os.path.exists(dest_path):
        # 验证文件大小（ONNX 模型约 1.8MB）
        file_size = os.path.getsize(dest_path)
        if file_size > 1024 * 1024:  # > 1MB
            log(f"Silero VAD 模型已存在: {dest_path}")
            return True, dest_path
        else:
            log(f"模型文件异常 (大小: {file_size} bytes)，重新下载...")
            os.remove(dest_path)
    
    log("开始下载 Silero VAD 模型...")
    log(f"版本: {SILERO_VAD_CONFIG['version']}")
    log(f"格式: {'ONNX' if use_onnx else 'JIT'}")
    
    if _download_file_with_mirrors(filename, dest_path, DOWNLOAD_MIRRORS, progress_callback):
        # 验证下载的文件
        file_size = os.path.getsize(dest_path)
        if file_size > 1024 * 1024:
            log("Silero VAD 模型下载完成!")
            return True, dest_path
        else:
            log(f"下载的文件异常 (大小: {file_size} bytes)")
            os.remove(dest_path)
            return False, "下载的模型文件无效"
    else:
        return False, "所有镜像源均下载失败"


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
