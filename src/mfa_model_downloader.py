# -*- coding: utf-8 -*-
"""
MFA 模型下载模块
支持下载中文和日文的声学模型及字典
"""

import os
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# 模型下载基础 URL
GITHUB_RELEASE_BASE = "https://github.com/MontrealCorpusTools/mfa-models/releases/download"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main"

# 支持的语言配置
# 格式: {语言代码: {名称, 声学模型信息, 字典信息}}
LANGUAGE_MODELS = {
    "mandarin": {
        "name": "中文 (普通话)",
        "acoustic": {
            "tag": "acoustic-mandarin_mfa-v3.0.0",
            "filename": "mandarin_mfa.zip",
            "description": "Mandarin MFA acoustic model v3.0.0"
        },
        "dictionary": {
            # 字典从 releases 下载，tag 格式: dictionary-{name}-v{version}
            "tag": "dictionary-mandarin_china_mfa-v3.0.0",
            "filename": "mandarin_china_mfa.dict",
            "description": "Mandarin (China) MFA dictionary v3.0.0"
        }
    },
    "japanese": {
        "name": "日文",
        "acoustic": {
            "tag": "acoustic-japanese_mfa-v3.0.0",
            "filename": "japanese_mfa.zip",
            "description": "Japanese MFA acoustic model v3.0.0"
        },
        "dictionary": {
            "tag": "dictionary-japanese_mfa-v3.0.0",
            "filename": "japanese_mfa.dict",
            "description": "Japanese MFA dictionary v3.0.0"
        }
    }
}


def get_available_languages() -> dict:
    """获取可用的语言列表"""
    return {k: v["name"] for k, v in LANGUAGE_MODELS.items()}


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
                log(f"文件大小: {total_size / 1024 / 1024:.1f} MB")
            
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


def download_acoustic_model(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    下载声学模型
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
    
    返回:
        (成功标志, 文件路径或错误信息)
    """
    if language not in LANGUAGE_MODELS:
        return False, f"不支持的语言: {language}"
    
    config = LANGUAGE_MODELS[language]["acoustic"]
    url = f"{GITHUB_RELEASE_BASE}/{config['tag']}/{config['filename']}"
    dest_path = os.path.join(output_dir, config["filename"])
    
    if os.path.exists(dest_path):
        if progress_callback:
            progress_callback(f"声学模型已存在: {dest_path}")
        return True, dest_path
    
    if _download_file(url, dest_path, progress_callback):
        return True, dest_path
    else:
        return False, "声学模型下载失败"


def download_dictionary(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    下载字典文件
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
    
    返回:
        (成功标志, 文件路径或错误信息)
    """
    if language not in LANGUAGE_MODELS:
        return False, f"不支持的语言: {language}"
    
    config = LANGUAGE_MODELS[language]["dictionary"]
    # 字典文件从 releases 下载
    url = f"{GITHUB_RELEASE_BASE}/{config['tag']}/{config['filename']}"
    dest_path = os.path.join(output_dir, config["filename"])
    
    if os.path.exists(dest_path):
        if progress_callback:
            progress_callback(f"字典文件已存在: {dest_path}")
        return True, dest_path
    
    if _download_file(url, dest_path, progress_callback):
        return True, dest_path
    else:
        return False, "字典文件下载失败"


def download_language_models(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str, str]:
    """
    下载指定语言的声学模型和字典
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
    
    返回:
        (成功标志, 声学模型路径, 字典路径)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if language not in LANGUAGE_MODELS:
        return False, "", f"不支持的语言: {language}"
    
    lang_name = LANGUAGE_MODELS[language]["name"]
    log(f"开始下载 {lang_name} 模型...")
    
    # 下载声学模型
    log("=" * 40)
    log("下载声学模型...")
    success, acoustic_path = download_acoustic_model(language, output_dir, progress_callback)
    if not success:
        return False, "", acoustic_path
    
    # 下载字典
    log("=" * 40)
    log("下载字典文件...")
    success, dict_path = download_dictionary(language, output_dir, progress_callback)
    if not success:
        return False, acoustic_path, dict_path
    
    log("=" * 40)
    log(f"{lang_name} 模型下载完成!")
    return True, acoustic_path, dict_path
