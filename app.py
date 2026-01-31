# -*- coding: utf-8 -*-
"""
人力V助手 (JinrikiHelper) - 云端部署入口
适用于 Hugging Face Spaces / 魔塔社区
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).parent.absolute()
MODELS_DIR = BASE_DIR / "models"
MFA_DIR = MODELS_DIR / "mfa"


def setup_environment():
    """初始化云端环境"""
    
    # 检测运行环境
    is_cloud = any([
        os.environ.get("SPACE_ID"),           # Hugging Face Spaces
        os.environ.get("MODELSCOPE_SPACE"),   # 魔塔社区
        os.environ.get("GRADIO_SERVER_NAME"), # 通用 Gradio 云端
    ])
    
    if is_cloud:
        logger.info("检测到云端环境，正在初始化...")
        
        # 设置临时目录
        os.environ.setdefault("TMPDIR", "/tmp")
        
        # 安装 MFA (如果未安装，仅 Linux)
        if platform.system() != "Windows":
            setup_mfa_linux()
        
        # 下载所有必需模型
        download_all_models()
    else:
        logger.info("本地环境运行")


def setup_mfa_linux():
    """Linux 环境下安装 MFA"""
    import shutil
    
    if shutil.which("mfa"):
        logger.info("MFA 已安装")
        return
    
    logger.info("正在安装 MFA...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", 
             "montreal-forced-aligner", "--quiet"],
            check=True,
            capture_output=True
        )
        logger.info("MFA 安装完成")
    except subprocess.CalledProcessError as e:
        logger.warning(f"MFA pip 安装失败: {e}")
        try:
            subprocess.run(
                ["conda", "install", "-c", "conda-forge", 
                 "montreal-forced-aligner", "-y"],
                check=True,
                capture_output=True
            )
            logger.info("MFA conda 安装完成")
        except Exception as e2:
            logger.error(f"MFA 安装失败: {e2}")


def download_all_models():
    """下载所有必需的模型"""
    logger.info("=" * 50)
    logger.info("开始下载模型...")
    logger.info("=" * 50)
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MFA_DIR, exist_ok=True)
    
    # 1. 下载 Silero VAD
    download_silero_vad_model()
    
    # 2. 下载 Whisper 模型
    download_whisper_models()
    
    # 3. 下载 MFA 模型（中文和日语）
    download_mfa_models_all()
    
    logger.info("=" * 50)
    logger.info("所有模型下载完成")
    logger.info("=" * 50)


def download_silero_vad_model():
    """下载 Silero VAD 模型"""
    logger.info("\n【下载 Silero VAD 模型】")
    
    try:
        from src.silero_vad_downloader import download_silero_vad, is_vad_model_downloaded
        
        if is_vad_model_downloaded(str(MODELS_DIR)):
            logger.info("Silero VAD 模型已存在，跳过下载")
            return
        
        success, result = download_silero_vad(str(MODELS_DIR), logger.info)
        if success:
            logger.info(f"Silero VAD 下载成功: {result}")
        else:
            logger.warning(f"Silero VAD 下载失败: {result}")
    except Exception as e:
        logger.error(f"Silero VAD 下载异常: {e}")


def download_whisper_models():
    """下载 Whisper 模型 (small 和 medium)"""
    logger.info("\n【下载 Whisper 模型】")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        
        cache_dir = str(MODELS_DIR / "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        models = ["openai/whisper-small", "openai/whisper-medium"]
        
        for model_name in models:
            logger.info(f"下载 {model_name}...")
            try:
                # 检查是否已下载
                model_cache_name = model_name.replace("/", "--")
                model_cache_path = Path(cache_dir) / f"models--{model_cache_name}"
                
                if model_cache_path.exists():
                    logger.info(f"{model_name} 已存在，跳过下载")
                    continue
                
                # 下载 processor 和 model
                _ = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
                _ = WhisperForConditionalGeneration.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info(f"{model_name} 下载完成")
            except Exception as e:
                logger.warning(f"{model_name} 下载失败: {e}")
                
    except Exception as e:
        logger.error(f"Whisper 模型下载异常: {e}")


def download_mfa_models_all():
    """下载 MFA 中文和日语模型"""
    logger.info("\n【下载 MFA 模型】")
    
    try:
        from src.mfa_model_downloader import download_language_models
        
        languages = ["mandarin", "japanese"]
        
        for lang in languages:
            logger.info(f"\n下载 {lang} 模型...")
            try:
                success, acoustic_path, dict_path = download_language_models(
                    lang, str(MFA_DIR), logger.info
                )
                if success:
                    logger.info(f"{lang} 模型下载完成")
                else:
                    logger.warning(f"{lang} 模型下载失败")
            except Exception as e:
                logger.warning(f"{lang} 模型下载异常: {e}")
                
    except Exception as e:
        logger.error(f"MFA 模型下载异常: {e}")


def main():
    """主入口"""
    setup_environment()
    
    # 导入并启动云端 GUI
    from src.gui_cloud import create_cloud_ui
    
    app = create_cloud_ui()
    
    # 云端配置
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
