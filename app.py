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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        
        # 安装 MFA (如果未安装)
        if platform.system() != "Windows":
            setup_mfa_linux()
    else:
        logger.info("本地环境运行")


def setup_mfa_linux():
    """Linux 环境下安装和配置 MFA"""
    import shutil
    
    # 检查 mfa 是否已安装
    if shutil.which("mfa"):
        logger.info("MFA 已安装")
        return
    
    logger.info("正在安装 MFA...")
    
    try:
        # 尝试 pip 安装
        subprocess.run(
            [sys.executable, "-m", "pip", "install", 
             "montreal-forced-aligner", "--quiet"],
            check=True,
            capture_output=True
        )
        logger.info("MFA 安装完成")
        
        # 下载中文模型
        download_mfa_models()
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"MFA pip 安装失败: {e}")
        logger.info("尝试使用 conda 安装...")
        try:
            subprocess.run(
                ["conda", "install", "-c", "conda-forge", 
                 "montreal-forced-aligner", "-y"],
                check=True,
                capture_output=True
            )
            download_mfa_models()
        except Exception as e2:
            logger.error(f"MFA 安装失败: {e2}")


def download_mfa_models():
    """下载 MFA 预训练模型"""
    models = [
        ("acoustic", "mandarin_mfa"),
        ("dictionary", "mandarin_china_mfa"),
    ]
    
    for model_type, model_name in models:
        try:
            logger.info(f"下载 MFA 模型: {model_type}/{model_name}")
            subprocess.run(
                ["mfa", "model", "download", model_type, model_name],
                check=True,
                capture_output=True,
                timeout=300  # 5分钟超时
            )
        except Exception as e:
            logger.warning(f"模型下载失败 {model_name}: {e}")


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
