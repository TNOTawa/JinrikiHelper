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

# 云端持久化模型目录（魔搭创空间 /home/studio_service/ 是持久化的）
PERSISTENT_MODELS_DIR = Path("/home/studio_service/models")
# 本地模型目录
LOCAL_MODELS_DIR = BASE_DIR / "models"

# 根据环境选择模型目录
def get_models_dir():
    """获取模型目录，云端使用持久化路径"""
    if PERSISTENT_MODELS_DIR.parent.exists() and not LOCAL_MODELS_DIR.is_symlink():
        # 魔搭创空间环境，使用持久化目录
        PERSISTENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # 如果本地 models 目录存在且不是符号链接，先迁移已有模型
        if LOCAL_MODELS_DIR.exists() and LOCAL_MODELS_DIR.is_dir():
            import shutil
            for item in LOCAL_MODELS_DIR.iterdir():
                dest = PERSISTENT_MODELS_DIR / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(LOCAL_MODELS_DIR, ignore_errors=True)
        # 创建符号链接
        if not LOCAL_MODELS_DIR.exists():
            LOCAL_MODELS_DIR.symlink_to(PERSISTENT_MODELS_DIR)
        return PERSISTENT_MODELS_DIR
    return LOCAL_MODELS_DIR

MODELS_DIR = None  # 延迟初始化
MFA_DIR = None


def setup_environment():
    """初始化云端环境"""
    global MODELS_DIR, MFA_DIR
    
    # 初始化模型目录（可能创建符号链接）
    MODELS_DIR = get_models_dir()
    MFA_DIR = MODELS_DIR / "mfa"
    
    # 检测运行环境
    is_cloud = any([
        os.environ.get("SPACE_ID"),           # Hugging Face Spaces
        os.environ.get("MODELSCOPE_SPACE"),   # 魔塔社区
        os.environ.get("GRADIO_SERVER_NAME"), # 通用 Gradio 云端
        Path("/home/studio_service").exists(), # 魔搭创空间特征目录
    ])
    
    # 魔搭创空间无法访问 HuggingFace，使用镜像
    if is_cloud and Path("/home/studio_service").exists():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("已设置 HuggingFace 镜像: hf-mirror.com")
    
    # Linux 环境下始终尝试安装 MFA（无论是否云端）
    if platform.system() != "Windows":
        logger.info("Linux 环境，检查并安装 MFA...")
        setup_mfa_linux()
    
    if is_cloud:
        logger.info("检测到云端环境，正在初始化...")
        
        # 设置临时目录
        os.environ.setdefault("TMPDIR", "/tmp")
        
        # 下载所有必需模型
        download_all_models()
    else:
        logger.info("本地环境运行")


def setup_mfa_linux():
    """Linux 环境下安装 MFA（使用 micromamba）"""
    import shutil
    
    def verify_mfa_working():
        """验证 MFA 是否能正常工作（包括 kalpy 依赖）"""
        try:
            result = subprocess.run(
                ["mfa", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    # 检查 mfa 是否已可用
    if shutil.which("mfa") and verify_mfa_working():
        logger.info("MFA 已安装且工作正常")
        return
    
    logger.info("MFA 不可用，尝试使用 micromamba 安装...")
    
    # micromamba 安装路径
    mamba_root = Path("/tmp/micromamba")
    mamba_bin = mamba_root / "bin" / "micromamba"
    mfa_env = mamba_root / "envs" / "mfa"
    
    try:
        # 1. 安装 micromamba（如果不存在）
        if not mamba_bin.exists():
            logger.info("下载 micromamba...")
            mamba_root.mkdir(parents=True, exist_ok=True)
            
            # 下载并安装 micromamba
            subprocess.run([
                "bash", "-c",
                f'curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C {mamba_root} bin/micromamba'
            ], check=True, capture_output=True, timeout=120)
            logger.info("micromamba 下载完成")
        
        # 2. 使用 micromamba 创建环境并安装 MFA
        mfa_bin_path = mfa_env / "bin" / "mfa"
        mfa_bin_dir = mfa_env / "bin"
        need_install = not mfa_bin_path.exists()
        
        # 如果 mfa 存在，先将其加入 PATH 再验证
        if mfa_bin_path.exists():
            # 临时加入 PATH 以便验证
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{mfa_bin_dir}:{old_path}"
            
            if not verify_mfa_working():
                logger.info("检测到损坏的 MFA 环境，删除重建...")
                os.environ["PATH"] = old_path  # 恢复 PATH
                import shutil as sh
                sh.rmtree(mfa_env, ignore_errors=True)
                need_install = True
            else:
                logger.info("MFA 环境验证通过，无需重新安装")
        
        if need_install:
            logger.info("使用 micromamba 安装 MFA...")
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = str(mamba_root)
            
            # 创建环境并安装 MFA（指定 Python 3.11）
            subprocess.run([
                str(mamba_bin), "create", "-n", "mfa",
                "-c", "conda-forge",
                "python=3.11",
                "montreal-forced-aligner",
                "-y"
            ], env=env, check=True, capture_output=True, text=True, timeout=600)
            
            # 更新确保使用 CPU 版本的 kaldi
            subprocess.run([
                str(mamba_bin), "install", "-n", "mfa",
                "-c", "conda-forge",
                "kalpy", "kaldi=*=cpu*",
                "-y"
            ], env=env, capture_output=True, text=True, timeout=300)
            
            logger.info("MFA 安装完成")
        
        # 3. 安装中文/日语分词依赖（无论新装还是已有环境都需要检查）
        pip_path = mfa_env / "bin" / "pip"
        python_path = mfa_env / "bin" / "python"
        
        # 设置 pkuseg 模型目录到持久化路径（避免每次重启重新下载）
        pkuseg_home = PERSISTENT_MODELS_DIR / "pkuseg" if PERSISTENT_MODELS_DIR.parent.exists() else Path("/root/.pkuseg")
        pkuseg_home.mkdir(parents=True, exist_ok=True)
        os.environ["PKUSEG_HOME"] = str(pkuseg_home)
        
        if pip_path.exists():
            # 检查是否已安装分词依赖
            pkuseg_path = mfa_env / "lib" / "python3.11" / "site-packages" / "spacy_pkuseg"
            if not pkuseg_path.exists():
                logger.info("安装中文/日语分词依赖...")
                subprocess.run([
                    str(pip_path), "install",
                    "spacy-pkuseg", "dragonmapper", "hanziconv",  # 中文
                    "sudachipy", "sudachidict_core"  # 日语
                ], capture_output=True, text=True, timeout=300)
                logger.info("分词依赖安装完成")
            else:
                logger.info("分词依赖已存在")
            
            # 预下载 pkuseg 模型（避免运行时从 GitHub 下载超时）
            # 重要：pkuseg 使用 torch.hub.download_url_to_file 检查 zip 文件是否存在
            # 必须保留 zip 文件，否则 pkuseg 会尝试重新下载
            pkuseg_model_zip = pkuseg_home / "spacy_ontonotes.zip"
            postag_model_zip = pkuseg_home / "postag.zip"
            
            if (not pkuseg_model_zip.exists() or not postag_model_zip.exists()) and python_path.exists():
                logger.info(f"预下载 pkuseg 中文分词模型到 {pkuseg_home}...")
                
                import urllib.request
                import zipfile
                
                # 手动下载模型文件（使用 GitHub 镜像）
                # 注意：必须保留 zip 文件，pkuseg 会检查 zip 是否存在
                models_to_download = [
                    {
                        "name": "spacy_ontonotes",
                        "urls": [
                            "https://ghfast.top/https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
                            "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
                        ],
                        "zip_path": pkuseg_model_zip,
                    },
                    {
                        "name": "postag",
                        "urls": [
                            "https://ghfast.top/https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
                            "https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip",
                        ],
                        "zip_path": postag_model_zip,
                    },
                ]
                
                for model in models_to_download:
                    zip_path = model["zip_path"]
                    if zip_path.exists():
                        logger.info(f"{model['name']}.zip 已存在")
                        continue
                    
                    downloaded = False
                    for url in model["urls"]:
                        try:
                            logger.info(f"下载 {model['name']} 从 {url[:60]}...")
                            urllib.request.urlretrieve(url, zip_path)
                            
                            # 验证 zip 文件有效性
                            with zipfile.ZipFile(zip_path, 'r') as zf:
                                # 只验证，不解压（pkuseg 会自己解压）
                                if zf.testzip() is None:
                                    logger.info(f"{model['name']}.zip 下载完成并验证通过")
                                    downloaded = True
                                    break
                                else:
                                    logger.warning(f"{model['name']}.zip 文件损坏，尝试其他源")
                                    zip_path.unlink()
                        except Exception as e:
                            logger.warning(f"从 {url[:40]}... 下载失败: {e}")
                            if zip_path.exists():
                                zip_path.unlink()
                    
                    if not downloaded:
                        logger.warning(f"{model['name']} 所有下载源均失败")
                
                # 验证模型是否可用
                if pkuseg_model_zip.exists() and postag_model_zip.exists():
                    logger.info("pkuseg 模型 zip 文件下载完成")
                else:
                    logger.warning("pkuseg 模型下载不完整，MFA 中文对齐可能不可用")
            else:
                logger.info(f"pkuseg 模型已存在: {pkuseg_home}")
        
        # 4. 确保 MFA 环境的 bin 目录在 PATH 中
        if mfa_bin_dir.exists() and str(mfa_bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{mfa_bin_dir}:{os.environ.get('PATH', '')}"
            logger.info(f"已将 {mfa_bin_dir} 加入 PATH")
            
            # 验证安装
            if verify_mfa_working():
                logger.info("MFA 验证通过")
            else:
                logger.warning("MFA 安装后验证失败")
        
    except subprocess.TimeoutExpired:
        logger.error("MFA 安装超时")
    except subprocess.CalledProcessError as e:
        logger.error(f"MFA 安装失败: {e.stderr[-500:] if e.stderr else e}")
    except Exception as e:
        logger.error(f"MFA 安装异常: {e}")


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
    """下载 MFA 中文和日语模型（带完整性校验）"""
    logger.info("\n【下载 MFA 模型】")
    
    try:
        from src.mfa_model_downloader import download_language_models, _verify_file_integrity, LANGUAGE_MODELS
        
        languages = ["mandarin", "japanese"]
        
        for lang in languages:
            logger.info(f"\n下载 {lang} 模型...")
            
            # 检查现有字典文件是否损坏
            dict_config = LANGUAGE_MODELS[lang]["dictionary"]
            dict_path = MFA_DIR / dict_config["filename"]
            hash_path = MFA_DIR / (dict_config["filename"] + ".sha256")
            
            if dict_path.exists():
                # 如果没有哈希文件，说明是旧版本下载的，需要验证
                if not hash_path.exists():
                    logger.info(f"检测到旧版字典文件（无哈希），验证完整性...")
                    min_lines = dict_config.get("min_lines")
                    is_valid, reason = _verify_file_integrity(str(dict_path), min_lines, logger.info)
                    if not is_valid:
                        logger.warning(f"字典文件损坏: {reason}，删除并重新下载...")
                        try:
                            dict_path.unlink()
                        except Exception as e:
                            logger.error(f"删除损坏文件失败: {e}")
            
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
