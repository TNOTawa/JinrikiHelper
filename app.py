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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 确保日志立即输出（禁用缓冲）
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

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


def cleanup_temp_files():
    """
    启动时清理临时文件，释放磁盘空间
    
    清理目标：
    - /tmp/gradio/* (Gradio 上传缓存)
    - /tmp/jinriki_* (本应用的工作空间)
    - /tmp/mfa_* (MFA 临时文件)
    """
    import shutil
    import time
    
    logger.info("清理临时文件...")
    
    cleaned_size = 0
    cleaned_count = 0
    
    # 清理 Gradio 缓存
    gradio_tmp = Path("/tmp/gradio")
    if gradio_tmp.exists():
        try:
            for item in gradio_tmp.iterdir():
                try:
                    if item.is_dir():
                        size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        shutil.rmtree(item)
                    else:
                        size = item.stat().st_size
                        item.unlink()
                    cleaned_size += size
                    cleaned_count += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"清理 Gradio 缓存失败: {e}")
    
    # 清理 jinriki 工作空间
    tmp_dir = Path("/tmp")
    if tmp_dir.exists():
        try:
            for item in tmp_dir.iterdir():
                if item.name.startswith("jinriki_") or item.name.startswith("mfa_"):
                    try:
                        if item.is_dir():
                            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                            shutil.rmtree(item)
                        else:
                            size = item.stat().st_size
                            item.unlink()
                        cleaned_size += size
                        cleaned_count += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"清理工作空间失败: {e}")
    
    if cleaned_count > 0:
        size_mb = cleaned_size / (1024 * 1024)
        logger.info(f"已清理 {cleaned_count} 个临时文件/目录，释放 {size_mb:.1f} MB")
    else:
        logger.info("无需清理临时文件")


def ensure_ffmpeg():
    """确保 ffmpeg 已安装（用于音频格式转换，支持 m4a 等格式）"""
    import shutil
    
    if shutil.which("ffmpeg"):
        logger.info("ffmpeg 已安装")
        return True
    
    logger.info("ffmpeg 未安装，尝试安装...")
    
    try:
        # 尝试使用 apt-get 安装（Debian/Ubuntu）
        result = subprocess.run(
            ["apt-get", "update"],
            capture_output=True, text=True, timeout=60
        )
        result = subprocess.run(
            ["apt-get", "install", "-y", "ffmpeg"],
            capture_output=True, text=True, timeout=120
        )
        
        if shutil.which("ffmpeg"):
            logger.info("ffmpeg 安装成功")
            return True
        else:
            logger.warning("ffmpeg 安装后仍未找到")
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg 安装超时")
        return False
    except Exception as e:
        logger.warning(f"ffmpeg 安装失败: {e}")
        return False


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
    
    # 云端环境启动时清理临时文件，释放磁盘空间
    if is_cloud:
        cleanup_temp_files()
    
    logger.info("清理完成，继续初始化...")
    
    # 确保 ffmpeg 已安装（支持 m4a 等音频格式）
    if is_cloud or platform.system() != "Windows":
        logger.info("检查 ffmpeg...")
        ensure_ffmpeg()
        logger.info("ffmpeg 检查完成")
    
    # 魔搭创空间无法访问 HuggingFace，使用镜像
    if is_cloud and Path("/home/studio_service").exists():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("已设置 HuggingFace 镜像: hf-mirror.com")
    
    # 设置 pkuseg 模型目录（中文分词依赖，必须在 MFA 运行前设置）
    if platform.system() != "Windows":
        pkuseg_home = PERSISTENT_MODELS_DIR / "pkuseg" if PERSISTENT_MODELS_DIR.parent.exists() else Path("/root/.pkuseg")
        pkuseg_home.mkdir(parents=True, exist_ok=True)
        os.environ["PKUSEG_HOME"] = str(pkuseg_home)
        logger.info(f"设置 PKUSEG_HOME: {pkuseg_home}")
    
    # Linux 环境下始终尝试安装 MFA（无论是否云端）
    if platform.system() != "Windows":
        logger.info("Linux 环境，检查并安装 MFA...")
        setup_mfa_linux()
        logger.info("MFA 设置完成")
    
    if is_cloud:
        logger.info("检测到云端环境，正在初始化...")
        
        # 设置临时目录
        os.environ.setdefault("TMPDIR", "/tmp")
        
        # 下载所有必需模型
        logger.info("开始下载模型...")
        download_all_models()
        logger.info("模型下载完成")
    else:
        logger.info("本地环境运行")


def setup_mfa_linux():
    """Linux 环境下安装 MFA（使用 micromamba），失败则退出程序"""
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
            result = subprocess.run([
                "bash", "-c",
                f'curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C {mamba_root} bin/micromamba'
            ], check=True, capture_output=True, text=True, timeout=120)
            logger.info("micromamba 下载完成")
        else:
            logger.info("micromamba 已存在，跳过下载")
        
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
            logger.info("使用 micromamba 安装 MFA（这可能需要几分钟）...")
            env = os.environ.copy()
            env["MAMBA_ROOT_PREFIX"] = str(mamba_root)
            
            # 创建环境并安装 MFA（指定 Python 3.11）
            logger.info("执行 micromamba create...")
            result = subprocess.run([
                str(mamba_bin), "create", "-n", "mfa",
                "-c", "conda-forge",
                "python=3.11",
                "montreal-forced-aligner",
                "-y"
            ], env=env, check=True, capture_output=True, text=True, timeout=600)
            logger.info("MFA 环境创建完成")
            
            # 更新确保使用 CPU 版本的 kaldi
            logger.info("安装 CPU 版本 kaldi...")
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
                result = subprocess.run([
                    str(pip_path), "install",
                    "spacy-pkuseg", "dragonmapper", "hanziconv",  # 中文
                    "sudachipy", "sudachidict_core"  # 日语
                ], capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.error(f"分词依赖安装失败: {result.stderr}")
                    sys.exit(1)
                logger.info("分词依赖安装完成")
            else:
                logger.info("分词依赖已存在")
            
            # pkuseg 模型下载移到 download_pkuseg_models() 独立函数中
        
        # 4. 确保 MFA 环境的 bin 目录在 PATH 中
        if mfa_bin_dir.exists() and str(mfa_bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{mfa_bin_dir}:{os.environ.get('PATH', '')}"
            logger.info(f"已将 {mfa_bin_dir} 加入 PATH")
            
            # 验证安装
            if verify_mfa_working():
                logger.info("MFA 验证通过")
            else:
                logger.error("MFA 安装后验证失败")
                sys.exit(1)
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"MFA 安装超时: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"MFA 安装失败: {e.stderr[-500:] if e.stderr else e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"MFA 安装异常: {e}")
        sys.exit(1)


def download_all_models():
    """下载所有必需的模型，任何模型下载失败则退出程序"""
    logger.info("=" * 50)
    logger.info("开始下载模型...")
    logger.info("=" * 50)
    
    # 确保目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MFA_DIR, exist_ok=True)
    
    errors = []
    
    # 1. 下载 Silero VAD
    if not download_silero_vad_model():
        errors.append("Silero VAD")
    
    # 2. 下载 Whisper 模型
    if not download_whisper_models():
        errors.append("Whisper")
    
    # 3. 下载 MFA 模型（中文和日语）
    if not download_mfa_models_all():
        errors.append("MFA")
    
    # 4. 下载 pkuseg 模型（中文分词必需，独立下载确保执行）
    if not download_pkuseg_models():
        errors.append("pkuseg")
    
    if errors:
        logger.error("=" * 50)
        logger.error(f"以下模型加载失败: {', '.join(errors)}")
        logger.error("程序无法继续运行，退出")
        logger.error("=" * 50)
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("所有模型下载完成")
    logger.info("=" * 50)


def download_pkuseg_models() -> bool:
    """下载 pkuseg 中文分词模型，返回是否成功
    
    spacy-pkuseg 检查模型的逻辑：
    1. 先检查 PKUSEG_HOME/<model_name>.zip 是否存在
    2. 如果 zip 存在，解压到 PKUSEG_HOME/<model_name>/ 目录
    3. 如果 zip 不存在，从 GitHub 下载
    
    因此我们需要保留 .zip 文件，否则 spacy_pkuseg 会尝试重新下载
    
    注意：只需要 spacy_ontonotes.zip，postag.zip 在 GitHub releases 中不存在
    """
    logger.info("\n【下载 pkuseg 模型】")
    
    pkuseg_home = Path(os.environ.get("PKUSEG_HOME", "/root/.pkuseg"))
    pkuseg_home.mkdir(parents=True, exist_ok=True)
    
    pkuseg_model_dir = pkuseg_home / "spacy_ontonotes"
    postag_model_dir = pkuseg_home / "postag"
    
    # 关键：检查 spacy_ontonotes.zip 是否存在（spacy_pkuseg 的检查逻辑）
    # postag.zip 在 GitHub releases 中不存在，不需要检查
    spacy_ontonotes_zip = pkuseg_home / "spacy_ontonotes.zip"
    
    if spacy_ontonotes_zip.exists():
        logger.info(f"pkuseg 模型 zip 文件已存在: {pkuseg_home}")
        # 列出目录内容供调试
        files = [f.name for f in pkuseg_home.iterdir()]
        logger.info(f"pkuseg 目录内容: {files}")
        return True
    
    # 检查是否有文件被错误解压到根目录（旧版本遗留问题）
    root_msgpack = pkuseg_home / "features.msgpack"
    if root_msgpack.exists():
        logger.info("检测到模型文件在根目录，移动到正确位置...")
        pkuseg_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动 spacy_ontonotes 相关文件
        for filename in ["features.msgpack", "weights.npz"]:
            src = pkuseg_home / filename
            if src.exists():
                dst = pkuseg_model_dir / filename
                src.rename(dst)
                logger.info(f"移动 {filename} -> spacy_ontonotes/")
        
        # 移动 postag 相关文件
        postag_model_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["features.pkl"]:
            src = pkuseg_home / filename
            if src.exists():
                dst = postag_model_dir / filename
                src.rename(dst)
                logger.info(f"移动 {filename} -> postag/")
    
    # 只检查 spacy_ontonotes.zip（这是 spacy_pkuseg 必需的）
    # postag 模型在 GitHub releases 中不存在，spacy_pkuseg 会使用内置的词性标注
    if spacy_ontonotes_zip.exists():
        logger.info(f"pkuseg 模型已就绪: {pkuseg_home}")
        return True
    
    # 需要下载模型
    logger.info("需要下载 pkuseg 模型: spacy_ontonotes")
    
    # 使用 spacy-pkuseg 的模型（新格式 msgpack）
    # 注意：必须保留 .zip 文件，spacy_pkuseg 会检查 zip 是否存在
    # postag.zip 在 GitHub releases 中不存在，不需要下载
    models = [
        {
            "name": "spacy_ontonotes",
            "urls": [
                "https://ghfast.top/https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
                "https://gh-proxy.com/https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip", 
                "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
            ],
            "check_file": "features.msgpack",
        },
    ]
    
    for model in models:
        model_name = model["name"]
        model_dir = pkuseg_home / model_name
        zip_path = pkuseg_home / f"{model_name}.zip"
        check_file = model_dir / model["check_file"]
        
        downloaded = False
        for url in model["urls"]:
            logger.info(f"下载 {model_name}: {url}")
            
            try:
                # 下载
                result = subprocess.run(
                    ["curl", "-fsSL", "-o", str(zip_path), url],
                    capture_output=True, text=True, timeout=180
                )
                
                if result.returncode != 0:
                    logger.warning(f"curl 下载失败: {result.stderr}")
                    continue
                
                if not zip_path.exists() or zip_path.stat().st_size < 1000:
                    logger.warning(f"下载文件无效或太小")
                    continue
                
                logger.info(f"下载完成，文件大小: {zip_path.stat().st_size} bytes")
                
                # 创建目标目录并解压到其中（zip 内部没有目录结构）
                model_dir.mkdir(parents=True, exist_ok=True)
                
                result = subprocess.run(
                    ["unzip", "-o", "-q", str(zip_path), "-d", str(model_dir)],
                    capture_output=True, text=True, timeout=60
                )
                
                if result.returncode != 0:
                    logger.warning(f"unzip 解压失败: {result.stderr}")
                    continue
                
                # 重要：保留 zip 文件！spacy_pkuseg 会检查 zip 是否存在
                # 不要删除 zip_path
                
                # 验证
                if check_file.exists():
                    logger.info(f"{model_name} 下载并解压成功（保留 zip 文件）")
                    files = [f.name for f in model_dir.iterdir()]
                    logger.info(f"{model_name} 目录内容: {files}")
                    downloaded = True
                    break
                else:
                    logger.warning(f"解压后文件不存在: {check_file}")
                    if model_dir.exists():
                        files = [f.name for f in model_dir.iterdir()]
                        logger.info(f"{model_name} 目录内容: {files}")
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"下载超时: {url}")
            except Exception as e:
                logger.warning(f"下载异常: {e}")
        
        if not downloaded:
            logger.error(f"{model_name} 所有镜像下载失败")
            return False
    
    logger.info("pkuseg 模型下载完成")
    return True


def download_silero_vad_model() -> bool:
    """下载 Silero VAD 模型，返回是否成功"""
    logger.info("\n【下载 Silero VAD 模型】")
    
    try:
        from src.silero_vad_downloader import download_silero_vad, is_vad_model_downloaded
        
        if is_vad_model_downloaded(str(MODELS_DIR)):
            logger.info("Silero VAD 模型已存在，跳过下载")
            return True
        
        success, result = download_silero_vad(str(MODELS_DIR), logger.info)
        if success:
            logger.info(f"Silero VAD 下载成功: {result}")
            return True
        else:
            logger.error(f"Silero VAD 下载失败: {result}")
            return False
    except Exception as e:
        logger.error(f"Silero VAD 下载异常: {e}")
        return False


def download_whisper_models() -> bool:
    """下载 Whisper 模型 (small 和 medium)，返回是否成功"""
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
                logger.error(f"{model_name} 下载失败: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Whisper 模型下载异常: {e}")
        return False


def download_mfa_models_all() -> bool:
    """下载 MFA 中文和日语模型（带完整性校验），返回是否成功"""
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
            
            success, acoustic_path, dict_path = download_language_models(
                lang, str(MFA_DIR), logger.info
            )
            if not success:
                logger.error(f"{lang} 模型下载失败")
                return False
            logger.info(f"{lang} 模型下载完成")
        
        return True
    except Exception as e:
        logger.error(f"MFA 模型下载异常: {e}")
        return False


def main():
    """主入口"""
    logger.info("=" * 50)
    logger.info("人力V助手 云端版启动")
    logger.info("=" * 50)
    
    setup_environment()
    
    logger.info("环境初始化完成，启动 Gradio UI...")
    
    # 导入并启动云端 GUI
    from src.gui_cloud import create_cloud_ui
    
    app = create_cloud_ui()
    
    logger.info("Gradio UI 创建完成，启动服务...")
    
    # 云端配置
    # 启用队列，魔搭CPU按需分配，无需设置并发上限
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
