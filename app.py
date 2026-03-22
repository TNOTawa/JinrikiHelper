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
import time
import zipfile
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


def ensure_ffmpeg():
    """确保 ffmpeg 已安装（用于音频格式转换，支持 m4a 等格式）"""
    import shutil

    if shutil.which("ffmpeg"):
        logger.info("ffmpeg 已安装")
        return True

    logger.info("ffmpeg 未安装，尝试安装...")

    try:
        subprocess.run(["apt-get", "update"], capture_output=True, text=True, timeout=60)
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"], capture_output=True, text=True, timeout=120)

        if shutil.which("ffmpeg"):
            logger.info("ffmpeg 安装成功")
            return True

        logger.warning("ffmpeg 安装后仍未找到")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg 安装超时")
        return False
    except Exception as e:
        logger.warning(f"ffmpeg 安装失败: {e}")
        return False


def setup_mfa_linux() -> bool:
    """Linux 环境下安装 MFA（使用 MFA 官方推荐的 conda-forge 方案）

    返回:
        bool: MFA 是否可用
    """
    import shutil
    import importlib.util
    import tarfile
    import tempfile
    import urllib.request
    import stat

    def _run_cmd_ok(cmd, timeout=30):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, (result.stdout or ""), (result.stderr or "")
        except Exception:
            return False, "", ""

    def _get_callable_conda_tools() -> dict[str, str]:
        """查找可调用的 conda 类工具（conda/mamba/micromamba）。"""
        tools = {}
        for tool_name in ["conda", "mamba", "micromamba"]:
            tool_path = shutil.which(tool_name)
            if not tool_path:
                continue

            ok, _, _ = _run_cmd_ok([tool_path, "--version"], timeout=20)
            if ok:
                tools[tool_name] = tool_path
            else:
                logger.warning(f"检测到 {tool_name} 但不可调用，已跳过: {tool_path}")
        return tools

    def _ensure_micromamba_available() -> str | None:
        """自动下载并配置 micromamba（无控制台场景）。"""
        mm_path = shutil.which("micromamba")
        if mm_path:
            return mm_path

        if PERSISTENT_MODELS_DIR.parent.exists():
            mamba_root = PERSISTENT_MODELS_DIR.parent / "micromamba"
        else:
            mamba_root = BASE_DIR / ".micromamba"

        mamba_bin_dir = mamba_root / "bin"
        mamba_bin_dir.mkdir(parents=True, exist_ok=True)
        target_bin = mamba_bin_dir / "micromamba"

        os.environ["MAMBA_ROOT_PREFIX"] = str(mamba_root)
        if str(mamba_bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{mamba_bin_dir}:{os.environ.get('PATH', '')}"

        if target_bin.exists():
            target_bin.chmod(target_bin.stat().st_mode | stat.S_IEXEC)
            logger.info(f"检测到已存在 micromamba: {target_bin}")
            return str(target_bin)

        urls = [
            "https://micro.mamba.pm/api/micromamba/linux-64/latest",
            "https://gh-proxy.com/https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64",
            "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64",
        ]

        for url in urls:
            for attempt in range(1, 3):
                try:
                    logger.info(f"下载 micromamba: {url} (第{attempt}次)")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        with urllib.request.urlopen(url, timeout=180) as resp:
                            data = resp.read()
                        tmp_path.write_bytes(data)

                        # 处理 tar.bz2 包格式
                        extracted = False
                        try:
                            with tarfile.open(tmp_path, mode="r:*") as tar:
                                member = None
                                for m in tar.getmembers():
                                    if m.name.endswith("/bin/micromamba") or m.name == "bin/micromamba":
                                        member = m
                                        break
                                if member is not None:
                                    src = tar.extractfile(member)
                                    if src is None:
                                        raise RuntimeError("无法从压缩包读取 micromamba")
                                    target_bin.write_bytes(src.read())
                                    extracted = True
                        except tarfile.TarError:
                            extracted = False

                        # 处理单文件二进制格式
                        if not extracted:
                            target_bin.write_bytes(tmp_path.read_bytes())

                        target_bin.chmod(target_bin.stat().st_mode | stat.S_IEXEC)
                        logger.info(f"micromamba 就绪: {target_bin}")
                        return str(target_bin)
                    finally:
                        if tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"下载/安装 micromamba 失败: {e}")

        logger.error("自动配置 micromamba 失败")
        return None

    def verify_mfa_working() -> bool:
        mfa_env_name = os.environ.get("JINRIKI_MFA_ENV_NAME", "mfa")
        commands = []
        conda_tools = _get_callable_conda_tools()

        # 优先官方 conda/mamba 入口，避免 PATH 中残留的 pip 版 mfa（缺少 _kalpy）
        conda = conda_tools.get("conda")
        if conda:
            commands.append([conda, "run", "-n", mfa_env_name, "mfa", "--help"])
            commands.append([conda, "run", "-n", "base", "mfa", "--help"])

        micromamba = conda_tools.get("micromamba")
        if micromamba:
            commands.append([micromamba, "run", "-n", mfa_env_name, "mfa", "--help"])
            commands.append([micromamba, "run", "-n", "base", "mfa", "--help"])

        mamba = conda_tools.get("mamba")
        if mamba:
            commands.append([mamba, "run", "-n", mfa_env_name, "mfa", "--help"])
            commands.append([mamba, "run", "-n", "base", "mfa", "--help"])

        commands.extend([
            [sys.executable, "-m", "montreal_forced_aligner.command_line.mfa", "--help"],
            [sys.executable, "-m", "montreal_forced_aligner", "--help"],
            ["mfa", "--help"],
        ])

        py_bin_dir = Path(sys.executable).parent
        mfa_bin = py_bin_dir / "mfa"
        if mfa_bin.exists():
            commands.insert(0, [str(mfa_bin), "--help"])

        for cmd in commands:
            ok, stdout, stderr = _run_cmd_ok(cmd, timeout=120)
            if ok:
                logger.info(f"MFA 验证命令通过: {' '.join(cmd)}")
                return True

            output = f"{stdout}\n{stderr}"
            if "No module named '_kalpy'" in output:
                logger.warning(f"命令 {' '.join(cmd)} 缺少 _kalpy，跳过该入口")

        logger.warning("MFA 验证命令均未通过，可能缺少 kalpy/_kalpy 或入口脚本未加入 PATH")
        return False

    # 检查是否已可用
    if verify_mfa_working():
        logger.info("MFA 已安装且工作正常")
        return True

    logger.info("MFA 不可用，Linux 下将使用 conda/mamba 从 conda-forge 安装（官方推荐）...")

    try:
        mfa_env_name = os.environ.get("JINRIKI_MFA_ENV_NAME", "mfa")
        conda_tools = _get_callable_conda_tools()

        # 无控制台场景：自动补齐 micromamba
        if not conda_tools:
            logger.info("未检测到 conda/mamba/micromamba，开始自动配置 micromamba...")
            _ensure_micromamba_available()
            conda_tools = _get_callable_conda_tools()

        install_attempts = []

        mamba = conda_tools.get("mamba")
        if mamba:
            install_attempts.append((
                "mamba",
                [mamba, "install", "-y", "-n", "base", "-c", "conda-forge", "montreal-forced-aligner"],
            ))

        micromamba = conda_tools.get("micromamba")
        if micromamba:
            install_attempts.append((
                f"micromamba(create:{mfa_env_name})",
                [micromamba, "create", "-y", "-n", mfa_env_name, "-c", "conda-forge", "montreal-forced-aligner"],
            ))
            install_attempts.append((
                f"micromamba(install:{mfa_env_name})",
                [micromamba, "install", "-y", "-n", mfa_env_name, "-c", "conda-forge", "montreal-forced-aligner"],
            ))
            install_attempts.append((
                "micromamba(base)",
                [micromamba, "install", "-y", "-n", "base", "-c", "conda-forge", "montreal-forced-aligner"],
            ))

        conda = conda_tools.get("conda")
        if conda:
            install_attempts.append((
                "conda",
                [conda, "install", "-y", "-n", "base", "-c", "conda-forge", "montreal-forced-aligner"],
            ))

        if not install_attempts:
            logger.error("未找到 conda/mamba/micromamba，无法按官方推荐方法安装 MFA")
            return False

        installed = False
        for installer_name, install_cmd in install_attempts:
            logger.info(f"尝试使用 {installer_name} 安装 MFA...")
            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            if result.returncode == 0:
                logger.info(f"{installer_name} 安装 MFA 完成")
                installed = True
                break

            stderr_tail = (result.stderr or result.stdout or "")[-800:]
            logger.warning(f"{installer_name} 安装 MFA 失败: {stderr_tail}")

        if not installed:
            logger.error("所有 conda/mamba 安装尝试均失败")
            return False

        # 某些云端环境不会自动刷新 PATH，补充常见 conda bin 目录
        if not shutil.which("mfa"):
            candidate_dirs = [Path("/opt/conda/bin"), Path.home() / "micromamba" / "bin"]
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                candidate_dirs.insert(0, Path(conda_prefix) / "bin")

            for candidate_dir in candidate_dirs:
                candidate_mfa = candidate_dir / "mfa"
                if candidate_mfa.exists() and str(candidate_dir) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{candidate_dir}:{os.environ.get('PATH', '')}"
                    logger.info(f"已将 {candidate_dir} 加入 PATH")
                    break

        # 在 MFA 环境中安装日语/中文分词依赖（必须在 MFA 环境中，而不是系统 Python）
        # 否则 MFA 无法访问 spacy/sudachipy 包
        pkuseg_home = PERSISTENT_MODELS_DIR / "pkuseg" if PERSISTENT_MODELS_DIR.parent.exists() else Path("/root/.pkuseg")
        pkuseg_home.mkdir(parents=True, exist_ok=True)
        os.environ["PKUSEG_HOME"] = str(pkuseg_home)

        logger.info("在 MFA 环境中安装日语/中文分词支持...")
        conda_tools = _get_callable_conda_tools()
        mfa_env_name = os.environ.get("JINRIKI_MFA_ENV_NAME", "mfa")
        
        # 优先使用conda-forge安装到MFA环境（最稳定）
        deps_installed = False
        
        # 尝试候选安装方式
        install_attempts_deps = []
        
        if conda_tools.get("micromamba"):
            micromamba = conda_tools["micromamba"]
            install_attempts_deps.extend([
                ("micromamba(conda-forge)", 
                 [micromamba, "install", "-y", "-n", mfa_env_name, "-c", "conda-forge", 
                  "spacy", "sudachipy", "sudachidict-core"]),
                ("micromamba(pip in mfa env)", 
                 [micromamba, "run", "-n", mfa_env_name, "pip", "install", "--no-cache-dir",
                  "spacy-pkuseg", "dragonmapper", "hanziconv"]),
            ])
        
        if conda_tools.get("mamba"):
            mamba = conda_tools["mamba"]
            install_attempts_deps.append(
                ("mamba(conda-forge)",
                 [mamba, "install", "-y", "-n", mfa_env_name, "-c", "conda-forge",
                  "spacy", "sudachipy", "sudachidict-core"])
            )
        
        if conda_tools.get("conda"):
            conda = conda_tools["conda"]
            install_attempts_deps.append(
                ("conda(conda-forge)",
                 [conda, "install", "-y", "-n", mfa_env_name, "-c", "conda-forge",
                  "spacy", "sudachipy", "sudachidict-core"])
            )
        
        for installer_name, dep_cmd in install_attempts_deps:
            logger.info(f"尝试用 {installer_name} 安装分词依赖...")
            dep_result = subprocess.run(dep_cmd, capture_output=True, text=True, timeout=600, check=False)
            if dep_result.returncode == 0:
                logger.info(f"{installer_name} 分词依赖安装完成")
                deps_installed = True
                break
            else:
                stderr_tail = (dep_result.stderr or "")[-500:]
                logger.warning(f"{installer_name} 分词依赖安装失败: {stderr_tail}")
        
        if not deps_installed:
            logger.warning("MFA 环境分词依赖安装失败，某些语言对齐可能不可用（继续运行）")

        if verify_mfa_working():
            logger.info("MFA 验证通过")
            return True

        logger.warning("MFA 安装后仍不可用，将以无 MFA 模式继续")
        return False
    except subprocess.TimeoutExpired as e:
        logger.error(f"MFA 安装超时: {e}")
        return False
    except Exception as e:
        logger.error(f"MFA 安装异常: {e}")
        return False


def setup_environment():
    """初始化云端环境"""
    global MODELS_DIR, MFA_DIR

    # 初始化模型目录（可能创建符号链接）
    MODELS_DIR = get_models_dir()
    MFA_DIR = MODELS_DIR / "mfa"

    # 检测运行环境
    is_cloud = any([
        os.environ.get("SPACE_ID"),
        os.environ.get("MODELSCOPE_SPACE"),
        os.environ.get("GRADIO_SERVER_NAME"),
        Path("/home/studio_service").exists(),
    ])

    if is_cloud or platform.system() != "Windows":
        ensure_ffmpeg()

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

        logger.info("Linux 环境，检查并安装 MFA...")
        mfa_ok = setup_mfa_linux()
        if not mfa_ok:
            logger.warning("MFA 当前不可用，将跳过对齐功能但继续启动服务")

    if is_cloud:
        os.environ.setdefault("TMPDIR", "/tmp")
        download_all_models()
    else:
        logger.info("本地环境运行")


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
        strict_mode = os.environ.get("JINRIKI_STRICT_MODEL_DOWNLOAD", "0") == "1"
        if strict_mode:
            logger.error("严格模式开启，程序退出（可通过 JINRIKI_STRICT_MODEL_DOWNLOAD=0 关闭）")
            logger.error("=" * 50)
            sys.exit(1)

        logger.warning("将以降级模式继续启动（部分功能可能不可用）")
        logger.warning("如需下载成功后再启动，请设置 JINRIKI_STRICT_MODEL_DOWNLOAD=1")
        logger.error("=" * 50)
    
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

    def _pkuseg_model_ready() -> bool:
        feature_candidates = [
            pkuseg_model_dir / "features.msgpack",
            pkuseg_model_dir / "features.pkl",
            pkuseg_model_dir / "features.json",
        ]
        has_feature = any(p.exists() for p in feature_candidates)
        has_unigram = (pkuseg_model_dir / "unigram_word.txt").exists()
        return spacy_ontonotes_zip.exists() and has_feature and has_unigram

    def _repair_extract_from_zip() -> bool:
        if not spacy_ontonotes_zip.exists():
            return False
        try:
            pkuseg_model_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(spacy_ontonotes_zip, "r") as zf:
                zf.extractall(pkuseg_model_dir)

            # 兼容 zip 内部带一层 spacy_ontonotes/ 目录的情况
            nested_dir = pkuseg_model_dir / "spacy_ontonotes"
            if nested_dir.is_dir():
                import shutil
                for item in nested_dir.iterdir():
                    dst = pkuseg_model_dir / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst, ignore_errors=True)
                        else:
                            dst.unlink(missing_ok=True)
                    shutil.move(str(item), str(dst))
                shutil.rmtree(nested_dir, ignore_errors=True)

            return _pkuseg_model_ready()
        except Exception as e:
            logger.warning(f"修复 pkuseg 解压失败: {e}")
            return False

    if _pkuseg_model_ready():
        logger.info(f"pkuseg 模型已就绪: {pkuseg_home}")
        return True

    # zip 已存在但模型不完整：优先尝试重新解压修复
    if spacy_ontonotes_zip.exists():
        logger.warning("检测到 spacy_ontonotes.zip 已存在，但模型目录不完整，尝试修复解压...")
        if _repair_extract_from_zip():
            logger.info("pkuseg 模型修复成功")
            return True
        logger.warning("pkuseg 模型修复失败，准备重新下载")
        try:
            spacy_ontonotes_zip.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"删除损坏 zip 失败: {e}")
    
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
    if _pkuseg_model_ready():
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
                "https://gitcode.com/gh_mirrors/sp/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
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
        timeout_seconds = int(os.environ.get("JINRIKI_PKUSEG_DOWNLOAD_TIMEOUT", "300"))
        max_rounds = int(os.environ.get("JINRIKI_PKUSEG_DOWNLOAD_ROUNDS", "3"))

        for round_idx in range(1, max_rounds + 1):
            logger.info(f"{model_name} 下载轮次: {round_idx}/{max_rounds}")
            for url in model["urls"]:
                logger.info(f"下载 {model_name}: {url}")

                try:
                    # 下载
                    result = subprocess.run(
                        ["curl", "-fL", "--retry", "2", "--retry-delay", "2", "-o", str(zip_path), url],
                        capture_output=True, text=True, timeout=timeout_seconds
                    )

                    if result.returncode != 0:
                        logger.warning(f"curl 下载失败: {result.stderr}")
                        continue

                    if not zip_path.exists() or zip_path.stat().st_size < 1000:
                        logger.warning("下载文件无效或太小")
                        continue

                    logger.info(f"下载完成，文件大小: {zip_path.stat().st_size} bytes")

                    # 创建目标目录并解压到其中（优先 python zipfile，避免系统 unzip 差异）
                    model_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(model_dir)
                    except Exception as zip_e:
                        logger.warning(f"zipfile 解压失败，尝试系统 unzip: {zip_e}")
                        result = subprocess.run(
                            ["unzip", "-o", "-q", str(zip_path), "-d", str(model_dir)],
                            capture_output=True, text=True, timeout=120
                        )
                        if result.returncode != 0:
                            logger.warning(f"unzip 解压失败: {result.stderr}")
                            continue

                    # 兼容 zip 内部带一层 model_name/ 的情况
                    nested_dir = model_dir / model_name
                    if nested_dir.is_dir():
                        import shutil
                        for item in nested_dir.iterdir():
                            dst = model_dir / item.name
                            if dst.exists():
                                if dst.is_dir():
                                    shutil.rmtree(dst, ignore_errors=True)
                                else:
                                    dst.unlink(missing_ok=True)
                            shutil.move(str(item), str(dst))
                        shutil.rmtree(nested_dir, ignore_errors=True)

                    # 重要：保留 zip 文件！spacy_pkuseg 会检查 zip 是否存在
                    # 不要删除 zip_path

                    # 验证：至少要有 check_file + unigram_word.txt
                    unigram_file = model_dir / "unigram_word.txt"
                    if check_file.exists() and unigram_file.exists():
                        logger.info(f"{model_name} 下载并解压成功（保留 zip 文件）")
                        files = [f.name for f in model_dir.iterdir()]
                        logger.info(f"{model_name} 目录内容: {files}")
                        downloaded = True
                        break
                    else:
                        logger.warning(f"解压后关键文件缺失: {check_file.name} / {unigram_file.name}")
                        if model_dir.exists():
                            files = [f.name for f in model_dir.iterdir()]
                            logger.info(f"{model_name} 目录内容: {files}")

                except subprocess.TimeoutExpired:
                    logger.warning(f"下载超时: {url}")
                except Exception as e:
                    logger.warning(f"下载异常: {e}")

            if downloaded:
                break

            sleep_seconds = min(5 * round_idx, 15)
            logger.info(f"{model_name} 本轮未成功，{sleep_seconds} 秒后重试")
            time.sleep(sleep_seconds)
        
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
        per_language_retries = int(os.environ.get("JINRIKI_MFA_MODEL_RETRIES", "3"))
        
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
            
            success = False
            last_err = ""
            for attempt in range(1, per_language_retries + 1):
                logger.info(f"{lang} 下载尝试: {attempt}/{per_language_retries}")
                ok, acoustic_path, dict_path = download_language_models(
                    lang, str(MFA_DIR), logger.info
                )
                if ok:
                    success = True
                    break

                last_err = dict_path
                wait_seconds = min(10 * attempt, 30)
                logger.warning(f"{lang} 下载失败（第{attempt}次）: {last_err}")
                if attempt < per_language_retries:
                    logger.info(f"{wait_seconds} 秒后重试 {lang}...")
                    time.sleep(wait_seconds)

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
    setup_environment()
    
    # 导入并启动云端 GUI
    from src.gui_cloud import create_cloud_ui
    
    app = create_cloud_ui()
    
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
