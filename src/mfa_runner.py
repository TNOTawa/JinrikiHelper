# -*- coding: utf-8 -*-
"""
MFA 调用模块
支持 Windows (外挂模式) 和 Linux (系统安装) 双平台
"""

import os
import platform
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# 定位路径
BASE_DIR = Path(__file__).parent.parent.absolute()
MFA_ENGINE_DIR = BASE_DIR / "tools" / "mfa_engine"
MFA_PYTHON = MFA_ENGINE_DIR / "python.exe"

# 默认模型路径
DEFAULT_DICT_PATH = BASE_DIR / "models" / "mandarin.dict"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "mandarin.zip"
DEFAULT_TEMP_DIR = BASE_DIR / "mfa_temp"

# 平台检测
IS_WINDOWS = platform.system() == "Windows"


def check_mfa_available() -> bool:
    """
    检查 MFA 是否可用
    Windows: 检查外挂 Python 环境
    Linux: 检查 mfa 命令是否可用
    """
    if IS_WINDOWS:
        if not MFA_ENGINE_DIR.exists():
            logger.warning(f"MFA 引擎目录不存在: {MFA_ENGINE_DIR}")
            return False
        if not MFA_PYTHON.exists():
            logger.warning(f"MFA Python 不存在: {MFA_PYTHON}")
            return False
        return True
    else:
        # Linux/macOS: 检查 mfa 命令
        mfa_path = shutil.which("mfa")
        if mfa_path:
            # 验证 mfa 能正常运行
            # 云端首次运行可能需要较长时间初始化，设置 120 秒超时
            try:
                result = subprocess.run(
                    ["mfa", "version"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    logger.info(f"MFA 可用: {result.stdout.strip()}")
                    return True
                else:
                    logger.warning(f"MFA 命令执行失败: {result.stderr or result.stdout}")
            except subprocess.TimeoutExpired:
                logger.warning("MFA 验证超时（120秒），可能正在初始化，将尝试继续使用")
                # 超时但命令存在，假设可用（实际对齐时会再次验证）
                return True
            except Exception as e:
                logger.warning(f"MFA 验证异常: {e}")
        else:
            logger.warning("未找到 mfa 命令，请使用 conda/micromamba 安装: conda install -c conda-forge montreal-forced-aligner")
        
        return False


def _get_mfa_command() -> list:
    """
    获取 MFA 命令前缀
    Windows: 使用外挂 Python 调用
    Linux: 使用系统 mfa 命令
    """
    if IS_WINDOWS:
        return [str(MFA_PYTHON), "-m", "montreal_forced_aligner"]
    else:
        return ["mfa"]


def _build_mfa_env(mfa_root: Optional[Path] = None) -> dict:
    """
    构造 MFA 专用环境变量
    
    参数:
        mfa_root: 会话独立的 MFA 数据目录（用于并发隔离）
    """
    env = os.environ.copy()
    
    if IS_WINDOWS:
        # Windows: 必须把 Library\bin 加入 PATH，否则 Kaldi DLL 找不到
        mfa_paths = [
            str(MFA_ENGINE_DIR),
            str(MFA_ENGINE_DIR / "Library" / "bin"),
            str(MFA_ENGINE_DIR / "Scripts"),
            str(MFA_ENGINE_DIR / "bin"),
        ]
        env["PATH"] = ";".join(mfa_paths) + ";" + env.get("PATH", "")
    else:
        # Linux: 设置会话独立的 MFA_ROOT_DIR（解决并发数据库冲突）
        if mfa_root:
            env["MFA_ROOT_DIR"] = str(mfa_root)
            logger.info(f"设置会话独立 MFA_ROOT_DIR: {mfa_root}")
        
        # Linux: 设置 pkuseg 模型目录（云端使用持久化路径）
        persistent_models = Path("/home/studio_service/models")
        if persistent_models.exists():
            pkuseg_home = persistent_models / "pkuseg"
            pkuseg_home.mkdir(parents=True, exist_ok=True)
            env["PKUSEG_HOME"] = str(pkuseg_home)
            logger.info(f"设置 PKUSEG_HOME: {pkuseg_home}")
            
            # 验证 pkuseg 模型是否存在（检查 zip 文件，这是 spacy_pkuseg 的检查方式）
            spacy_ontonotes_zip = pkuseg_home / "spacy_ontonotes.zip"
            if spacy_ontonotes_zip.exists():
                logger.info(f"pkuseg 模型 zip 已存在: {spacy_ontonotes_zip}")
            else:
                logger.warning(f"pkuseg 模型 zip 不存在: {spacy_ontonotes_zip}")
                # 列出目录内容供调试
                if pkuseg_home.exists():
                    files = list(pkuseg_home.iterdir())
                    logger.info(f"pkuseg 目录内容: {[f.name for f in files]}")
        
        # 确保从系统环境继承 PKUSEG_HOME（如果已设置）
        if "PKUSEG_HOME" not in env and os.environ.get("PKUSEG_HOME"):
            env["PKUSEG_HOME"] = os.environ["PKUSEG_HOME"]
    
    return env


def _clean_dict_empty_lines(dict_path: str) -> int:
    """
    清理字典文件中的空行和无效行
    MFA 3.x 解析字典时遇到空行会报 IndexError
    
    返回: 清理的无效行数量
    """
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        original_count = len(lines)
        
        # 过滤空行和只有空白字符的行
        # 同时过滤没有制表符分隔的无效行（字典格式: word\tprob\t...）
        valid_lines = []
        for line in lines:
            stripped = line.strip()
            # 跳过空行
            if not stripped:
                continue
            # 跳过没有制表符的行（无效格式）
            if '\t' not in line:
                logger.warning(f"跳过无效字典行: {stripped[:50]}...")
                continue
            valid_lines.append(line)
        
        removed_count = original_count - len(valid_lines)
        
        if removed_count > 0:
            with open(dict_path, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            logger.info(f"字典文件清理完成: 原 {original_count} 行, 现 {len(valid_lines)} 行, 移除 {removed_count} 行")
        else:
            logger.info(f"字典文件无需清理: {original_count} 行")
        
        return removed_count
    except PermissionError as e:
        logger.error(f"清理字典文件失败 - 权限不足: {e}")
        return 0
    except Exception as e:
        logger.error(f"清理字典文件失败: {e}")
        return 0


def _create_isolated_mfa_root(session_id: str) -> Path:
    """
    为每个会话创建独立的 MFA_ROOT_DIR，避免多用户并发时数据库冲突
    
    MFA 使用 MFA_ROOT_DIR 环境变量指定数据目录，包含：
    - pretrained_models/: 预训练模型缓存
    - 各种 .db 文件: SQLite 数据库
    
    通过为每个会话创建独立目录，完全隔离并发用户
    """
    import tempfile
    
    # 在系统临时目录下创建会话专属的 MFA 根目录
    mfa_root = Path(tempfile.gettempdir()) / f"mfa_session_{session_id}"
    mfa_root.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"创建会话独立 MFA 目录: {mfa_root}")
    return mfa_root


def _cleanup_isolated_mfa_root(mfa_root: Path):
    """清理会话独立的 MFA 目录"""
    if mfa_root and mfa_root.exists() and "mfa_session_" in str(mfa_root):
        try:
            shutil.rmtree(mfa_root)
            logger.info(f"已清理会话 MFA 目录: {mfa_root}")
        except Exception as e:
            logger.warning(f"清理会话 MFA 目录失败: {e}")


def run_mfa_alignment(
    corpus_dir: str,
    output_dir: str,
    dict_path: Optional[str] = None,
    model_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    single_speaker: bool = True,
    clean: bool = True,
    num_jobs: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    执行 MFA 对齐
    
    参数:
        corpus_dir: 包含 wav 和 lab/txt 的输入目录
        output_dir: TextGrid 输出目录
        dict_path: 字典文件路径，默认使用 models/mandarin.dict
        model_path: 声学模型路径，默认使用 models/mandarin.zip
        temp_dir: 临时目录，默认使用 mfa_temp（云端会自动创建独立目录）
        single_speaker: 是否为单说话人模式
        clean: 是否清理旧缓存
        num_jobs: 并行进程数，默认使用 CPU 核心数
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 输出信息或错误信息)
    """
    import uuid
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    # 为本次会话创建独立的 MFA 数据目录（并发安全）
    session_id = uuid.uuid4().hex[:8]
    isolated_mfa_root = _create_isolated_mfa_root(session_id) if not IS_WINDOWS else None
    
    # 检查环境
    if not check_mfa_available():
        platform_hint = "tools/mfa_engine 目录" if IS_WINDOWS else "pip install montreal-forced-aligner"
        return False, f"MFA 环境不可用，请检查 {platform_hint}"
    
    # 设置默认路径
    dict_path = dict_path or str(DEFAULT_DICT_PATH)
    model_path = model_path or str(DEFAULT_MODEL_PATH)
    
    # 临时目录：如果未指定，创建独立目录避免多用户冲突
    if temp_dir is None:
        session_id = uuid.uuid4().hex[:8]
        temp_dir = str(DEFAULT_TEMP_DIR / f"session_{session_id}")
    
    # 验证路径
    if not os.path.isdir(corpus_dir):
        return False, f"输入目录不存在: {corpus_dir}"
    if not os.path.isfile(dict_path):
        return False, f"字典文件不存在: {dict_path}"
    if not os.path.isfile(model_path):
        return False, f"声学模型不存在: {model_path}"
    
    # 清理字典文件中的空行（MFA 3.x 不支持空行）
    log(f"检查字典文件: {dict_path}")
    removed = _clean_dict_empty_lines(dict_path)
    if removed > 0:
        log(f"已清理字典文件中的 {removed} 个无效行")
    
    # 创建输出和临时目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 构造命令
    cmd = _get_mfa_command() + [
        "align",
        str(corpus_dir),
        str(dict_path),
        str(model_path),
        str(output_dir),
        "--temp_directory", str(temp_dir),
    ]
    
    # 设置并行进程数（默认使用 CPU 核心数，最少 1 个）
    import multiprocessing
    if num_jobs is None:
        num_jobs = max(1, multiprocessing.cpu_count())
    cmd.extend(["--num_jobs", str(num_jobs)])
    
    # Windows 外挂模式：启用多进程可能有兼容性问题，但可以尝试
    # 如果遇到问题，用户可以通过设置 num_jobs=1 来禁用
    # 注释掉原来的禁用逻辑，让 Windows 也能使用多进程
    # if IS_WINDOWS:
    #     cmd.extend(["--use_mp", "false"])
    
    if clean:
        cmd.append("--clean")
    if single_speaker:
        cmd.append("--single_speaker")
    
    log(f"正在启动 MFA 对齐引擎...")
    log(f"运行平台: {'Windows (外挂模式)' if IS_WINDOWS else 'Linux (系统安装)'}")
    log(f"并行进程数: {num_jobs}")
    log(f"输入目录: {corpus_dir}")
    log(f"输出目录: {output_dir}")
    
    try:
        env = _build_mfa_env(isolated_mfa_root)
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            log("MFA 对齐完成!")
            # 清理临时目录（仅清理会话独立目录）
            if "session_" in temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    log(f"已清理临时目录: {temp_dir}")
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {e}")
            return True, result.stdout
        else:
            error_msg = result.stderr or result.stdout or "未知错误"
            log(f"MFA 运行出错: {error_msg}")
            return False, error_msg
            
    except FileNotFoundError as e:
        msg = f"找不到 MFA 命令: {e}"
        log(msg)
        return False, msg
    except Exception as e:
        msg = f"MFA 执行异常: {e}"
        log(msg)
        return False, msg
    finally:
        # 确保临时目录被清理（即使出错）
        if "session_" in temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        # 清理会话独立的 MFA 数据目录
        if isolated_mfa_root:
            _cleanup_isolated_mfa_root(isolated_mfa_root)


def run_mfa_validate(
    corpus_dir: str,
    dict_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    验证语料库格式是否正确
    
    参数:
        corpus_dir: 语料库目录
        dict_path: 字典文件路径
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 输出信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if not check_mfa_available():
        return False, "MFA 环境不可用"
    
    dict_path = dict_path or str(DEFAULT_DICT_PATH)
    
    cmd = _get_mfa_command() + [
        "validate",
        str(corpus_dir),
        str(dict_path),
    ]
    
    log("正在验证语料库...")
    
    try:
        env = _build_mfa_env()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output = result.stdout + "\n" + result.stderr
        log("验证完成")
        return result.returncode == 0, output
        
    except Exception as e:
        return False, str(e)


def install_mfa_model(
    model_type: str,
    model_name: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    下载 MFA 预训练模型 (仅 Linux 支持)
    
    参数:
        model_type: 模型类型 ("acoustic" 或 "dictionary")
        model_name: 模型名称 (如 "mandarin_mfa", "mandarin_china_mfa")
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 输出信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if IS_WINDOWS:
        return False, "Windows 平台请手动下载模型文件"
    
    if not check_mfa_available():
        return False, "MFA 环境不可用"
    
    cmd = _get_mfa_command() + [
        "model", "download", model_type, model_name
    ]
    
    log(f"正在下载 MFA 模型: {model_type}/{model_name}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode == 0:
            log(f"模型下载完成: {model_name}")
            return True, result.stdout
        else:
            error_msg = result.stderr or result.stdout or "未知错误"
            log(f"模型下载失败: {error_msg}")
            return False, error_msg
            
    except Exception as e:
        return False, str(e)


def get_mfa_model_path(model_type: str, model_name: str) -> Optional[str]:
    """
    获取 MFA 模型路径
    Linux: 返回 MFA 内置模型名称 (mfa 会自动查找)
    Windows: 返回本地文件路径
    
    参数:
        model_type: 模型类型 ("acoustic" 或 "dictionary")
        model_name: 模型名称
    
    返回:
        模型路径或名称，不存在返回 None
    """
    if IS_WINDOWS:
        # Windows: 使用本地文件
        mfa_dir = BASE_DIR / "models" / "mfa"
        if model_type == "acoustic":
            path = mfa_dir / f"{model_name}.zip"
        else:
            path = mfa_dir / f"{model_name}.dict"
        return str(path) if path.exists() else None
    else:
        # Linux: 直接返回模型名称，mfa 会从缓存中查找
        return model_name
