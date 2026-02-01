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
            try:
                result = subprocess.run(
                    ["mfa", "version"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    logger.info(f"MFA 可用: {result.stdout.strip()}")
                    return True
                else:
                    logger.warning(f"MFA 命令执行失败: {result.stderr or result.stdout}")
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


def _build_mfa_env() -> dict:
    """构造 MFA 专用环境变量"""
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
        # Linux: 设置 pkuseg 模型目录（云端使用持久化路径）
        persistent_models = Path("/home/studio_service/models")
        if persistent_models.exists():
            pkuseg_home = persistent_models / "pkuseg"
            pkuseg_home.mkdir(parents=True, exist_ok=True)
            env["PKUSEG_HOME"] = str(pkuseg_home)
    
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


def run_mfa_alignment(
    corpus_dir: str,
    output_dir: str,
    dict_path: Optional[str] = None,
    model_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    single_speaker: bool = True,
    clean: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    执行 MFA 对齐
    
    参数:
        corpus_dir: 包含 wav 和 lab/txt 的输入目录
        output_dir: TextGrid 输出目录
        dict_path: 字典文件路径，默认使用 models/mandarin.dict
        model_path: 声学模型路径，默认使用 models/mandarin.zip
        temp_dir: 临时目录，默认使用 mfa_temp
        single_speaker: 是否为单说话人模式
        clean: 是否清理旧缓存
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 输出信息或错误信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    # 检查环境
    if not check_mfa_available():
        platform_hint = "tools/mfa_engine 目录" if IS_WINDOWS else "pip install montreal-forced-aligner"
        return False, f"MFA 环境不可用，请检查 {platform_hint}"
    
    # 设置默认路径
    dict_path = dict_path or str(DEFAULT_DICT_PATH)
    model_path = model_path or str(DEFAULT_MODEL_PATH)
    temp_dir = temp_dir or str(DEFAULT_TEMP_DIR)
    
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
    
    # Windows 禁用多进程避免问题，Linux 可以启用
    if IS_WINDOWS:
        cmd.extend(["--use_mp", "false"])
    
    if clean:
        cmd.append("--clean")
    if single_speaker:
        cmd.append("--single_speaker")
    
    log(f"正在启动 MFA 对齐引擎...")
    log(f"运行平台: {'Windows (外挂模式)' if IS_WINDOWS else 'Linux (系统安装)'}")
    log(f"输入目录: {corpus_dir}")
    log(f"输出目录: {output_dir}")
    
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
        
        if result.returncode == 0:
            log("MFA 对齐完成!")
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
