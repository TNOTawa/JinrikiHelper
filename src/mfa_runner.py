# -*- coding: utf-8 -*-
"""
MFA 外挂调用模块
采用 Sidecar Pattern，通过 subprocess 调用独立的 MFA 环境
"""

import os
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


def check_mfa_available() -> bool:
    """检查 MFA 外挂环境是否可用"""
    if not MFA_ENGINE_DIR.exists():
        logger.warning(f"MFA 引擎目录不存在: {MFA_ENGINE_DIR}")
        return False
    if not MFA_PYTHON.exists():
        logger.warning(f"MFA Python 不存在: {MFA_PYTHON}")
        return False
    return True


def _build_mfa_env() -> dict:
    """构造 MFA 专用环境变量"""
    env = os.environ.copy()
    
    # 必须把 Library\bin 加入 PATH，否则 Kaldi DLL 找不到
    mfa_paths = [
        str(MFA_ENGINE_DIR),
        str(MFA_ENGINE_DIR / "Library" / "bin"),
        str(MFA_ENGINE_DIR / "Scripts"),
        str(MFA_ENGINE_DIR / "bin"),
    ]
    env["PATH"] = ";".join(mfa_paths) + ";" + env.get("PATH", "")
    
    return env


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
        return False, "MFA 外挂环境不可用，请检查 tools/mfa_engine 目录"
    
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
    
    # 创建输出和临时目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 构造命令
    cmd = [
        str(MFA_PYTHON),
        "-m", "montreal_forced_aligner",
        "align",
        str(corpus_dir),
        str(dict_path),
        str(model_path),
        str(output_dir),
        "--temp_directory", str(temp_dir),
        "--use_mp", "false",  # 禁用多进程，避免Windows问题
    ]
    
    if clean:
        cmd.append("--clean")
    if single_speaker:
        cmd.append("--single_speaker")
    
    log(f"正在启动 MFA 对齐引擎...")
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
        msg = f"找不到 MFA Python: {e}"
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
        return False, "MFA 外挂环境不可用"
    
    dict_path = dict_path or str(DEFAULT_DICT_PATH)
    
    cmd = [
        str(MFA_PYTHON),
        "-m", "montreal_forced_aligner",
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
