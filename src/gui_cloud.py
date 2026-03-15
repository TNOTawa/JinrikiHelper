# -*- coding: utf-8 -*-
"""
人力V助手 (JinrikiHelper) 云端 Web UI
基于 Gradio 6.2.0 构建
专为云端部署优化：上传 → 处理 → 下载

作者：TNOT
"""

import gradio as gr
import logging
import os
import sys
import json
import tempfile
import zipfile
import shutil
import uuid
import threading
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class TaskToken:
    """任务令牌（用于取消/超时/状态跟踪）"""
    task_id: str
    session_id: str
    kind: str
    started_at: float
    last_touch: float
    cancel_event: threading.Event


class CloudTaskManager:
    """云端任务管理器：有界并发 + 会话取消 + 滞留回收"""

    def __init__(self, max_jobs: int, max_job_seconds: int):
        self.max_jobs = max(1, int(max_jobs))
        self.max_job_seconds = max(60, int(max_job_seconds))
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskToken] = {}
        self._session_index: Dict[str, set] = {}
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name="cloud-task-watchdog",
            daemon=True
        )
        self._watchdog.start()

    def start_task(self, kind: str, session_id: Optional[str]) -> Optional[TaskToken]:
        sid = (session_id or "").strip() or uuid.uuid4().hex[:8]
        now = time.time()
        with self._lock:
            active = len(self._tasks)
            if active >= self.max_jobs:
                return None

            task_id = uuid.uuid4().hex[:10]
            token = TaskToken(
                task_id=task_id,
                session_id=sid,
                kind=kind,
                started_at=now,
                last_touch=now,
                cancel_event=threading.Event()
            )
            self._tasks[task_id] = token
            self._session_index.setdefault(sid, set()).add(task_id)
            return token

    def finish_task(self, task_id: Optional[str]):
        if not task_id:
            return
        with self._lock:
            token = self._tasks.pop(task_id, None)
            if not token:
                return
            session_tasks = self._session_index.get(token.session_id)
            if session_tasks:
                session_tasks.discard(task_id)
                if not session_tasks:
                    self._session_index.pop(token.session_id, None)

    def touch(self, task_id: Optional[str]):
        if not task_id:
            return
        now = time.time()
        with self._lock:
            token = self._tasks.get(task_id)
            if token:
                token.last_touch = now

    def should_cancel(self, task_id: Optional[str]) -> bool:
        if not task_id:
            return False
        with self._lock:
            token = self._tasks.get(task_id)
            return bool(token and token.cancel_event.is_set())

    def cancel_session(self, session_id: Optional[str]) -> int:
        sid = (session_id or "").strip()
        if not sid:
            return 0
        cancelled = 0
        with self._lock:
            for task_id in list(self._session_index.get(sid, set())):
                token = self._tasks.get(task_id)
                if token and not token.cancel_event.is_set():
                    token.cancel_event.set()
                    cancelled += 1
        return cancelled

    def get_status_text(self) -> str:
        with self._lock:
            active = len(self._tasks)
            running = [f"{t.kind}:{t.task_id}" for t in self._tasks.values()]
        if running:
            detail = " | ".join(running[:3])
            if len(running) > 3:
                detail += f" ...共{len(running)}个"
            return f"当前并发数：{active}/{self.max_jobs}\\n运行中：{detail}"
        return f"当前并发数：{active}/{self.max_jobs}"

    def _watchdog_loop(self):
        while True:
            time.sleep(5)
            now = time.time()
            with self._lock:
                for token in self._tasks.values():
                    runtime = now - token.started_at
                    if runtime > self.max_job_seconds and not token.cancel_event.is_set():
                        token.cancel_event.set()
                        logger.warning(
                            "任务超时，已标记取消: %s (%s, %.1fs)",
                            token.task_id, token.kind, runtime
                        )


def get_concurrency_status() -> str:
    """获取当前并发状态文本"""
    return TASK_MANAGER.get_status_text()


def sanitize_source_name(name: Optional[str]) -> Optional[str]:
    """净化用户输入的音源名，避免路径逃逸与非法文件名。"""
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", (name or "").strip())
    cleaned = cleaned.strip("._- ")
    if not cleaned:
        return None
    return cleaned[:64]


def safe_gradio_handler(func):
    """
    Gradio 处理函数的安全包装器
    
    捕获所有异常并返回友好的错误信息，避免 Gradio 显示默认的"错误"状态
    """
    import functools
    import traceback
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 记录完整的异常堆栈
            error_trace = traceback.format_exc()
            logger.error(f"处理函数 {func.__name__} 发生异常:\n{error_trace}")
            
            # 根据函数返回值数量返回错误信息
            # 检查函数的类型注解来确定返回值数量
            annotations = getattr(func, '__annotations__', {})
            return_type = annotations.get('return', None)
            
            error_msg = f"❌ 系统错误: {str(e)}"
            error_detail = f"异常类型: {type(e).__name__}\n详情: {str(e)}"
            
            # 根据函数名判断返回值数量
            if func.__name__ == 'process_make_voicebank':
                return error_msg, error_detail, None, None
            elif func.__name__ == 'process_export_voicebank':
                return error_msg, error_detail, None
            elif func.__name__ == 'collect_and_export':
                return error_msg, error_detail, None
            elif func.__name__ == 'process_mfa_realign':
                return error_msg, error_detail, None
            else:
                # 默认返回单个错误消息
                return error_msg
    
    return wrapper

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.absolute()


class CloudConfig:
    """云端配置"""
    
    # 临时工作目录
    TEMP_BASE = tempfile.gettempdir()
    
    # 模型目录（云端使用项目内目录）
    MODELS_DIR = str(BASE_DIR / "models")
    MFA_DIR = str(BASE_DIR / "models" / "mfa")
    
    # 支持的音频格式
    AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    
    # Whisper 模型选项（含速度说明）
    WHISPER_MODELS = {
        "whisper-small (快速，约4秒/句)": "openai/whisper-small",
        "whisper-medium (精准，约12秒/句)": "openai/whisper-medium"
    }
    
    # 语言选项
    LANGUAGES = ["chinese", "japanese"]

    # 并发与任务监控
    MAX_CONCURRENT_JOBS = int(os.environ.get("JINRIKI_MAX_CONCURRENT_JOBS", "2"))
    MAX_JOB_SECONDS = int(os.environ.get("JINRIKI_MAX_JOB_SECONDS", "1800"))


TASK_MANAGER = CloudTaskManager(
    max_jobs=CloudConfig.MAX_CONCURRENT_JOBS,
    max_job_seconds=CloudConfig.MAX_JOB_SECONDS
)


def create_temp_workspace() -> str:
    """创建临时工作空间"""
    workspace_id = str(uuid.uuid4())[:8]
    workspace = os.path.join(CloudConfig.TEMP_BASE, f"jinriki_{workspace_id}")
    os.makedirs(workspace, exist_ok=True)
    return workspace


def cleanup_workspace(workspace: str):
    """清理工作空间"""
    if workspace and os.path.exists(workspace):
        try:
            shutil.rmtree(workspace)
            logger.info(f"已清理工作空间: {workspace}")
        except Exception as e:
            logger.warning(f"清理工作空间失败: {e}")


def create_zip(source_dir: str, zip_name: str) -> Optional[str]:
    """打包目录为 zip（使用 uuid 避免多用户冲突）"""
    if not os.path.isdir(source_dir):
        logger.warning(f"打包失败: 目录不存在 {source_dir}")
        return None
    try:
        unique_id = str(uuid.uuid4())[:8]
        zip_path = os.path.join(CloudConfig.TEMP_BASE, f"{zip_name}_{unique_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            file_count = 0
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)
                    file_count += 1
        logger.info(f"打包完成: {zip_path} ({file_count} 个文件)")
        return zip_path
    except Exception as e:
        logger.error(f"打包失败: {e}", exc_info=True)
        return None


def extract_zip(zip_path: str, target_dir: str) -> Tuple[bool, str]:
    """安全解压 zip 文件（防 Zip Slip）"""
    try:
        target_real = os.path.realpath(target_dir)
        os.makedirs(target_real, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                member = info.filename.replace('\\\\', '/')
                if member.startswith('/') or member.startswith('..'):
                    return False, f"解压失败: 非法路径 {member}"

                if '..' in Path(member).parts:
                    return False, f"解压失败: 非法路径 {member}"

                out_path = os.path.realpath(os.path.join(target_real, member))
                if out_path != target_real and not out_path.startswith(target_real + os.sep):
                    return False, f"解压失败: 路径越界 {member}"

                if info.is_dir():
                    os.makedirs(out_path, exist_ok=True)
                    continue

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with zf.open(info, 'r') as src, open(out_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

        return True, "解压成功"
    except Exception as e:
        return False, f"解压失败: {e}"


def scan_mfa_models() -> Dict[str, List[str]]:
    """扫描 MFA 模型"""
    result = {"acoustic": [], "dictionary": []}
    if not os.path.exists(CloudConfig.MFA_DIR):
        return result
    for f in os.listdir(CloudConfig.MFA_DIR):
        if f.endswith('.zip'):
            result["acoustic"].append(f)
        elif f.endswith('.dict') or f.endswith('.txt'):
            result["dictionary"].append(f)
    return result


def check_mfa_available() -> bool:
    """检查 MFA 是否可用"""
    from src.mfa_runner import check_mfa_available as _check
    return _check()


def cancel_current_session_tasks(session_id: Optional[str]) -> str:
    """取消当前会话的运行中任务"""
    cancelled = TASK_MANAGER.cancel_session(session_id)
    if cancelled <= 0:
        return "⚠️ 当前会话没有可取消任务"
    return f"🛑 已请求取消 {cancelled} 个任务"


# ==================== 制作音源功能 ====================

def get_audio_duration(file_path: str) -> Optional[float]:
    """
    获取音频文件时长（秒）
    
    返回: 时长秒数，失败返回 None
    """
    try:
        import wave
        import contextlib
        
        # 对于 WAV 文件，使用 wave 模块快速获取时长
        if file_path.lower().endswith('.wav'):
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                return frames / float(rate)
        
        # 对于其他格式，使用 pydub（如果可用）
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # 毫秒转秒
        except ImportError:
            # pydub 不可用，尝试使用 librosa
            try:
                import librosa
                duration = librosa.get_duration(path=file_path)
                return duration
            except ImportError:
                logger.warning(f"无法获取音频时长，缺少 pydub 或 librosa: {file_path}")
                return None
    except Exception as e:
        logger.warning(f"获取音频时长失败 {file_path}: {e}")
        return None


# 云端音频时长限制（秒）
MAX_AUDIO_DURATION_SECONDS = 600  # 10分钟


def validate_audio_upload(files) -> Tuple[bool, str, List[str]]:
    """
    验证上传的音频文件
    
    返回: (是否有效, 消息, 文件路径列表)
    """
    if not files:
        return False, "请上传音频文件", []
    
    valid_files = []
    for f in files:
        if hasattr(f, 'name'):
            path = f.name
        else:
            path = str(f)
        
        if path.lower().endswith(CloudConfig.AUDIO_EXTENSIONS):
            valid_files.append(path)
    
    if not valid_files:
        return False, f"未找到有效音频文件，支持格式: {', '.join(CloudConfig.AUDIO_EXTENSIONS)}", []
    
    return True, f"找到 {len(valid_files)} 个音频文件", valid_files


def validate_audio_duration(file_paths: List[str]) -> Tuple[bool, str, List[str]]:
    """
    验证音频文件时长，过滤超时文件
    
    返回: (是否全部通过, 消息, 有效文件列表)
    """
    valid_files = []
    rejected_files = []
    max_minutes = MAX_AUDIO_DURATION_SECONDS / 60
    
    for path in file_paths:
        duration = get_audio_duration(path)
        filename = os.path.basename(path)
        
        if duration is None:
            # 无法获取时长，允许通过（后续处理可能会失败）
            valid_files.append(path)
            logger.warning(f"无法检测时长，允许通过: {filename}")
        elif duration > MAX_AUDIO_DURATION_SECONDS:
            duration_min = duration / 60
            rejected_files.append(f"{filename} ({duration_min:.1f}分钟)")
        else:
            valid_files.append(path)
    
    if rejected_files:
        if not valid_files:
            # 全部被拒绝
            return False, f"所有音频超过{max_minutes:.0f}分钟限制: {', '.join(rejected_files)}", []
        else:
            # 部分被拒绝
            msg = f"已过滤 {len(rejected_files)} 个超时音频（>{max_minutes:.0f}分钟）: {', '.join(rejected_files[:3])}"
            if len(rejected_files) > 3:
                msg += f" 等{len(rejected_files)}个"
            return True, msg, valid_files
    
    return True, "", valid_files


@safe_gradio_handler
def process_make_voicebank(
    audio_files,
    source_name: str,
    language: str,
    whisper_model: str,
    session_id: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    制作音源：上传音频 → VAD切片 → Whisper转录 → MFA对齐 → 打包下载
    
    返回: (状态, 日志, 下载文件路径, 会话存储的音源包路径)
    """
    task = TASK_MANAGER.start_task("make", session_id)
    if not task:
        return f"❌ 服务繁忙，当前并发已达上限（{CloudConfig.MAX_CONCURRENT_JOBS}）", "", None, None
    logs = []
    workspace = None

    def log(msg):
        logs.append(msg)
        logger.info(msg)
        TASK_MANAGER.touch(task.task_id)

    def is_cancelled() -> bool:
        return TASK_MANAGER.should_cancel(task.task_id)

    try:
        try:
            # 导入依赖（放在 try 块内以捕获导入错误）
            from src.pipeline import PipelineConfig, VoiceBankPipeline
        except Exception as e:
            logger.error(f"导入模块失败: {e}", exc_info=True)
            return f"❌ 系统错误: 模块加载失败", str(e), None, None

        # 验证输入
        sanitized_name = sanitize_source_name(source_name)
        if not sanitized_name:
            return "❌ 请输入音源名称", "", None, None
        source_name = sanitized_name

        valid, msg, file_paths = validate_audio_upload(audio_files)
        if not valid:
            return f"❌ {msg}", "", None, None

        log(f"📁 {msg}")

        # 检查音频时长限制
        valid, duration_msg, file_paths = validate_audio_duration(file_paths)
        if not valid:
            return f"❌ {duration_msg}", "", None, None

        if duration_msg:
            log(f"⚠️ {duration_msg}")

        # 创建临时工作空间
        workspace = create_temp_workspace()
        log(f"🔧 创建工作空间: {workspace}")

        if is_cancelled():
            return "⚠️ 任务已取消", "任务在开始处理前被取消", None, None

        # 准备输入目录
        input_dir = os.path.join(workspace, "input")
        bank_dir = os.path.join(workspace, "bank")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(bank_dir, exist_ok=True)
        
        # 复制音频文件到输入目录
        progress(0.05, desc="复制音频文件...")
        copied_count = 0
        for src_path in file_paths:
            if is_cancelled():
                return "⚠️ 任务已取消", "\n".join(logs), None, None
            # 检查源文件是否存在
            if not os.path.exists(src_path):
                log(f"⚠️ 文件不存在或已被清理: {src_path}")
                continue
            try:
                dst_path = os.path.join(input_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                log(f"⚠️ 复制文件失败 {os.path.basename(src_path)}: {e}")
        
        if copied_count == 0:
            return "❌ 无法访问上传的文件，请重新上传", "\n".join(logs), None, None
        
        log(f"📋 已复制 {copied_count}/{len(file_paths)} 个文件到工作目录")
        
        # 获取 MFA 模型路径
        mfa_models = scan_mfa_models()
        dict_path = None
        acoustic_path = None
        
        if mfa_models["dictionary"]:
            # 根据语言选择字典
            for d in mfa_models["dictionary"]:
                if language == "japanese" and "japanese" in d.lower():
                    dict_path = os.path.join(CloudConfig.MFA_DIR, d)
                    break
                elif language == "chinese" and "mandarin" in d.lower():
                    dict_path = os.path.join(CloudConfig.MFA_DIR, d)
                    break
            if not dict_path:
                dict_path = os.path.join(CloudConfig.MFA_DIR, mfa_models["dictionary"][0])
        
        if mfa_models["acoustic"]:
            for a in mfa_models["acoustic"]:
                if language == "japanese" and "japanese" in a.lower():
                    acoustic_path = os.path.join(CloudConfig.MFA_DIR, a)
                    break
                elif language == "chinese" and "mandarin" in a.lower():
                    acoustic_path = os.path.join(CloudConfig.MFA_DIR, a)
                    break
            if not acoustic_path:
                acoustic_path = os.path.join(CloudConfig.MFA_DIR, mfa_models["acoustic"][0])
        
        # 配置流水线
        whisper_model_name = CloudConfig.WHISPER_MODELS.get(whisper_model, "openai/whisper-small")
        
        config = PipelineConfig(
            source_name=source_name,
            input_path=input_dir,
            output_base_dir=bank_dir,
            models_dir=CloudConfig.MODELS_DIR,
            whisper_model=whisper_model_name,
            mfa_dict_path=dict_path,
            mfa_model_path=acoustic_path,
            language=language,
            cancel_checker=is_cancelled
        )
        
        pipeline = VoiceBankPipeline(config, log)
        
        # 步骤0: VAD切片 + Whisper转录
        progress(0.1, desc="VAD切片 + Whisper转录...")
        log("\n" + "=" * 50)
        log("【步骤1】VAD切片 + Whisper转录")
        success, msg, slices = pipeline.step0_preprocess()
        if not success:
            return f"❌ 预处理失败: {msg}", "\n".join(logs), None, None
        log(f"✅ {msg}")

        if is_cancelled():
            return "⚠️ 任务已取消", "\n".join(logs), None, None
        
        # 步骤1: MFA对齐
        progress(0.6, desc="MFA语音对齐...")
        log("\n" + "=" * 50)
        log("【步骤2】MFA语音对齐")
        
        mfa_success = False
        if check_mfa_available():
            success, msg = pipeline.step1_mfa_align()
            if not success:
                log(f"⚠️ MFA对齐失败: {msg}")
                log("继续导出（无TextGrid）...")
            else:
                log(f"✅ {msg}")
                mfa_success = True
        else:
            log("⚠️ MFA不可用，跳过对齐步骤")
        
        # MFA 不可用时的提示
        if not mfa_success:
            log("")
            log("💡 提示：音源包中缺少 TextGrid 对齐数据")
            log("   如需导出 UTAU oto.ini，请前往「MFA补对齐」页面进行修复")
        
        # 打包结果
        progress(0.9, desc="打包结果...")
        log("\n" + "=" * 50)
        log("【打包结果】")
        
        source_dir = os.path.join(bank_dir, source_name)
        zip_name = f"{source_name}_音源数据"
        zip_path = create_zip(source_dir, zip_name)
        
        if zip_path:
            log(f"📦 已打包: {os.path.basename(zip_path)}")
            progress(1.0, desc="完成")
            # 返回路径到会话状态，供导出页面使用
            return "✅ 音源制作完成", "\n".join(logs), zip_path, zip_path
        else:
            return "❌ 打包失败", "\n".join(logs), None, None
        
    except Exception as e:
        logger.error(f"制作音源失败: {e}", exc_info=True)
        return f"❌ 处理失败: {e}", "\n".join(logs), None, None
    finally:
        cleanup_workspace(workspace)
        TASK_MANAGER.finish_task(task.task_id)


# ==================== 导出音源功能 ====================

def get_last_made_voicebank(session_voicebank: Optional[str]) -> Tuple[Optional[str], str]:
    """
    获取当前会话制作的音源包
    
    参数:
        session_voicebank: 会话状态中存储的音源包路径
    
    返回: (文件路径, 信息消息)
    """
    if session_voicebank and os.path.exists(session_voicebank):
        valid, msg, name = validate_voicebank_zip_path(session_voicebank)
        if valid:
            return session_voicebank, f"✅ 已选择刚制作的音源: {name}"
    return None, "❌ 没有找到刚制作的音源，请先在「制作音源」页面制作，或手动上传"


def validate_voicebank_zip_path(zip_path: str) -> Tuple[bool, str, Optional[str]]:
    """
    验证音源压缩包路径
    
    返回: (是否有效, 消息, 音源名称)
    """
    if not zip_path or not os.path.exists(zip_path):
        return False, "文件不存在", None
    
    if not zip_path.lower().endswith('.zip'):
        return False, "请上传 .zip 格式的压缩包", None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            
            has_wav = any(n.endswith('.wav') for n in names)
            has_slices = any('slices/' in n for n in names)
            has_textgrid = any('textgrid/' in n for n in names)
            
            if not has_wav:
                return False, "压缩包中未找到 .wav 音频文件", None
            
            source_name = None
            if 'meta.json' in names:
                try:
                    with zf.open('meta.json') as mf:
                        meta = json.load(mf)
                        source_name = meta.get('source_name')
                except:
                    pass
            
            if not source_name:
                source_name = Path(zip_path).stem.replace('_音源数据', '')
            source_name = sanitize_source_name(source_name) or "voicebank"
            
            info_parts = []
            if has_slices:
                wav_count = len([n for n in names if 'slices/' in n and n.endswith('.wav')])
                info_parts.append(f"切片: {wav_count} 个")
            if has_textgrid:
                tg_count = len([n for n in names if 'textgrid/' in n and n.endswith('.TextGrid')])
                info_parts.append(f"TextGrid: {tg_count} 个")
            
            info = " | ".join(info_parts) if info_parts else "有效的音源包"
            
            return True, f"✅ {info}", source_name
            
    except zipfile.BadZipFile:
        return False, "无效的 zip 文件", None
    except Exception as e:
        return False, f"验证失败: {e}", None


def validate_voicebank_zip(zip_file) -> Tuple[bool, str, Optional[str]]:
    """
    验证上传的音源压缩包
    
    返回: (是否有效, 消息, 音源名称)
    """
    if not zip_file:
        return False, "请上传音源压缩包", None
    
    zip_path = zip_file.name if hasattr(zip_file, 'name') else str(zip_file)
    return validate_voicebank_zip_path(zip_path)


@safe_gradio_handler
def process_export_voicebank(
    zip_file,
    plugin_name: str,
    options_json: str,
    session_id: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    导出音源：上传音源包 → 解压 → 导出 → 打包下载
    
    参数:
        zip_file: 上传的音源压缩包
        plugin_name: 导出插件名称
        options_json: JSON 格式的插件选项
    
    返回: (状态, 日志, 下载文件路径)
    """
    task = TASK_MANAGER.start_task("export", session_id)
    if not task:
        return f"❌ 服务繁忙，当前并发已达上限（{CloudConfig.MAX_CONCURRENT_JOBS}）", "", None
    logs = []
    workspace = None

    def log(msg):
        logs.append(msg)
        logger.info(msg)
        TASK_MANAGER.touch(task.task_id)

    def is_cancelled() -> bool:
        return TASK_MANAGER.should_cancel(task.task_id)

    try:
        # 验证输入
        valid, msg, source_name = validate_voicebank_zip(zip_file)
        if not valid:
            return f"❌ {msg}", "", None

        source_name = sanitize_source_name(source_name or "") or "voicebank"

        log(f"📦 {msg}")
        log(f"📝 音源名称: {source_name}")

        # 解析选项
        try:
            options = json.loads(options_json) if options_json else {}
        except json.JSONDecodeError:
            options = {}

        # 创建临时工作空间
        workspace = create_temp_workspace()
        log(f"🔧 创建工作空间")

        if is_cancelled():
            return "⚠️ 任务已取消", "任务在开始处理前被取消", None

        zip_path = zip_file.name if hasattr(zip_file, 'name') else str(zip_file)
        
        # 解压音源包
        progress(0.1, desc="解压音源包...")
        bank_dir = os.path.join(workspace, "bank")
        source_dir = os.path.join(bank_dir, source_name)
        os.makedirs(source_dir, exist_ok=True)
        
        success, msg = extract_zip(zip_path, source_dir)
        if not success:
            return f"❌ {msg}", "\n".join(logs), None
        log(f"📂 已解压到工作目录")
        
        # 检查目录结构，处理可能的嵌套
        slices_dir = os.path.join(source_dir, "slices")
        if not os.path.exists(slices_dir):
            # 可能解压后有额外的一层目录
            subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
            if len(subdirs) == 1:
                nested_dir = os.path.join(source_dir, subdirs[0])
                if os.path.exists(os.path.join(nested_dir, "slices")):
                    # 移动内容到上层
                    for item in os.listdir(nested_dir):
                        shutil.move(os.path.join(nested_dir, item), source_dir)
                    os.rmdir(nested_dir)
        
        # 执行导出
        progress(0.3, desc="执行导出...")
        log("\n" + "=" * 50)
        log(f"【{plugin_name}】")
        
        from src.export_plugins import load_plugins
        plugins = load_plugins()
        
        if plugin_name not in plugins:
            return f"❌ 未找到插件: {plugin_name}", "\n".join(logs), None
        
        plugin = plugins[plugin_name]
        plugin.set_progress_callback(log)
        
        # 添加默认选项
        options["clean_temp"] = True
        
        success, msg = plugin.export(source_name, bank_dir, options)
        
        if not success:
            return f"❌ 导出失败: {msg}", "\n".join(logs), None
        
        log(f"✅ {msg}")
        
        # 打包导出结果
        progress(0.9, desc="打包结果...")
        log("\n" + "=" * 50)
        log("【打包结果】")
        
        # 根据插件类型确定导出目录和导出标识
        if "UTAU" in plugin_name:
            export_subdir = "utau_oto"
            export_id = "utau_oto_export"
        else:
            export_subdir = "simple_export"
            export_id = "simple_export"
        
        export_dir = os.path.join(workspace, "export", source_name, export_subdir)
        
        # 如果导出目录不存在，尝试其他位置
        if not os.path.exists(export_dir):
            alt_export = os.path.join(os.path.dirname(bank_dir), "export", source_name, export_subdir)
            if os.path.exists(alt_export):
                export_dir = alt_export
        
        # 再尝试另一个子目录
        if not os.path.exists(export_dir):
            other_subdir = "simple_export" if export_subdir == "utau_oto" else "utau_oto"
            other_id = "simple_export" if export_id == "utau_oto_export" else "utau_oto_export"
            export_dir = os.path.join(workspace, "export", source_name, other_subdir)
            if not os.path.exists(export_dir):
                alt_export = os.path.join(os.path.dirname(bank_dir), "export", source_name, other_subdir)
                if os.path.exists(alt_export):
                    export_dir = alt_export
                    export_id = other_id
            else:
                export_id = other_id
        
        if not os.path.exists(export_dir):
            return "❌ 未找到导出结果", "\n".join(logs), None

        if is_cancelled():
            return "⚠️ 任务已取消", "\n".join(logs), None
        
        # 命名格式: [音源名称]_[插件标识]
        zip_name = f"{source_name}_{export_id}"
        result_zip = create_zip(export_dir, zip_name)
        
        if result_zip:
            # 统计导出文件数
            file_count = len([f for f in os.listdir(export_dir) if f.endswith(('.wav', '.ini'))])
            log(f"📦 已打包: {file_count} 个文件")
            progress(1.0, desc="完成")
            return "✅ 导出完成", "\n".join(logs), result_zip
        else:
            return "❌ 打包失败", "\n".join(logs), None
        
    except Exception as e:
        logger.error(f"导出失败: {e}", exc_info=True)
        return f"❌ 处理失败: {e}", "\n".join(logs), None
    finally:
        cleanup_workspace(workspace)
        TASK_MANAGER.finish_task(task.task_id)


# ==================== MFA补对齐功能 ====================

def validate_mfa_voicebank(zip_file) -> str:
    """
    验证上传的音源包，检查 TextGrid 状态
    
    返回: 状态信息字符串
    """
    if not zip_file:
        return "⏳ 请上传音源压缩包"
    
    zip_path = zip_file.name if hasattr(zip_file, 'name') else str(zip_file)
    
    if not zip_path.lower().endswith('.zip'):
        return "❌ 请上传 .zip 格式的压缩包"
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            
            # 统计文件
            wav_files = [n for n in names if n.endswith('.wav') and 'slices/' in n]
            lab_files = [n for n in names if n.endswith('.lab') and 'slices/' in n]
            tg_files = [n for n in names if n.endswith('.TextGrid') and 'textgrid/' in n]
            
            if not wav_files:
                return "❌ 未找到 slices 目录下的 .wav 文件"
            
            if not lab_files:
                return "❌ 未找到 slices 目录下的 .lab 标注文件"
            
            # 获取音源名称
            source_name = None
            if 'meta.json' in names:
                try:
                    with zf.open('meta.json') as mf:
                        meta = json.load(mf)
                        source_name = meta.get('source_name')
                except:
                    pass
            
            if not source_name:
                source_name = Path(zip_path).stem.replace('_音源数据', '')
            
            # 构建状态信息
            info_parts = [f"📝 音源: {source_name}"]
            info_parts.append(f"🎵 切片: {len(wav_files)} 个 WAV")
            info_parts.append(f"📄 标注: {len(lab_files)} 个 LAB")
            
            if tg_files:
                coverage = len(tg_files) / len(wav_files) * 100 if wav_files else 0
                if coverage >= 100:
                    info_parts.append(f"✅ TextGrid: {len(tg_files)} 个 (已完整)")
                else:
                    info_parts.append(f"⚠️ TextGrid: {len(tg_files)} 个 ({coverage:.0f}% 覆盖)")
            else:
                info_parts.append("❌ TextGrid: 无 (需要对齐)")
            
            return " | ".join(info_parts)
            
    except zipfile.BadZipFile:
        return "❌ 无效的 zip 文件"
    except Exception as e:
        return f"❌ 验证失败: {e}"


@safe_gradio_handler
def process_mfa_realign(
    zip_file,
    language: str,
    session_id: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    MFA 补对齐：为缺少 TextGrid 的音源包执行 MFA 对齐
    
    参数:
        zip_file: 上传的音源压缩包
        language: 语言选择
    
    返回: (状态, 日志, 下载文件路径)
    """
    task = TASK_MANAGER.start_task("mfa", session_id)
    if not task:
        return f"❌ 服务繁忙，当前并发已达上限（{CloudConfig.MAX_CONCURRENT_JOBS}）", "", None
    logs = []
    workspace = None

    def log(msg):
        logs.append(msg)
        logger.info(msg)
        TASK_MANAGER.touch(task.task_id)

    def is_cancelled() -> bool:
        return TASK_MANAGER.should_cancel(task.task_id)

    try:
        # 验证输入
        if not zip_file:
            return "❌ 请上传音源压缩包", "", None

        zip_path = zip_file.name if hasattr(zip_file, 'name') else str(zip_file)

        # 检查 MFA 是否可用
        if not check_mfa_available():
            return "❌ MFA 不可用，无法执行对齐", "请检查 MFA 环境配置", None

        log("📦 开始处理音源包...")

        # 创建临时工作空间
        workspace = create_temp_workspace()
        log(f"🔧 创建工作空间")

        if is_cancelled():
            return "⚠️ 任务已取消", "任务在开始处理前被取消", None

        # 解压音源包
        progress(0.1, desc="解压音源包...")
        
        # 先检查压缩包结构，确定音源名称
        source_name = None
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            
            # 尝试从 meta.json 获取音源名称
            if 'meta.json' in names:
                try:
                    with zf.open('meta.json') as mf:
                        meta = json.load(mf)
                        source_name = meta.get('source_name')
                except:
                    pass
            
            if not source_name:
                source_name = Path(zip_path).stem.replace('_音源数据', '')

        source_name = sanitize_source_name(source_name) or "voicebank"
        
        log(f"📝 音源名称: {source_name}")
        
        # 解压到工作目录
        bank_dir = os.path.join(workspace, "bank")
        source_dir = os.path.join(bank_dir, source_name)
        os.makedirs(source_dir, exist_ok=True)
        
        success, msg = extract_zip(zip_path, source_dir)
        if not success:
            return f"❌ {msg}", "\n".join(logs), None
        
        log(f"📂 已解压到工作目录")
        
        # 检查目录结构，处理可能的嵌套
        slices_dir = os.path.join(source_dir, "slices")
        if not os.path.exists(slices_dir):
            # 可能解压后有额外的一层目录
            subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
            if len(subdirs) == 1:
                nested_dir = os.path.join(source_dir, subdirs[0])
                if os.path.exists(os.path.join(nested_dir, "slices")):
                    # 移动内容到上层
                    for item in os.listdir(nested_dir):
                        shutil.move(os.path.join(nested_dir, item), source_dir)
                    os.rmdir(nested_dir)
                    slices_dir = os.path.join(source_dir, "slices")
        
        if not os.path.exists(slices_dir):
            return "❌ 未找到 slices 目录", "\n".join(logs), None
        
        # 统计文件
        wav_files = [f for f in os.listdir(slices_dir) if f.endswith('.wav')]
        lab_files = [f for f in os.listdir(slices_dir) if f.endswith('.lab')]
        
        log(f"🎵 找到 {len(wav_files)} 个 WAV 文件")
        log(f"📄 找到 {len(lab_files)} 个 LAB 标注文件")
        
        if not wav_files:
            return "❌ slices 目录中没有 WAV 文件", "\n".join(logs), None
        
        if not lab_files:
            return "❌ slices 目录中没有 LAB 标注文件", "\n".join(logs), None
        
        # 获取 MFA 模型路径
        progress(0.2, desc="加载 MFA 模型...")
        mfa_models = scan_mfa_models()
        dict_path = None
        acoustic_path = None
        
        if mfa_models["dictionary"]:
            for d in mfa_models["dictionary"]:
                if language == "japanese" and "japanese" in d.lower():
                    dict_path = os.path.join(CloudConfig.MFA_DIR, d)
                    break
                elif language == "chinese" and "mandarin" in d.lower():
                    dict_path = os.path.join(CloudConfig.MFA_DIR, d)
                    break
            if not dict_path:
                dict_path = os.path.join(CloudConfig.MFA_DIR, mfa_models["dictionary"][0])
        
        if mfa_models["acoustic"]:
            for a in mfa_models["acoustic"]:
                if language == "japanese" and "japanese" in a.lower():
                    acoustic_path = os.path.join(CloudConfig.MFA_DIR, a)
                    break
                elif language == "chinese" and "mandarin" in a.lower():
                    acoustic_path = os.path.join(CloudConfig.MFA_DIR, a)
                    break
            if not acoustic_path:
                acoustic_path = os.path.join(CloudConfig.MFA_DIR, mfa_models["acoustic"][0])
        
        if not dict_path or not os.path.exists(dict_path):
            return f"❌ 未找到 {language} 语言的字典文件", "\n".join(logs), None
        
        if not acoustic_path or not os.path.exists(acoustic_path):
            return f"❌ 未找到 {language} 语言的声学模型", "\n".join(logs), None
        
        log(f"📚 字典: {os.path.basename(dict_path)}")
        log(f"🔊 声学模型: {os.path.basename(acoustic_path)}")
        
        # 执行 MFA 对齐
        progress(0.3, desc="执行 MFA 对齐...")
        log("\n" + "=" * 50)
        log("【MFA 强制对齐】")
        
        textgrid_dir = os.path.join(source_dir, "textgrid")
        os.makedirs(textgrid_dir, exist_ok=True)
        
        from src.mfa_runner import run_mfa_alignment
        
        success, mfa_msg = run_mfa_alignment(
            corpus_dir=slices_dir,
            output_dir=textgrid_dir,
            dict_path=dict_path,
            model_path=acoustic_path,
            single_speaker=True,
            clean=True,
            progress_callback=log,
            cancel_checker=is_cancelled
        )
        
        if not success:
            log(f"❌ MFA 对齐失败: {mfa_msg}")
            return "❌ MFA 对齐失败", "\n".join(logs), None
        
        log("✅ MFA 对齐完成")
        
        # 统计生成的 TextGrid 文件
        tg_files = [f for f in os.listdir(textgrid_dir) if f.endswith('.TextGrid')]
        log(f"📊 生成 {len(tg_files)} 个 TextGrid 文件")
        
        if not tg_files:
            return "❌ 未生成任何 TextGrid 文件", "\n".join(logs), None
        
        # 更新 meta.json
        meta_path = os.path.join(source_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['textgrid_count'] = len(tg_files)
                meta['mfa_realigned'] = True
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                log("📝 已更新 meta.json")
            except Exception as e:
                log(f"⚠️ 更新 meta.json 失败: {e}")
        
        # 打包结果
        progress(0.9, desc="打包结果...")
        log("\n" + "=" * 50)
        log("【打包结果】")
        
        zip_name = f"{source_name}_音源数据_已对齐"
        result_zip = create_zip(source_dir, zip_name)
        
        if result_zip:
            log(f"📦 已打包: {os.path.basename(result_zip)}")
            progress(1.0, desc="完成")
            return "✅ MFA 补对齐完成", "\n".join(logs), result_zip
        else:
            return "❌ 打包失败", "\n".join(logs), None
        
    except Exception as e:
        logger.error(f"MFA 补对齐失败: {e}", exc_info=True)
        return f"❌ 处理失败: {e}", "\n".join(logs), None
    finally:
        cleanup_workspace(workspace)
        TASK_MANAGER.finish_task(task.task_id)


# ==================== 插件选项 UI 生成 ====================

def get_plugin_options_config(plugins: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    获取所有插件的选项配置
    
    返回: {插件名: [选项配置列表]}
    """
    from src.export_plugins.base import OptionType
    
    config = {}
    for name, plugin in plugins.items():
        options = []
        for opt in plugin.get_options():
            opt_config = {
                "key": opt.key,
                "label": opt.label,
                "type": opt.option_type.value,
                "default": opt.default,
                "description": opt.description,
                "choices": opt.choices,
                "min_value": opt.min_value,
                "max_value": opt.max_value,
                "step": opt.step,
            }
            options.append(opt_config)
        config[name] = options
    return config


def get_default_options_json(plugin_name: str, plugins_config: Dict) -> str:
    """获取插件的默认选项 JSON"""
    if plugin_name not in plugins_config:
        return "{}"
    
    options = plugins_config[plugin_name]
    defaults = {}
    for opt in options:
        if opt["type"] != "label":
            defaults[opt["key"]] = opt["default"]
    
    return json.dumps(defaults, ensure_ascii=False)


def create_dynamic_plugin_options(plugins: Dict[str, Any], plugins_config: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    创建动态插件选项组件
    
    返回:
        (组件字典, 收集选项函数)
        
    组件字典结构: {
        "container": gr.Column,  # 主容器
        "groups": {插件名: gr.Group},  # 每个插件的选项组
        "components": {插件名: {选项key: 组件}},  # 所有组件
    }
    """
    from src.export_plugins.base import OptionType
    
    all_groups = {}
    all_components = {}
    
    # 为每个插件创建选项组
    for plugin_name, options in plugins_config.items():
        plugin_components = {}
        
        # 创建该插件的选项组（初始隐藏，第一个插件除外）
        is_first = (plugin_name == list(plugins_config.keys())[0])
        
        with gr.Group(visible=is_first) as plugin_group:
            # 显示插件描述
            if plugin_name in plugins:
                gr.Markdown(f"> {plugins[plugin_name].description}")
            
            for opt in options:
                opt_type = opt["type"]
                key = opt["key"]
                label = opt["label"]
                default = opt["default"]
                description = opt.get("description", "")
                choices = opt.get("choices", [])
                min_val = opt.get("min_value")
                max_val = opt.get("max_value")
                step = opt.get("step")
                
                # 根据类型创建对应的 Gradio 组件
                if opt_type == "label":
                    # 纯文本标签
                    gr.Markdown(f"*{label}*")
                    continue
                    
                elif opt_type == "text":
                    component = gr.Textbox(
                        label=label,
                        value=default or "",
                        info=description
                    )
                    
                elif opt_type == "number":
                    component = gr.Number(
                        label=label,
                        value=default if default is not None else 0,
                        minimum=min_val,
                        maximum=max_val,
                        step=step or 1,
                        info=description
                    )
                    
                elif opt_type == "switch":
                    component = gr.Checkbox(
                        label=label,
                        value=bool(default),
                        info=description
                    )
                    
                elif opt_type == "combo":
                    component = gr.Dropdown(
                        label=label,
                        choices=choices,
                        value=default if default in choices else (choices[0] if choices else None),
                        info=description
                    )
                    
                elif opt_type == "multi_select":
                    component = gr.CheckboxGroup(
                        label=label,
                        choices=choices,
                        value=default if isinstance(default, list) else [],
                        info=description
                    )
                    
                else:
                    # 未知类型，使用文本框
                    component = gr.Textbox(
                        label=label,
                        value=str(default) if default else "",
                        info=description
                    )
                
                plugin_components[key] = component
        
        all_groups[plugin_name] = plugin_group
        all_components[plugin_name] = plugin_components
    
    return all_groups, all_components


def build_options_collector(plugins_config: Dict, all_components: Dict):
    """
    构建选项收集函数
    
    返回一个函数，该函数接收插件名和所有组件值，返回选项字典
    """
    # 构建组件到选项的映射
    component_keys = {}
    for plugin_name, components in all_components.items():
        component_keys[plugin_name] = list(components.keys())
    
    def collect_options(plugin_name: str, *values) -> Dict[str, Any]:
        """收集当前插件的选项值"""
        if plugin_name not in component_keys:
            return {}
        
        keys = component_keys[plugin_name]
        options = {}
        
        # 计算当前插件的值在 values 中的起始位置
        start_idx = 0
        for pname in component_keys:
            if pname == plugin_name:
                break
            start_idx += len(component_keys[pname])
        
        # 提取当前插件的值
        for i, key in enumerate(keys):
            if start_idx + i < len(values):
                options[key] = values[start_idx + i]
        
        return options
    
    return collect_options


# ==================== 构建界面 ====================

def create_cloud_ui():
    """创建云端 Gradio 界面"""
    
    # 检查 MFA 状态
    mfa_available = check_mfa_available()
    mfa_status = "✅ MFA 已就绪" if mfa_available else "⚠️ MFA 不可用（将跳过对齐步骤）"
    
    # 加载导出插件
    from src.export_plugins import load_plugins
    plugins = load_plugins()
    plugin_names = list(plugins.keys()) if plugins else ["简单单字导出"]
    
    with gr.Blocks(
        title="人力V助手 (JinrikiHelper)",
        theme=gr.themes.Soft()
    ) as app:
        
        session_id_state = gr.State(value=uuid.uuid4().hex[:8])
        # 会话状态：存储当前用户制作的音源包路径
        session_voicebank = gr.State(value=None)
        
        # 标题行：左侧标题 + 右侧并发状态
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# 🎤 人力V助手 (JinrikiHelper)")
            with gr.Column(scale=1, min_width=200):
                concurrency_display = gr.Markdown(
                    value=get_concurrency_status(),
                    elem_id="concurrency-status"
                )
                concurrency_timer = gr.Timer(2.0)

        with gr.Row():
            cancel_jobs_btn = gr.Button("🛑 取消当前会话任务", variant="stop")
            cancel_jobs_msg = gr.Textbox(label="任务控制", interactive=False, value="")
        
        gr.Markdown("语音数据集处理工具 - 自动化制作语音音源库")
        gr.Markdown("> ☁️ 云端版：上传音频 → 自动处理 → 下载结果")
        gr.Markdown("""
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-left: 4px solid #28a745; padding: 10px 14px; border-radius: 8px; margin: 8px 0;">
            <p style="margin: 0; color: #155724; line-height: 1.5;">
                <strong>📊 处理状态说明：</strong>只有当<strong>进度条弹出并开始滚动</strong>时，才表示正在处理中。<br/>
                如果点击按钮后长时间没有进度条出现，可能是正在排队等待，或遇到了问题。
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # ==================== 制作音源页 ====================
            with gr.Tab("🎵 制作音源"):
                gr.Markdown("""
                <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
                    <p style="margin: 0 0 8px 0; font-weight: bold; color: #856404;">⚠️ 温馨提示</p>
                    <p style="margin: 0; color: #856404; line-height: 1.6;">
                        <strong style="font-size: 1.1em;">音频质量 >> 音频数量！</strong><br/>
                        请控制上传音频的数量！经测试，<strong>8 分钟以内的高质量音频已经非常充足</strong>。<br/>
                        上传过多音频可能导致混入低质量样本，同时也会占用服务器并发资源。<br/>
                        建议大量音频先人工筛选后再上传，感谢配合！🙏
                    </p>
                </div>
                """)
                
                gr.Markdown("### 上传音频文件")
                gr.Markdown("支持格式: WAV, MP3, FLAC, OGG, M4A")
                gr.Markdown("允许同时拖拽多个文件上传，也可点击上传框的右上角追加文件")
                
                audio_upload = gr.File(
                    label="上传音频文件",
                    file_count="multiple",
                    file_types=["audio"]
                )
                
                # 上传状态提示
                upload_status = gr.Textbox(
                    label="上传状态",
                    value="⏳ 请上传音频文件",
                    interactive=False
                )
                
                with gr.Row():
                    make_source_name = gr.Textbox(
                        label="音源名称",
                        placeholder="my_voice",
                        info="用于标识输出的音源包"
                    )
                    make_language = gr.Dropdown(
                        choices=CloudConfig.LANGUAGES,
                        value="chinese",
                        label="语言"
                    )
                
                with gr.Row():
                    make_whisper = gr.Dropdown(
                        choices=list(CloudConfig.WHISPER_MODELS.keys()),
                        value=list(CloudConfig.WHISPER_MODELS.keys())[0],
                        label="Whisper 模型"
                    )
                    make_mfa_status = gr.Textbox(
                        label="MFA 状态",
                        value=mfa_status,
                        interactive=False
                    )
                
                gr.Markdown("""
                > ⏱️ **模型速度参考**：small 约 4 秒/句，medium 约 12 秒/句（medium 慢 2-3 倍但更准确）
                > 
                > <span style="color: red;">**small完全够用的，medium费时还容易炸空间，除非实在识别不出来字再用**</span>
                """)
                
                make_btn = gr.Button("🚀 开始制作", variant="primary", size="lg", interactive=False)
                
                # 时长估算显示
                time_estimate = gr.Markdown(
                    value="",
                    visible=False
                )
                
                make_status = gr.Textbox(label="状态", interactive=False)
                make_log = gr.Textbox(label="处理日志", lines=12, interactive=False)
                
                gr.Markdown("### 下载结果")
                make_download = gr.File(label="音源包下载", interactive=False)
                
                gr.Markdown("""
                > 💡 处理流程：
                > 1. VAD 语音活动检测，自动切分音频
                > 2. Whisper 语音识别，生成文本标注
                > 3. MFA 强制对齐，生成音素级时间标注
                > 4. 打包为 zip 供下载
                """)
                
                # 音频上传状态检测
                def check_audio_upload(files):
                    """检查音频上传状态，返回状态文本、按钮可用性和时长估算"""
                    if not files:
                        return "⏳ 请上传音频文件", gr.update(interactive=False), gr.update(value="", visible=False)
                    
                    valid_count = 0
                    total_duration = 0.0
                    for f in files:
                        path = f.name if hasattr(f, 'name') else str(f)
                        if path.lower().endswith(CloudConfig.AUDIO_EXTENSIONS):
                            valid_count += 1
                            # 计算时长
                            duration = get_audio_duration(path)
                            if duration:
                                total_duration += duration
                    
                    if valid_count == 0:
                        return f"❌ 未找到有效音频，支持: {', '.join(CloudConfig.AUDIO_EXTENSIONS)}", gr.update(interactive=False), gr.update(value="", visible=False)
                    
                    # 格式化总时长
                    total_minutes = int(total_duration // 60)
                    total_seconds = int(total_duration % 60)
                    duration_str = f"{total_minutes}分{total_seconds}秒" if total_minutes > 0 else f"{total_seconds}秒"
                    
                    # 计算预估处理时间
                    # 根据实测数据：1分钟音频约产生79个切片，每个切片处理约3.9秒
                    # 即每分钟音频需要约 79 * 3.9 / 60 ≈ 5.1 分钟处理时间
                    PROCESS_TIME_RATIO = 5.1  # 处理时间与音频时长的比例
                    estimated_seconds = total_duration * PROCESS_TIME_RATIO
                    est_minutes = int(estimated_seconds // 60)
                    est_seconds = int(estimated_seconds % 60)
                    
                    if est_minutes >= 60:
                        est_hours = est_minutes // 60
                        est_minutes = est_minutes % 60
                        estimate_str = f"{est_hours}小时{est_minutes}分钟"
                    elif est_minutes > 0:
                        estimate_str = f"{est_minutes}分{est_seconds}秒"
                    else:
                        estimate_str = f"{est_seconds}秒"
                    
                    estimate_md = f"> ⏱️ **预估处理时间**：约 {estimate_str}（基于 small 模型，medium 约为 2-3 倍）"
                    
                    # 根据时长给出不同提示
                    if total_duration > MAX_AUDIO_DURATION_SECONDS:
                        return f"⚠️ 已上传 {valid_count} 个音频，总时长 {duration_str}（超过10分钟，部分文件将被过滤）", gr.update(interactive=True), gr.update(value=estimate_md, visible=True)
                    elif total_duration > 480:  # 8分钟
                        return f"⚠️ 已上传 {valid_count} 个音频，总时长 {duration_str}（建议控制在8分钟内）", gr.update(interactive=True), gr.update(value=estimate_md, visible=True)
                    else:
                        return f"✅ 已上传 {valid_count} 个音频，总时长 {duration_str}", gr.update(interactive=True), gr.update(value=estimate_md, visible=True)
                
                audio_upload.change(
                    fn=check_audio_upload,
                    inputs=[audio_upload],
                    outputs=[upload_status, make_btn, time_estimate]
                )
                
                make_btn.click(
                    fn=process_make_voicebank,
                    inputs=[audio_upload, make_source_name, make_language, make_whisper, session_id_state],
                    outputs=[make_status, make_log, make_download, session_voicebank],
                    concurrency_limit=CloudConfig.MAX_CONCURRENT_JOBS
                )
            
            # ==================== 导出音源页 ====================
            with gr.Tab("📤 导出音源"):
                gr.Markdown("### 选择音源包")
                
                # 使用刚制作的音源按钮
                use_last_btn = gr.Button("📦 使用刚制作的音源", variant="secondary")
                
                gr.Markdown("或者上传之前制作的音源压缩包（包含 slices 和 textgrid 目录）")
                
                export_upload = gr.File(
                    label="上传音源包 (.zip)",
                    file_types=[".zip"]
                )
                
                export_info = gr.Textbox(
                    label="音源信息",
                    interactive=False,
                    placeholder="上传后显示音源信息"
                )
                
                # 上传后自动验证
                def on_upload(file):
                    if file:
                        valid, msg, name = validate_voicebank_zip(file)
                        return msg
                    return ""
                
                export_upload.change(
                    fn=on_upload,
                    inputs=[export_upload],
                    outputs=[export_info]
                )
                
                # 使用刚制作的音源
                use_last_btn.click(
                    fn=get_last_made_voicebank,
                    inputs=[session_voicebank],
                    outputs=[export_upload, export_info]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 导出设置")
                
                # 获取插件选项配置
                plugins_config = get_plugin_options_config(plugins)
                
                export_plugin = gr.Dropdown(
                    choices=plugin_names,
                    value=plugin_names[0] if plugin_names else None,
                    label="导出插件"
                )
                
                # ===== 动态选项区域 =====
                # 为每个插件动态创建选项组件
                all_plugin_groups = {}
                all_plugin_components = {}
                
                for idx, (pname, poptions) in enumerate(plugins_config.items()):
                    is_first = (idx == 0)
                    plugin_components = {}
                    
                    with gr.Group(visible=is_first) as plugin_group:
                        # 插件描述
                        if pname in plugins:
                            gr.Markdown(f"> {plugins[pname].description}")
                        
                        # 动态创建选项组件
                        for opt in poptions:
                            opt_type = opt["type"]
                            key = opt["key"]
                            label = opt["label"]
                            default = opt["default"]
                            description = opt.get("description", "")
                            choices = opt.get("choices", [])
                            min_val = opt.get("min_value")
                            max_val = opt.get("max_value")
                            step = opt.get("step")
                            
                            if opt_type == "label":
                                gr.Markdown(f"*{label}*")
                                continue
                            elif opt_type == "text":
                                component = gr.Textbox(
                                    label=label,
                                    value=default or "",
                                    info=description
                                )
                            elif opt_type == "number":
                                component = gr.Number(
                                    label=label,
                                    value=default if default is not None else 0,
                                    minimum=min_val,
                                    maximum=max_val,
                                    step=step or 1,
                                    info=description
                                )
                            elif opt_type == "switch":
                                component = gr.Checkbox(
                                    label=label,
                                    value=bool(default),
                                    info=description
                                )
                            elif opt_type == "combo":
                                component = gr.Dropdown(
                                    label=label,
                                    choices=choices,
                                    value=default if default in choices else (choices[0] if choices else None),
                                    info=description
                                )
                            elif opt_type == "multi_select":
                                component = gr.CheckboxGroup(
                                    label=label,
                                    choices=choices,
                                    value=default if isinstance(default, list) else [],
                                    info=description
                                )
                            else:
                                component = gr.Textbox(
                                    label=label,
                                    value=str(default) if default else "",
                                    info=description
                                )
                            
                            plugin_components[key] = component
                    
                    all_plugin_groups[pname] = plugin_group
                    all_plugin_components[pname] = plugin_components
                
                # 插件切换时更新选项组可见性
                def on_plugin_change(selected_plugin):
                    """切换插件时更新选项区域可见性"""
                    updates = []
                    for pname in plugins_config.keys():
                        updates.append(gr.update(visible=(pname == selected_plugin)))
                    return updates
                
                # 绑定插件切换事件
                export_plugin.change(
                    fn=on_plugin_change,
                    inputs=[export_plugin],
                    outputs=list(all_plugin_groups.values())
                )
                
                # 收集选项并导出
                def collect_and_export(zip_file, plugin_name, session_id, *all_values, progress=gr.Progress()):
                    """收集当前插件的选项并执行导出"""
                    # 根据插件名找到对应的选项配置
                    if plugin_name not in plugins_config:
                        return "❌ 未找到插件配置", "", None
                    
                    # 计算当前插件的值在 all_values 中的位置
                    start_idx = 0
                    for pname in plugins_config.keys():
                        if pname == plugin_name:
                            break
                        # 统计该插件的非 label 选项数量
                        start_idx += sum(1 for opt in plugins_config[pname] if opt["type"] != "label")
                    
                    # 提取当前插件的选项值
                    options = {}
                    current_idx = start_idx
                    for opt in plugins_config[plugin_name]:
                        if opt["type"] == "label":
                            continue
                        key = opt["key"]
                        if current_idx < len(all_values):
                            value = all_values[current_idx]
                            # 类型转换
                            if opt["type"] == "number":
                                value = float(value) if value is not None else opt["default"]
                            options[key] = value
                        current_idx += 1
                    
                    options_json = json.dumps(options, ensure_ascii=False)
                    return process_export_voicebank(zip_file, plugin_name, options_json, session_id, progress)
                
                export_btn = gr.Button("📤 开始导出", variant="primary", size="lg")
                
                export_status = gr.Textbox(label="状态", interactive=False)
                export_log = gr.Textbox(label="处理日志", lines=10, interactive=False)
                
                gr.Markdown("### 下载结果")
                export_download = gr.File(label="导出结果下载", interactive=False)
                
                gr.Markdown("""
                > 💡 导出说明：
                > - 从 TextGrid 提取每个汉字/音节的时间边界
                > - 按拼音/罗马音分类，选取最佳样本
                > - 导出为适配其他软件的音源格式
                """)
                
                # 收集所有插件的所有组件作为输入
                all_option_components = []
                for pname in plugins_config.keys():
                    if pname in all_plugin_components:
                        for opt in plugins_config[pname]:
                            if opt["type"] != "label" and opt["key"] in all_plugin_components[pname]:
                                all_option_components.append(all_plugin_components[pname][opt["key"]])
                
                export_btn.click(
                    fn=collect_and_export,
                    inputs=[export_upload, export_plugin, session_id_state] + all_option_components,
                    outputs=[export_status, export_log, export_download],
                    concurrency_limit=CloudConfig.MAX_CONCURRENT_JOBS
                )
            
            # ==================== MFA补对齐页 ====================
            with gr.Tab("🔧 MFA补对齐"):
                gr.Markdown("""
                ## MFA 补对齐
                
                <div style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%); border-left: 4px solid #004085; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
                    <p style="margin: 0 0 8px 0; font-weight: bold; color: #004085;">ℹ️ 功能说明</p>
                    <p style="margin: 0; color: #004085; line-height: 1.6;">
                        如果在「制作音源」时 MFA 不可用或对齐失败，音源包中将缺少 TextGrid 文件。<br/>
                        此页面可以为已有的音源包补充 MFA 对齐数据，以便后续导出 UTAU oto.ini 等格式。
                    </p>
                </div>
                """)
                
                gr.Markdown("### 上传音源包")
                gr.Markdown("上传包含 `slices` 目录（.wav + .lab 文件）的音源压缩包")
                
                mfa_upload = gr.File(
                    label="上传音源包 (.zip)",
                    file_types=[".zip"]
                )
                
                mfa_info = gr.Textbox(
                    label="音源信息",
                    interactive=False,
                    placeholder="上传后显示音源信息和 TextGrid 状态"
                )
                
                # 上传后自动验证
                mfa_upload.change(
                    fn=validate_mfa_voicebank,
                    inputs=[mfa_upload],
                    outputs=[mfa_info]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 对齐设置")
                
                with gr.Row():
                    mfa_language = gr.Dropdown(
                        choices=CloudConfig.LANGUAGES,
                        value="chinese",
                        label="语言",
                        info="选择与音源匹配的语言"
                    )
                    mfa_status_display = gr.Textbox(
                        label="MFA 状态",
                        value=mfa_status,
                        interactive=False
                    )
                
                mfa_btn = gr.Button("🔧 开始对齐", variant="primary", size="lg")
                
                mfa_process_status = gr.Textbox(label="状态", interactive=False)
                mfa_log = gr.Textbox(label="处理日志", lines=10, interactive=False)
                
                gr.Markdown("### 下载结果")
                mfa_download = gr.File(label="补齐后的音源包下载", interactive=False)
                
                gr.Markdown("""
                > 💡 **处理流程**：
                > 1. 解压上传的音源包
                > 2. 检测 slices 目录中的 .wav 和 .lab 文件
                > 3. 执行 MFA 强制对齐，生成 TextGrid 文件
                > 4. 打包完整音源包供下载
                """)
                
                mfa_btn.click(
                    fn=process_mfa_realign,
                    inputs=[mfa_upload, mfa_language, session_id_state],
                    outputs=[mfa_process_status, mfa_log, mfa_download],
                    concurrency_limit=CloudConfig.MAX_CONCURRENT_JOBS
                )
            
            # ==================== 关于页 ====================
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown("""
                ## 人力V助手 (JinrikiHelper)
                
                语音数据集处理工具，用于自动化制作语音音源库。
                
                ### 功能特点
                
                - **VAD 切片**: 使用 Silero VAD 自动检测语音片段
                - **语音识别**: 使用 Whisper 模型转录文本
                - **强制对齐**: 使用 MFA 生成音素级时间标注
                - **智能导出**: 按拼音分类，选取最佳样本
                
                ### 支持语言
                
                - 中文（普通话）
                - 日语
                
                ### 使用流程
                
                1. **制作音源**: 上传原始音频 → 自动处理 → 下载音源包
                2. **导出音源**: 上传音源包 → 选择导出格式 → 下载导出结果
                
                ---
                
                **作者**: TNOT | **协议**: MIT
                
                本工具集成 Montreal Forced Aligner (MIT License)
                """)
        
        cancel_jobs_btn.click(
            fn=cancel_current_session_tasks,
            inputs=[session_id_state],
            outputs=[cancel_jobs_msg]
        )

        # 页面加载时刷新并发状态
        app.load(
            fn=get_concurrency_status,
            outputs=[concurrency_display]
        )

        concurrency_timer.tick(
            fn=get_concurrency_status,
            outputs=[concurrency_display]
        )
    
    return app


def main():
    """云端入口"""
    app = create_cloud_ui()
    # 启用队列，实际并发由 CloudTaskManager + 事件并发上限共同控制
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
