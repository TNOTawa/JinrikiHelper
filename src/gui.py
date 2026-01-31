# -*- coding: utf-8 -*-
"""
人力V助手 (JinrikiHelper) Web UI
基于 Gradio 6.2.0 构建
支持本地运行和云端部署 (HF Spaces / 魔塔社区)
作者：TNOT
"""

import gradio as gr
import threading
import logging
import os
import sys
import json
import platform
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Callable

# 环境检测
IS_CLOUD = any([
    os.environ.get("SPACE_ID"),           # Hugging Face Spaces
    os.environ.get("MODELSCOPE_SPACE"),   # 魔塔社区
])

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    CONFIG_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.json"
    )
    
    WHISPER_MODELS = {
        "whisper-small": {"name": "openai/whisper-small", "desc": "小型模型 (~500MB)", "size": "~500MB"},
        "whisper-medium": {"name": "openai/whisper-medium", "desc": "中型模型 (~1.5GB)", "size": "~1.5GB"}
    }
    
    def __init__(self):
        self._default_models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models"
        )
        self._default_bank_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "bank"
        )
        self.config = self._load()
    
    def _load(self) -> dict:
        """加载配置"""
        default = {
            "whisper_model": "whisper-small",
            "models_dir": self._default_models_dir,
            "mfa_dir": os.path.join(self._default_models_dir, "mfa"),
            "bank_dir": self._default_bank_dir
        }
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    default.update(json.load(f))
            except Exception as e:
                logger.warning(f"加载配置失败: {e}")
        return default
    
    def save(self):
        """保存配置"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save()


# 全局配置管理器
config_manager = ConfigManager()


# ==================== 工具函数 ====================

def check_mfa_available() -> bool:
    """检查MFA引擎是否可用"""
    from src.mfa_runner import check_mfa_available as _check
    return _check()


def check_vad_status() -> str:
    """检查VAD模型状态"""
    from src.silero_vad_downloader import is_vad_model_downloaded
    if is_vad_model_downloaded(config_manager.get("models_dir")):
        return "✅ 已下载"
    return "⏳ 未下载"


def scan_whisper_models() -> List[str]:
    """扫描已下载的Whisper模型"""
    from src.pipeline import scan_whisper_models as _scan
    models_dir = config_manager.get("models_dir")
    models = _scan(models_dir)
    preset_names = [m["name"] for m in ConfigManager.WHISPER_MODELS.values()]
    for m in models:
        if m not in preset_names:
            preset_names.append(m)
    return preset_names if preset_names else ["openai/whisper-small"]


def scan_mfa_models() -> Dict[str, List[str]]:
    """扫描MFA模型"""
    from src.pipeline import scan_mfa_models as _scan
    mfa_dir = config_manager.get("mfa_dir")
    return _scan(os.path.dirname(mfa_dir))


def scan_voice_banks() -> List[str]:
    """扫描音源列表"""
    bank_dir = config_manager.get("bank_dir")
    voice_banks = []
    if os.path.exists(bank_dir):
        for name in os.listdir(bank_dir):
            source_dir = os.path.join(bank_dir, name)
            if os.path.isdir(source_dir) and not name.startswith('.'):
                slices_dir = os.path.join(source_dir, "slices")
                textgrid_dir = os.path.join(source_dir, "textgrid")
                if os.path.exists(slices_dir) or os.path.exists(textgrid_dir):
                    voice_banks.append(name)
    return voice_banks if voice_banks else ["(未找到音源)"]


def get_voice_bank_info(name: str) -> str:
    """获取音源信息"""
    if not name or name.startswith("("):
        return ""
    bank_dir = config_manager.get("bank_dir")
    source_dir = os.path.join(bank_dir, name)
    slices_dir = os.path.join(source_dir, "slices")
    textgrid_dir = os.path.join(source_dir, "textgrid")
    
    slices_count = 0
    textgrid_count = 0
    if os.path.exists(slices_dir):
        slices_count = len([f for f in os.listdir(slices_dir) if f.endswith('.wav')])
    if os.path.exists(textgrid_dir):
        textgrid_count = len([f for f in os.listdir(textgrid_dir) if f.endswith('.TextGrid')])
    
    return f"切片: {slices_count} 个 | TextGrid: {textgrid_count} 个"


def load_export_plugins() -> Dict:
    """加载导出插件"""
    from src.export_plugins import load_plugins
    plugins_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "export_plugins"
    )
    return load_plugins(plugins_dir)


# ==================== 模型下载功能 ====================

def download_vad_model(progress=gr.Progress()):
    """下载VAD模型"""
    from src.silero_vad_downloader import download_silero_vad
    logs = []
    
    def log_callback(msg):
        logs.append(msg)
    
    progress(0, desc="开始下载 Silero VAD 模型...")
    success, result = download_silero_vad(config_manager.get("models_dir"), log_callback)
    progress(1, desc="完成")
    
    if success:
        return "✅ 已下载", "\n".join(logs) + f"\n模型已保存: {result}"
    return "❌ 下载失败", "\n".join(logs)


def download_mfa_model(language: str, progress=gr.Progress()):
    """下载MFA模型"""
    from src.mfa_model_downloader import download_language_models
    logs = []
    
    def log_callback(msg):
        logs.append(msg)
    
    output_dir = config_manager.get("mfa_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    progress(0, desc=f"开始下载 MFA 模型: {language}")
    success, acoustic_path, dict_path = download_language_models(
        language=language, output_dir=output_dir, progress_callback=log_callback
    )
    progress(1, desc="完成")
    
    if success:
        return "✅ 已下载", "\n".join(logs) + f"\n声学模型: {acoustic_path}\n字典文件: {dict_path}"
    return "❌ 下载失败", "\n".join(logs)


def download_whisper_model(model_key: str, progress=gr.Progress()):
    """下载/加载Whisper模型"""
    logs = []
    
    try:
        logs.append("正在加载 transformers 库...")
        progress(0.1, desc="加载 transformers...")
        from transformers import pipeline
        import torch
        
        model_name = ConfigManager.WHISPER_MODELS.get(model_key, {}).get("name", model_key)
        cache_dir = os.path.join(config_manager.get("models_dir"), "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        
        logs.append(f"正在下载/加载 {model_key}...")
        progress(0.3, desc=f"下载/加载 {model_key}...")
        
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        _ = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            model_kwargs={"cache_dir": cache_dir}
        )
        
        progress(1, desc="完成")
        config_manager.set("whisper_model", model_key)
        return "✅ 已就绪", "\n".join(logs) + "\nWhisper 模型加载完成"
        
    except Exception as e:
        logger.error(f"Whisper 模型加载失败: {e}", exc_info=True)
        return "❌ 加载失败", "\n".join(logs) + f"\n错误: {e}"


# ==================== 制作音源功能 ====================

def run_step0(source_name: str, input_path: str, output_dir: str,
              whisper_model: str, language: str, progress=gr.Progress()):
    """步骤0: VAD切片 + Whisper转录"""
    from src.pipeline import PipelineConfig, VoiceBankPipeline
    
    if not source_name.strip():
        return "❌ 请输入音源名称", ""
    if not input_path.strip():
        return "❌ 请选择输入音频", ""
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
    
    mfa_models = scan_mfa_models()
    dict_path = None
    acoustic_path = None
    mfa_dir = config_manager.get("mfa_dir")
    
    if mfa_models["dictionary"]:
        dict_path = os.path.join(mfa_dir, mfa_models["dictionary"][0])
    if mfa_models["acoustic"]:
        acoustic_path = os.path.join(mfa_dir, mfa_models["acoustic"][0])
    
    config = PipelineConfig(
        source_name=source_name,
        input_path=input_path,
        output_base_dir=output_dir or config_manager.get("bank_dir"),
        models_dir=config_manager.get("models_dir"),
        whisper_model=whisper_model,
        mfa_dict_path=dict_path,
        mfa_model_path=acoustic_path,
        language=language
    )
    
    pipeline = VoiceBankPipeline(config, log_callback)
    
    progress(0, desc="开始音频预处理...")
    log_callback("=" * 50)
    log_callback("【步骤0】音频预处理 (VAD切片 + Whisper转录)")
    
    success, msg, _ = pipeline.step0_preprocess()
    progress(1, desc="完成")
    
    if success:
        log_callback(f"✅ {msg}")
    else:
        log_callback(f"❌ {msg}")
    log_callback("=" * 50)
    
    status = "✅ 预处理完成" if success else f"❌ {msg}"
    return status, "\n".join(logs)


def run_step1(source_name: str, output_dir: str, dict_file: str, 
              acoustic_file: str, progress=gr.Progress()):
    """步骤1: MFA对齐"""
    from src.pipeline import PipelineConfig, VoiceBankPipeline
    
    if not source_name.strip():
        return "❌ 请输入音源名称", ""
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
    
    mfa_dir = config_manager.get("mfa_dir")
    dict_path = os.path.join(mfa_dir, dict_file) if dict_file and not dict_file.startswith("(") else None
    acoustic_path = os.path.join(mfa_dir, acoustic_file) if acoustic_file and not acoustic_file.startswith("(") else None
    
    config = PipelineConfig(
        source_name=source_name,
        input_path="",
        output_base_dir=output_dir or config_manager.get("bank_dir"),
        models_dir=config_manager.get("models_dir"),
        mfa_dict_path=dict_path,
        mfa_model_path=acoustic_path
    )
    
    pipeline = VoiceBankPipeline(config, log_callback)
    
    progress(0, desc="开始MFA对齐...")
    log_callback("=" * 50)
    log_callback("【步骤1】MFA语音对齐")
    
    success, msg = pipeline.step1_mfa_align()
    progress(1, desc="完成")
    
    if success:
        log_callback(f"✅ {msg}")
    else:
        log_callback(f"❌ {msg}")
    log_callback("=" * 50)
    
    status = "✅ MFA对齐完成" if success else f"❌ {msg}"
    return status, "\n".join(logs)


def run_full_pipeline(source_name: str, input_path: str, output_dir: str,
                      whisper_model: str, dict_file: str, acoustic_file: str,
                      language: str, progress=gr.Progress()):
    """一键执行全部流程"""
    from src.pipeline import PipelineConfig, VoiceBankPipeline
    
    if not source_name.strip():
        return "❌ 请输入音源名称", ""
    if not input_path.strip():
        return "❌ 请选择输入音频", ""
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
    
    mfa_dir = config_manager.get("mfa_dir")
    dict_path = os.path.join(mfa_dir, dict_file) if dict_file and not dict_file.startswith("(") else None
    acoustic_path = os.path.join(mfa_dir, acoustic_file) if acoustic_file and not acoustic_file.startswith("(") else None
    
    config = PipelineConfig(
        source_name=source_name,
        input_path=input_path,
        output_base_dir=output_dir or config_manager.get("bank_dir"),
        models_dir=config_manager.get("models_dir"),
        whisper_model=whisper_model,
        mfa_dict_path=dict_path,
        mfa_model_path=acoustic_path,
        language=language
    )
    
    pipeline = VoiceBankPipeline(config, log_callback)
    
    progress(0, desc="开始制作音源...")
    success, msg = pipeline.run_make_pipeline()
    progress(1, desc="完成")
    
    status = "✅ 音源制作完成" if success else f"❌ {msg}"
    return status, "\n".join(logs)


# ==================== 导出音源功能 ====================

def create_download_zip(source_dir: str, zip_name: str) -> Optional[str]:
    """
    打包目录为 zip 文件供下载
    
    参数:
        source_dir: 要打包的目录
        zip_name: zip 文件名 (不含扩展名)
    
    返回:
        zip 文件路径，失败返回 None
    """
    if not os.path.isdir(source_dir):
        return None
    
    try:
        zip_path = os.path.join(tempfile.gettempdir(), f"{zip_name}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zf.write(file_path, arcname)
        return zip_path
    except Exception as e:
        logger.error(f"打包失败: {e}")
        return None


def run_export(voice_bank: str, plugin_name: str, options: dict, progress=gr.Progress()):
    """执行导出"""
    if not voice_bank or voice_bank.startswith("("):
        return "❌ 请选择有效的音源", "", None
    
    plugins = load_export_plugins()
    if plugin_name not in plugins:
        return f"❌ 未找到插件: {plugin_name}", "", None
    
    plugin = plugins[plugin_name]
    bank_dir = config_manager.get("bank_dir")
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
    
    progress(0, desc=f"开始导出: {plugin_name}")
    log_callback("=" * 50)
    log_callback(f"【{plugin_name}】音源: {voice_bank}")
    
    plugin.set_progress_callback(log_callback)
    success, msg = plugin.export(voice_bank, bank_dir, options)
    progress(1, desc="完成")
    
    download_file = None
    if success:
        log_callback(f"✅ {msg}")
        # 打包导出结果供下载
        export_dir = os.path.join(
            os.path.dirname(bank_dir), "export", voice_bank, plugin_name.replace(" ", "_")
        )
        if os.path.isdir(export_dir):
            zip_name = f"{voice_bank}_{plugin_name.replace(' ', '_')}"
            download_file = create_download_zip(export_dir, zip_name)
            if download_file:
                log_callback(f"📦 已打包: {os.path.basename(download_file)}")
    else:
        log_callback(f"❌ {msg}")
    log_callback("=" * 50)
    
    status = "✅ 导出完成" if success else f"❌ {msg}"
    return status, "\n".join(logs), download_file


def download_voice_bank(voice_bank: str) -> Optional[str]:
    """打包音源数据供下载"""
    if not voice_bank or voice_bank.startswith("("):
        return None
    
    bank_dir = config_manager.get("bank_dir")
    source_dir = os.path.join(bank_dir, voice_bank)
    
    if not os.path.isdir(source_dir):
        return None
    
    return create_download_zip(source_dir, f"{voice_bank}_音源数据")


# ==================== 构建界面 ====================

def create_ui():
    """创建Gradio界面"""
    
    # 初始化数据
    whisper_models = list(ConfigManager.WHISPER_MODELS.keys())
    mfa_models = scan_mfa_models()
    dict_files = mfa_models["dictionary"] if mfa_models["dictionary"] else ["(未找到字典文件)"]
    acoustic_files = mfa_models["acoustic"] if mfa_models["acoustic"] else ["(未找到声学模型)"]
    voice_banks = scan_voice_banks()
    
    # MFA 状态检测 (区分平台)
    if check_mfa_available():
        mfa_status = "✅ MFA 环境已就绪"
    elif platform.system() == "Windows":
        mfa_status = "❌ MFA 环境不可用，请检查 tools/mfa_engine"
    else:
        mfa_status = "❌ MFA 未安装，请运行: pip install montreal-forced-aligner"
    
    # 云端环境提示
    env_notice = ""
    if IS_CLOUD:
        env_notice = "> ☁️ 当前为云端环境，处理完成后请及时下载结果"
    
    with gr.Blocks(title="人力V助手 (JinrikiHelper)") as app:
        gr.Markdown("# 🎤 人力V助手 (JinrikiHelper)")
        gr.Markdown("语音数据集处理工具 - 自动化制作语音音源库")
        if env_notice:
            gr.Markdown(env_notice)
        
        with gr.Tabs():
            # ==================== 模型下载页 ====================
            with gr.Tab("📦 模型下载"):
                gr.Markdown("### Whisper 语音识别模型")
                with gr.Row():
                    whisper_select = gr.Dropdown(
                        choices=whisper_models,
                        value=config_manager.get("whisper_model", "whisper-small"),
                        label="模型版本"
                    )
                    whisper_status = gr.Textbox(label="状态", value="⏳ 未加载", interactive=False)
                whisper_btn = gr.Button("下载 / 加载模型", variant="primary")
                whisper_log = gr.Textbox(label="日志", lines=3, interactive=False)
                
                whisper_btn.click(
                    fn=download_whisper_model,
                    inputs=[whisper_select],
                    outputs=[whisper_status, whisper_log]
                )
                
                gr.Markdown("---")
                gr.Markdown("### Silero VAD 模型")
                gr.Markdown("用于语音活动检测和音频切片")
                with gr.Row():
                    vad_status = gr.Textbox(label="状态", value=check_vad_status(), interactive=False)
                    vad_btn = gr.Button("下载模型", variant="primary")
                vad_log = gr.Textbox(label="日志", lines=3, interactive=False)
                
                vad_btn.click(
                    fn=download_vad_model,
                    outputs=[vad_status, vad_log]
                )
                
                gr.Markdown("---")
                gr.Markdown("### MFA 声学模型")
                gr.Markdown("Montreal Forced Aligner 模型，用于语音对齐")
                with gr.Row():
                    mfa_lang = gr.Dropdown(
                        choices=["mandarin", "japanese"],
                        value="mandarin",
                        label="选择语言"
                    )
                    mfa_status_box = gr.Textbox(label="状态", value="⏳ 未下载", interactive=False)
                mfa_btn = gr.Button("下载模型", variant="primary")
                mfa_log = gr.Textbox(label="日志", lines=3, interactive=False)
                
                mfa_btn.click(
                    fn=download_mfa_model,
                    inputs=[mfa_lang],
                    outputs=[mfa_status_box, mfa_log]
                )

            
            # ==================== 制作音源页 ====================
            with gr.Tab("🎵 制作音源"):
                gr.Markdown("### 基本设置")
                with gr.Row():
                    source_name = gr.Textbox(
                        label="音源名称",
                        placeholder="my_voice",
                        info="输出到 bank/[音源名称]/"
                    )
                    language = gr.Dropdown(
                        choices=["chinese", "japanese", "english"],
                        value="chinese",
                        label="转录语言"
                    )
                
                with gr.Row():
                    input_audio = gr.Textbox(
                        label="输入音频",
                        placeholder="选择音频文件或文件夹路径",
                        scale=3
                    )
                    input_file_btn = gr.UploadButton(
                        "📁 上传文件",
                        file_types=["audio"],
                        file_count="multiple"
                    )
                
                output_dir = gr.Textbox(
                    label="输出目录",
                    value=config_manager.get("bank_dir"),
                    placeholder="bank"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 模型选择")
                
                with gr.Row():
                    whisper_model_select = gr.Dropdown(
                        choices=scan_whisper_models(),
                        value=scan_whisper_models()[0] if scan_whisper_models() else "openai/whisper-small",
                        label="Whisper模型"
                    )
                    refresh_whisper_btn = gr.Button("🔄", scale=0)
                
                with gr.Row():
                    dict_select = gr.Dropdown(
                        choices=dict_files,
                        value=dict_files[0] if dict_files else None,
                        label="MFA字典"
                    )
                    acoustic_select = gr.Dropdown(
                        choices=acoustic_files,
                        value=acoustic_files[0] if acoustic_files else None,
                        label="MFA声学模型"
                    )
                    refresh_mfa_btn = gr.Button("🔄", scale=0)
                
                mfa_env_status = gr.Markdown(mfa_status)
                
                gr.Markdown("---")
                gr.Markdown("### 执行流程")
                
                with gr.Row():
                    step0_btn = gr.Button("步骤0: 切片+转录", variant="secondary")
                    step1_btn = gr.Button("步骤1: MFA对齐", variant="secondary")
                
                full_btn = gr.Button("▶ 一键执行全部流程", variant="primary", size="lg")
                
                make_status = gr.Textbox(label="状态", interactive=False)
                make_log = gr.Textbox(label="日志输出", lines=10, interactive=False)
                
                # 文件上传处理
                def handle_upload(files):
                    if files:
                        if len(files) == 1:
                            return files[0].name
                        return os.path.dirname(files[0].name)
                    return ""
                
                input_file_btn.upload(
                    fn=handle_upload,
                    inputs=[input_file_btn],
                    outputs=[input_audio]
                )
                
                # 刷新模型列表
                def refresh_whisper():
                    models = scan_whisper_models()
                    return gr.update(choices=models, value=models[0] if models else "openai/whisper-small")
                
                def refresh_mfa():
                    models = scan_mfa_models()
                    dict_files = models["dictionary"] if models["dictionary"] else ["(未找到字典文件)"]
                    acoustic_files = models["acoustic"] if models["acoustic"] else ["(未找到声学模型)"]
                    return (
                        gr.update(choices=dict_files, value=dict_files[0]),
                        gr.update(choices=acoustic_files, value=acoustic_files[0])
                    )
                
                refresh_whisper_btn.click(fn=refresh_whisper, outputs=[whisper_model_select])
                refresh_mfa_btn.click(fn=refresh_mfa, outputs=[dict_select, acoustic_select])
                
                # 执行按钮
                step0_btn.click(
                    fn=run_step0,
                    inputs=[source_name, input_audio, output_dir, whisper_model_select, language],
                    outputs=[make_status, make_log]
                )
                
                step1_btn.click(
                    fn=run_step1,
                    inputs=[source_name, output_dir, dict_select, acoustic_select],
                    outputs=[make_status, make_log]
                )
                
                full_btn.click(
                    fn=run_full_pipeline,
                    inputs=[source_name, input_audio, output_dir, whisper_model_select, 
                            dict_select, acoustic_select, language],
                    outputs=[make_status, make_log]
                )
            
            # ==================== 导出音源页 ====================
            with gr.Tab("📤 导出音源"):
                gr.Markdown("### 选择音源")
                
                with gr.Row():
                    voice_bank_select = gr.Dropdown(
                        choices=voice_banks,
                        value=voice_banks[0] if voice_banks else None,
                        label="音源"
                    )
                    refresh_bank_btn = gr.Button("🔄 刷新", scale=0)
                
                bank_info = gr.Textbox(
                    label="音源信息",
                    value=get_voice_bank_info(voice_banks[0]) if voice_banks and not voice_banks[0].startswith("(") else "",
                    interactive=False
                )
                
                def refresh_banks():
                    banks = scan_voice_banks()
                    info = get_voice_bank_info(banks[0]) if banks and not banks[0].startswith("(") else ""
                    return gr.update(choices=banks, value=banks[0] if banks else None), info
                
                def update_bank_info(name):
                    return get_voice_bank_info(name)
                
                refresh_bank_btn.click(fn=refresh_banks, outputs=[voice_bank_select, bank_info])
                voice_bank_select.change(fn=update_bank_info, inputs=[voice_bank_select], outputs=[bank_info])
                
                gr.Markdown("---")
                gr.Markdown("### 导出方式")
                
                # 加载插件
                plugins = load_export_plugins()
                plugin_names = list(plugins.keys()) if plugins else ["(未找到导出插件)"]
                
                plugin_select = gr.Dropdown(
                    choices=plugin_names,
                    value=plugin_names[0] if plugin_names else None,
                    label="导出插件"
                )
                
                # 插件选项（简化版，使用JSON输入）
                gr.Markdown("#### 插件选项")
                plugin_options = gr.JSON(
                    value={},
                    label="选项配置 (JSON格式)"
                )
                
                export_btn = gr.Button("📤 导出", variant="primary")
                export_status = gr.Textbox(label="状态", interactive=False)
                export_log = gr.Textbox(label="日志输出", lines=8, interactive=False)
                
                # 下载区域
                gr.Markdown("---")
                gr.Markdown("### 下载结果")
                with gr.Row():
                    export_download = gr.File(label="导出结果下载", interactive=False)
                    bank_download_btn = gr.Button("📥 下载音源数据", variant="secondary")
                bank_download = gr.File(label="音源数据下载", interactive=False)
                
                if IS_CLOUD:
                    gr.Markdown("> 💡 云端环境数据不会持久保存，请及时下载处理结果")
                
                export_btn.click(
                    fn=run_export,
                    inputs=[voice_bank_select, plugin_select, plugin_options],
                    outputs=[export_status, export_log, export_download]
                )
                
                bank_download_btn.click(
                    fn=download_voice_bank,
                    inputs=[voice_bank_select],
                    outputs=[bank_download]
                )
            
            # ==================== 设置页 ====================
            with gr.Tab("⚙️ 设置"):
                gr.Markdown("### 目录设置")
                
                models_dir_input = gr.Textbox(
                    label="模型目录",
                    value=config_manager.get("models_dir"),
                    info="Whisper、VAD、MFA模型存放目录"
                )
                
                mfa_dir_input = gr.Textbox(
                    label="MFA模型目录",
                    value=config_manager.get("mfa_dir"),
                    info="MFA声学模型和字典存放目录"
                )
                
                bank_dir_input = gr.Textbox(
                    label="音源库目录",
                    value=config_manager.get("bank_dir"),
                    info="音源数据存放目录"
                )
                
                def save_settings(models_dir, mfa_dir, bank_dir):
                    config_manager.set("models_dir", models_dir)
                    config_manager.set("mfa_dir", mfa_dir)
                    config_manager.set("bank_dir", bank_dir)
                    return "✅ 设置已保存"
                
                save_btn = gr.Button("💾 保存设置", variant="primary")
                save_status = gr.Textbox(label="状态", interactive=False)
                
                save_btn.click(
                    fn=save_settings,
                    inputs=[models_dir_input, mfa_dir_input, bank_dir_input],
                    outputs=[save_status]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 关于")
                gr.Markdown("""
                **人力V助手 (JinrikiHelper)**
                
                作者：TNOT | 开源协议：MIT
                
                本工具集成 Montreal Forced Aligner (MIT License)
                """)
    
    return app


def main():
    """主函数"""
    # 检查MFA引擎
    if not check_mfa_available():
        logger.warning("未检测到 MFA 引擎 (tools/mfa_engine)")
    
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
