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
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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


# 全局状态：存储最近制作的音源包路径
_last_made_voicebank: Optional[str] = None


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
    """打包目录为 zip"""
    if not os.path.isdir(source_dir):
        return None
    try:
        zip_path = os.path.join(CloudConfig.TEMP_BASE, f"{zip_name}.zip")
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


def extract_zip(zip_path: str, target_dir: str) -> Tuple[bool, str]:
    """解压 zip 文件"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
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


# ==================== 制作音源功能 ====================

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


def process_make_voicebank(
    audio_files,
    source_name: str,
    language: str,
    whisper_model: str,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    制作音源：上传音频 → VAD切片 → Whisper转录 → MFA对齐 → 打包下载
    
    返回: (状态, 日志, 下载文件路径)
    """
    global _last_made_voicebank
    from src.pipeline import PipelineConfig, VoiceBankPipeline
    
    logs = []
    def log(msg):
        logs.append(msg)
        logger.info(msg)
    
    # 验证输入
    if not source_name or not source_name.strip():
        return "❌ 请输入音源名称", "", None
    
    source_name = source_name.strip()
    
    valid, msg, file_paths = validate_audio_upload(audio_files)
    if not valid:
        return f"❌ {msg}", "", None
    
    log(f"📁 {msg}")
    
    # 创建临时工作空间
    workspace = create_temp_workspace()
    log(f"🔧 创建工作空间: {workspace}")
    
    try:
        # 准备输入目录
        input_dir = os.path.join(workspace, "input")
        bank_dir = os.path.join(workspace, "bank")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(bank_dir, exist_ok=True)
        
        # 复制音频文件到输入目录
        progress(0.05, desc="复制音频文件...")
        for src_path in file_paths:
            dst_path = os.path.join(input_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
        log(f"📋 已复制 {len(file_paths)} 个文件到工作目录")
        
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
            language=language
        )
        
        pipeline = VoiceBankPipeline(config, log)
        
        # 步骤0: VAD切片 + Whisper转录
        progress(0.1, desc="VAD切片 + Whisper转录...")
        log("\n" + "=" * 50)
        log("【步骤1】VAD切片 + Whisper转录")
        success, msg, slices = pipeline.step0_preprocess()
        if not success:
            return f"❌ 预处理失败: {msg}", "\n".join(logs), None
        log(f"✅ {msg}")
        
        # 步骤1: MFA对齐
        progress(0.6, desc="MFA语音对齐...")
        log("\n" + "=" * 50)
        log("【步骤2】MFA语音对齐")
        
        if check_mfa_available():
            success, msg = pipeline.step1_mfa_align()
            if not success:
                log(f"⚠️ MFA对齐失败: {msg}")
                log("继续导出（无TextGrid）...")
            else:
                log(f"✅ {msg}")
        else:
            log("⚠️ MFA不可用，跳过对齐步骤")
        
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
            # 保存到全局状态，供导出页面使用
            _last_made_voicebank = zip_path
            return "✅ 音源制作完成", "\n".join(logs), zip_path
        else:
            return "❌ 打包失败", "\n".join(logs), None
        
    except Exception as e:
        logger.error(f"制作音源失败: {e}", exc_info=True)
        return f"❌ 处理失败: {e}", "\n".join(logs), None
    
    finally:
        # 清理工作空间（保留zip文件）
        cleanup_workspace(workspace)


# ==================== 导出音源功能 ====================

def get_last_made_voicebank() -> Tuple[Optional[str], str]:
    """
    获取最近制作的音源包
    
    返回: (文件路径, 信息消息)
    """
    global _last_made_voicebank
    if _last_made_voicebank and os.path.exists(_last_made_voicebank):
        valid, msg, name = validate_voicebank_zip_path(_last_made_voicebank)
        if valid:
            return _last_made_voicebank, f"✅ 已选择刚制作的音源: {name}"
    return None, ""


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


def process_export_voicebank(
    zip_file,
    plugin_name: str,
    max_samples: int,
    naming_rule: str,
    first_naming_rule: str,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[str]]:
    """
    导出音源：上传音源包 → 解压 → 导出 → 打包下载
    
    返回: (状态, 日志, 下载文件路径)
    """
    logs = []
    def log(msg):
        logs.append(msg)
        logger.info(msg)
    
    # 验证输入
    valid, msg, source_name = validate_voicebank_zip(zip_file)
    if not valid:
        return f"❌ {msg}", "", None
    
    log(f"📦 {msg}")
    log(f"📝 音源名称: {source_name}")
    
    # 创建临时工作空间
    workspace = create_temp_workspace()
    log(f"🔧 创建工作空间")
    
    try:
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
        
        options = {
            "max_samples": max_samples,
            "naming_rule": naming_rule,
            "first_naming_rule": first_naming_rule,
            "clean_temp": True
        }
        
        success, msg = plugin.export(source_name, bank_dir, options)
        
        if not success:
            return f"❌ 导出失败: {msg}", "\n".join(logs), None
        
        log(f"✅ {msg}")
        
        # 打包导出结果
        progress(0.9, desc="打包结果...")
        log("\n" + "=" * 50)
        log("【打包结果】")
        
        export_dir = os.path.join(workspace, "export", source_name, "simple_export")
        
        # 如果导出目录不存在，尝试其他位置
        if not os.path.exists(export_dir):
            alt_export = os.path.join(os.path.dirname(bank_dir), "export", source_name, "simple_export")
            if os.path.exists(alt_export):
                export_dir = alt_export
        
        if not os.path.exists(export_dir):
            return "❌ 未找到导出结果", "\n".join(logs), None
        
        zip_name = f"{source_name}_导出结果"
        result_zip = create_zip(export_dir, zip_name)
        
        if result_zip:
            # 统计导出文件数
            file_count = len([f for f in os.listdir(export_dir) if f.endswith('.wav')])
            log(f"📦 已打包: {file_count} 个音频文件")
            progress(1.0, desc="完成")
            return "✅ 导出完成", "\n".join(logs), result_zip
        else:
            return "❌ 打包失败", "\n".join(logs), None
        
    except Exception as e:
        logger.error(f"导出失败: {e}", exc_info=True)
        return f"❌ 处理失败: {e}", "\n".join(logs), None
    
    finally:
        cleanup_workspace(workspace)


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
        
        gr.Markdown("# 🎤 人力V助手 (JinrikiHelper)")
        gr.Markdown("语音数据集处理工具 - 自动化制作语音音源库")
        gr.Markdown("> ☁️ 云端版：上传音频 → 自动处理 → 下载结果")
        
        with gr.Tabs():
            # ==================== 制作音源页 ====================
            with gr.Tab("🎵 制作音源"):
                gr.Markdown("### 上传音频文件")
                gr.Markdown("支持格式: WAV, MP3, FLAC, OGG, M4A")
                
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
                """)
                
                make_btn = gr.Button("🚀 开始制作", variant="primary", size="lg", interactive=False)
                
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
                    """检查音频上传状态，返回状态文本和按钮可用性"""
                    if not files:
                        return "⏳ 请上传音频文件", gr.update(interactive=False)
                    
                    valid_count = 0
                    for f in files:
                        path = f.name if hasattr(f, 'name') else str(f)
                        if path.lower().endswith(CloudConfig.AUDIO_EXTENSIONS):
                            valid_count += 1
                    
                    if valid_count == 0:
                        return f"❌ 未找到有效音频，支持: {', '.join(CloudConfig.AUDIO_EXTENSIONS)}", gr.update(interactive=False)
                    
                    return f"✅ 已上传 {valid_count} 个音频文件，可以开始制作", gr.update(interactive=True)
                
                audio_upload.change(
                    fn=check_audio_upload,
                    inputs=[audio_upload],
                    outputs=[upload_status, make_btn]
                )
                
                make_btn.click(
                    fn=process_make_voicebank,
                    inputs=[audio_upload, make_source_name, make_language, make_whisper],
                    outputs=[make_status, make_log, make_download]
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
                def use_last_voicebank():
                    """使用刚制作的音源包"""
                    path, msg = get_last_made_voicebank()
                    if path:
                        # 返回文件路径和信息
                        return path, msg
                    return None, "❌ 没有找到刚制作的音源，请先在「制作音源」页面制作，或手动上传"
                
                use_last_btn.click(
                    fn=use_last_voicebank,
                    inputs=[],
                    outputs=[export_upload, export_info]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 导出设置")
                
                export_plugin = gr.Dropdown(
                    choices=plugin_names,
                    value=plugin_names[0] if plugin_names else None,
                    label="导出插件"
                )
                
                with gr.Row():
                    export_max_samples = gr.Number(
                        label="每个拼音最大样本数",
                        value=10,
                        minimum=1,
                        maximum=1000
                    )
                
                with gr.Row():
                    export_naming = gr.Textbox(
                        label="命名规则",
                        value="%p%%n%",
                        info="%p%=拼音, %n%=序号"
                    )
                    export_first_naming = gr.Textbox(
                        label="首个样本命名",
                        value="%p%",
                        info="第0个样本的特殊规则"
                    )
                
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
                
                export_btn.click(
                    fn=process_export_voicebank,
                    inputs=[
                        export_upload, export_plugin,
                        export_max_samples, export_naming, export_first_naming
                    ],
                    outputs=[export_status, export_log, export_download]
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
    
    return app


def main():
    """云端入口"""
    app = create_cloud_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
