# -*- coding: utf-8 -*-
"""
语音数据集处理工具 GUI
基于 CustomTkinter 构建
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import logging
import os
import sys
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 设置外观
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


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
        self.config = self._load()
    
    def _load(self) -> dict:
        """加载配置"""
        default = {
            "whisper_model": "whisper-small",
            "models_dir": self._default_models_dir,
            "mfa_dir": os.path.join(self._default_models_dir, "mfa"),
            "bank_dir": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "bank"
            ),
            "show_log": False  # 默认关闭日志
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


class ModelDownloadFrame(ctk.CTkFrame):
    """模型配置功能面板"""
    
    def __init__(self, master, log_callback, config: ConfigManager):
        super().__init__(master)
        self.log_callback = log_callback
        self.config = config
        self.whisper_pipe = None
        self._download_thread = None
        self._setup_ui()

    def _setup_ui(self):
        # Whisper 模型区域
        ctk.CTkLabel(self, text="Whisper 语音识别模型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w"
        )
        
        ctk.CTkLabel(self, text="模型版本:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.whisper_model_var = ctk.StringVar(value=self.config.get("whisper_model"))
        ctk.CTkComboBox(
            self, values=list(ConfigManager.WHISPER_MODELS.keys()),
            variable=self.whisper_model_var, width=200,
            command=self._on_model_change
        ).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.model_desc_label = ctk.CTkLabel(self, text=self._get_model_desc(), text_color="gray")
        self.model_desc_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        
        ctk.CTkLabel(self, text="模型目录:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.models_dir_var = ctk.StringVar(value=self.config.get("models_dir"))
        ctk.CTkEntry(self, textvariable=self.models_dir_var, width=320).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_models_dir).grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(self, text="状态:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.whisper_status = ctk.CTkLabel(self, text="⏳ 未加载", text_color="gray")
        self.whisper_status.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.whisper_btn = ctk.CTkButton(self, text="下载 / 加载模型", command=self._download_whisper, width=140)
        self.whisper_btn.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        
        self.progress_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.progress_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # Silero VAD 区域
        ctk.CTkLabel(self, text="Silero VAD 模型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=5, column=0, columnspan=3, padx=10, pady=(20, 5), sticky="w"
        )
        ctk.CTkLabel(self, text="用于语音活动检测和音频切片", text_color="gray").grid(
            row=6, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w"
        )
        
        ctk.CTkLabel(self, text="状态:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.vad_status = ctk.CTkLabel(self, text="⏳ 未下载", text_color="gray")
        self.vad_status.grid(row=7, column=1, padx=5, pady=5, sticky="w")
        self.vad_btn = ctk.CTkButton(self, text="下载模型", command=self._download_vad, width=140)
        self.vad_btn.grid(row=7, column=2, padx=5, pady=5, sticky="w")
        
        # MFA 模型区域
        ctk.CTkLabel(self, text="MFA 声学模型", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=8, column=0, columnspan=3, padx=10, pady=(20, 5), sticky="w"
        )
        ctk.CTkLabel(self, text="Montreal Forced Aligner 模型，用于语音对齐", text_color="gray").grid(
            row=9, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w"
        )
        
        ctk.CTkLabel(self, text="模型目录:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.mfa_dir_var = ctk.StringVar(value=self.config.get("mfa_dir"))
        ctk.CTkEntry(self, textvariable=self.mfa_dir_var, width=320).grid(row=10, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_mfa_dir).grid(row=10, column=2, padx=5, pady=5)
        
        ctk.CTkLabel(self, text="选择语言:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.mfa_lang_var = ctk.StringVar(value="mandarin")
        ctk.CTkComboBox(
            self, values=["mandarin", "japanese"],
            variable=self.mfa_lang_var, width=200,
            command=self._on_mfa_lang_change
        ).grid(row=11, column=1, padx=5, pady=5, sticky="w")
        self.mfa_lang_desc = ctk.CTkLabel(self, text="中文 (普通话)", text_color="gray")
        self.mfa_lang_desc.grid(row=11, column=2, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(self, text="状态:").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.mfa_status = ctk.CTkLabel(self, text="⏳ 未下载", text_color="gray")
        self.mfa_status.grid(row=12, column=1, padx=5, pady=5, sticky="w")
        self.mfa_download_btn = ctk.CTkButton(self, text="下载模型", command=self._download_mfa_models, width=140)
        self.mfa_download_btn.grid(row=12, column=2, padx=5, pady=5, sticky="w")
        
        self._check_vad_status()

    def _get_model_desc(self):
        info = ConfigManager.WHISPER_MODELS.get(self.whisper_model_var.get(), {})
        return info.get('desc', '')
    
    def _on_model_change(self, choice):
        self.model_desc_label.configure(text=self._get_model_desc())
        self.config.set("whisper_model", choice)
        self.whisper_status.configure(text="⏳ 未加载", text_color="gray")
        self.whisper_pipe = None
    
    def _browse_models_dir(self):
        path = filedialog.askdirectory(title="选择模型下载目录")
        if path:
            self.models_dir_var.set(path)
            self.config.set("models_dir", path)
    
    def _browse_mfa_dir(self):
        path = filedialog.askdirectory(title="选择 MFA 模型目录")
        if path:
            self.mfa_dir_var.set(path)
            self.config.set("mfa_dir", path)
    
    def _on_mfa_lang_change(self, choice):
        from src.mfa_model_downloader import get_available_languages
        self.mfa_lang_desc.configure(text=get_available_languages().get(choice, ""))
    
    def _check_vad_status(self):
        from src.silero_vad_downloader import is_vad_model_downloaded
        if is_vad_model_downloaded(self.config.get("models_dir")):
            self.vad_status.configure(text="✅ 已下载", text_color="green")
        else:
            self.vad_status.configure(text="⏳ 未下载", text_color="gray")
    
    def _download_vad(self):
        if self._download_thread and self._download_thread.is_alive():
            return
        self.vad_btn.configure(state="disabled")
        self.vad_status.configure(text="⏳ 下载中...", text_color="gray")
        self._download_thread = threading.Thread(target=self._do_download_vad, daemon=True)
        self._download_thread.start()
    
    def _do_download_vad(self):
        from src.silero_vad_downloader import download_silero_vad
        self.log_callback("开始下载 Silero VAD 模型...")
        success, result = download_silero_vad(self.config.get("models_dir"), self.log_callback)
        if success:
            self.after(0, lambda: self.vad_status.configure(text="✅ 已下载", text_color="green"))
            self.log_callback(f"VAD 模型已保存: {result}")
        else:
            self.after(0, lambda: self.vad_status.configure(text="❌ 下载失败", text_color="red"))
        self.after(0, lambda: self.vad_btn.configure(state="normal"))
    
    def _download_mfa_models(self):
        if self._download_thread and self._download_thread.is_alive():
            return
        self.mfa_download_btn.configure(state="disabled")
        self.mfa_status.configure(text="⏳ 下载中...", text_color="gray")
        self._download_thread = threading.Thread(target=self._do_download_mfa, daemon=True)
        self._download_thread.start()
    
    def _do_download_mfa(self):
        from src.mfa_model_downloader import download_language_models
        language = self.mfa_lang_var.get()
        output_dir = self.mfa_dir_var.get()
        os.makedirs(output_dir, exist_ok=True)
        self.log_callback(f"开始下载 MFA 模型: {language}")
        success, acoustic_path, dict_path = download_language_models(
            language=language, output_dir=output_dir, progress_callback=self.log_callback
        )
        if success:
            self.after(0, lambda: self.mfa_status.configure(text="✅ 已下载", text_color="green"))
            self.log_callback(f"声学模型: {acoustic_path}")
            self.log_callback(f"字典文件: {dict_path}")
        else:
            self.after(0, lambda: self.mfa_status.configure(text="❌ 下载失败", text_color="red"))
        self.after(0, lambda: self.mfa_download_btn.configure(state="normal"))
    
    def _download_whisper(self):
        if self._download_thread and self._download_thread.is_alive():
            return
        self.whisper_btn.configure(state="disabled")
        self.whisper_status.configure(text="⏳ 加载中...", text_color="gray")
        self._download_thread = threading.Thread(target=self._do_download_whisper, daemon=True)
        self._download_thread.start()
    
    def _do_download_whisper(self):
        try:
            self._update_progress("正在加载 transformers 库...")
            from transformers import pipeline
            import torch
            
            model_key = self.whisper_model_var.get()
            model_name = ConfigManager.WHISPER_MODELS[model_key]["name"]
            cache_dir = os.path.join(self.models_dir_var.get(), "whisper")
            os.makedirs(cache_dir, exist_ok=True)
            
            self._update_progress(f"正在下载/加载 {model_key}...")
            self.log_callback(f"开始加载 Whisper 模型: {model_name}")
            
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                model_kwargs={"cache_dir": cache_dir}
            )
            
            self._update_progress("")
            self.after(0, lambda: self.whisper_status.configure(text="✅ 已就绪", text_color="green"))
            self.after(0, lambda: self.whisper_btn.configure(state="normal", text="重新加载"))
            self.log_callback("Whisper 模型加载完成")
        except Exception as e:
            self._update_progress("")
            self.after(0, lambda: self.whisper_status.configure(text="❌ 加载失败", text_color="red"))
            self.after(0, lambda: self.whisper_btn.configure(state="normal"))
            self.log_callback(f"Whisper 模型加载失败: {e}")
            logger.error(f"Whisper 模型加载失败: {e}", exc_info=True)
    
    def _update_progress(self, text):
        self.after(0, lambda: self.progress_label.configure(text=text))
    
    def get_whisper_pipeline(self):
        return self.whisper_pipe
    
    def get_models_dir(self):
        return self.models_dir_var.get()
    
    def get_mfa_dir(self):
        return self.mfa_dir_var.get()
    
    def get_whisper_model_name(self):
        return ConfigManager.WHISPER_MODELS[self.whisper_model_var.get()]["name"]


class MakeVoiceBankFrame(ctk.CTkFrame):
    """制作音源页面 - 简化工作流"""
    
    def __init__(self, master, log_callback, config: ConfigManager, model_frame: ModelDownloadFrame):
        super().__init__(master)
        self.log_callback = log_callback
        self.config = config
        self.model_frame = model_frame
        self._is_running = False
        self._setup_ui()
        self._check_mfa_status()
    
    def _setup_ui(self):
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        row = 0
        
        # ========== 基本设置 ==========
        ctk.CTkLabel(
            self.scroll_frame, text="基本设置",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=row, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="w")
        row += 1
        
        # 音源名称
        ctk.CTkLabel(self.scroll_frame, text="音源名称:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.source_name_var = ctk.StringVar(value="my_voice")
        ctk.CTkEntry(self.scroll_frame, textvariable=self.source_name_var, width=200).grid(
            row=row, column=1, padx=5, pady=5, sticky="w"
        )
        ctk.CTkLabel(self.scroll_frame, text="输出到 bank/[音源名称]/", text_color="gray").grid(
            row=row, column=2, padx=5, pady=5, sticky="w"
        )
        row += 1
        
        # 输入音频
        ctk.CTkLabel(self.scroll_frame, text="输入音频:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.input_audio_var = ctk.StringVar()
        ctk.CTkEntry(self.scroll_frame, textvariable=self.input_audio_var, width=300).grid(
            row=row, column=1, padx=5, pady=5
        )
        btn_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        btn_frame.grid(row=row, column=2, padx=5, pady=5)
        ctk.CTkButton(btn_frame, text="文件", width=50, command=self._browse_input_file,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="文件夹", width=60, command=self._browse_input_dir,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").pack(side="left", padx=2)
        row += 1
        
        # 输出目录
        ctk.CTkLabel(self.scroll_frame, text="输出目录:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_var = ctk.StringVar(value=self.config.get("bank_dir", "bank"))
        ctk.CTkEntry(self.scroll_frame, textvariable=self.output_dir_var, width=300).grid(
            row=row, column=1, padx=5, pady=5
        )
        ctk.CTkButton(self.scroll_frame, text="浏览", width=60, command=self._browse_output_dir,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").grid(
            row=row, column=2, padx=5, pady=5, sticky="w"
        )
        row += 1
        
        # 分隔线
        ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray50").grid(
            row=row, column=0, columnspan=3, padx=10, pady=15, sticky="ew"
        )
        row += 1
        
        # ========== 模型选择 ==========
        ctk.CTkLabel(
            self.scroll_frame, text="模型选择",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=row, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="w")
        row += 1
        
        # Whisper模型
        ctk.CTkLabel(self.scroll_frame, text="Whisper模型:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.whisper_combo = ctk.CTkComboBox(
            self.scroll_frame, values=["(扫描中...)"], width=250
        )
        self.whisper_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.scroll_frame, text="刷新", width=60, command=self._refresh_whisper_models,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").grid(
            row=row, column=2, padx=5, pady=5, sticky="w"
        )
        row += 1
        
        # MFA字典
        ctk.CTkLabel(self.scroll_frame, text="MFA字典:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.dict_combo = ctk.CTkComboBox(self.scroll_frame, values=["(扫描中...)"], width=250)
        self.dict_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1
        
        # MFA声学模型
        ctk.CTkLabel(self.scroll_frame, text="MFA声学模型:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.acoustic_combo = ctk.CTkComboBox(self.scroll_frame, values=["(扫描中...)"], width=250)
        self.acoustic_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.scroll_frame, text="刷新", width=60, command=self._refresh_mfa_models,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").grid(
            row=row, column=2, padx=5, pady=5, sticky="w"
        )
        row += 1
        
        # 语言
        ctk.CTkLabel(self.scroll_frame, text="转录语言:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.language_var = ctk.StringVar(value="chinese")
        ctk.CTkComboBox(
            self.scroll_frame, values=["chinese", "japanese", "english"],
            variable=self.language_var, width=150
        ).grid(row=row, column=1, padx=5, pady=5, sticky="w")
        row += 1
        
        # 分隔线
        ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray50").grid(
            row=row, column=0, columnspan=3, padx=10, pady=15, sticky="ew"
        )
        row += 1
        
        # ========== MFA状态 ==========
        self.mfa_status_label = ctk.CTkLabel(
            self.scroll_frame, text="⏳ 检查 MFA 环境...",
            font=ctk.CTkFont(size=12)
        )
        self.mfa_status_label.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        row += 1
        
        # 分隔线
        ctk.CTkFrame(self.scroll_frame, height=2, fg_color="gray50").grid(
            row=row, column=0, columnspan=3, padx=10, pady=15, sticky="ew"
        )
        row += 1
        
        # ========== 执行按钮 ==========
        ctk.CTkLabel(
            self.scroll_frame, text="执行流程",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=row, column=0, columnspan=3, padx=10, pady=(10, 15), sticky="w")
        row += 1
        
        # 按钮容器 - 优化排版
        btn_container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        btn_container.grid(row=row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # 分步执行按钮 - 降饱和颜色
        self.step0_btn = ctk.CTkButton(
            btn_container, text="步骤0: 切片+转录",
            command=self._run_step0, width=150, height=36,
            fg_color="#5c7a9a", hover_color="#4a6888"
        )
        self.step0_btn.pack(side="left", padx=8)
        
        self.step1_btn = ctk.CTkButton(
            btn_container, text="步骤1: MFA对齐",
            command=self._run_step1, width=150, height=36,
            fg_color="#6a9a7a", hover_color="#588868"
        )
        self.step1_btn.pack(side="left", padx=8)
        row += 1
        
        # 一键执行 - 降饱和
        self.full_btn = ctk.CTkButton(
            self.scroll_frame, text="▶ 一键执行全部流程",
            command=self._run_full, width=320, height=40,
            fg_color="#8a6a8a", hover_color="#785878",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.full_btn.grid(row=row, column=0, columnspan=3, pady=15)
        row += 1
        
        # 进度提示
        self.progress_label = ctk.CTkLabel(self.scroll_frame, text="", text_color="gray")
        self.progress_label.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # 初始化模型列表
        self.after(500, self._refresh_all_models)
    
    def _check_mfa_status(self):
        from src.mfa_runner import check_mfa_available
        if check_mfa_available():
            self.mfa_status_label.configure(text="✅ MFA 环境已就绪", text_color="green")
        else:
            self.mfa_status_label.configure(text="❌ MFA 环境不可用，请检查 tools/mfa_engine", text_color="red")
    
    def _refresh_all_models(self):
        self._refresh_whisper_models()
        self._refresh_mfa_models()
    
    def _refresh_whisper_models(self):
        from src.pipeline import scan_whisper_models
        models_dir = self.model_frame.get_models_dir()
        models = scan_whisper_models(models_dir)
        
        all_models = list(ConfigManager.WHISPER_MODELS.values())
        preset_names = [m["name"] for m in all_models]
        
        for m in models:
            if m not in preset_names:
                preset_names.append(m)
        
        if preset_names:
            self.whisper_combo.configure(values=preset_names)
            self.whisper_combo.set(preset_names[0])
        else:
            self.whisper_combo.configure(values=["openai/whisper-small"])
            self.whisper_combo.set("openai/whisper-small")
    
    def _refresh_mfa_models(self):
        from src.pipeline import scan_mfa_models
        mfa_dir = self.model_frame.get_mfa_dir()
        models = scan_mfa_models(os.path.dirname(mfa_dir))
        
        if models["dictionary"]:
            self.dict_combo.configure(values=models["dictionary"])
            self.dict_combo.set(models["dictionary"][0])
        else:
            self.dict_combo.configure(values=["(未找到字典文件)"])
            self.dict_combo.set("(未找到字典文件)")
        
        if models["acoustic"]:
            self.acoustic_combo.configure(values=models["acoustic"])
            self.acoustic_combo.set(models["acoustic"][0])
        else:
            self.acoustic_combo.configure(values=["(未找到声学模型)"])
            self.acoustic_combo.set("(未找到声学模型)")
    
    def _browse_input_file(self):
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("所有文件", "*.*")]
        )
        if path:
            self.input_audio_var.set(path)
    
    def _browse_input_dir(self):
        path = filedialog.askdirectory(title="选择音频文件夹")
        if path:
            self.input_audio_var.set(path)
    
    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_dir_var.set(path)
            self.config.set("bank_dir", path)
    
    def _get_pipeline_config(self):
        """获取流水线配置"""
        from src.pipeline import PipelineConfig
        
        mfa_dir = self.model_frame.get_mfa_dir()
        dict_file = self.dict_combo.get()
        acoustic_file = self.acoustic_combo.get()
        
        dict_path = None
        if dict_file and not dict_file.startswith("("):
            dict_path = os.path.join(mfa_dir, dict_file)
        
        acoustic_path = None
        if acoustic_file and not acoustic_file.startswith("("):
            acoustic_path = os.path.join(mfa_dir, acoustic_file)
        
        return PipelineConfig(
            source_name=self.source_name_var.get(),
            input_path=self.input_audio_var.get(),
            output_base_dir=self.output_dir_var.get(),
            models_dir=self.model_frame.get_models_dir(),
            whisper_model=self.whisper_combo.get(),
            mfa_dict_path=dict_path,
            mfa_model_path=acoustic_path,
            language=self.language_var.get()
        )
    
    def _set_buttons_state(self, state: str):
        """设置所有按钮状态"""
        for btn in [self.step0_btn, self.step1_btn, self.full_btn]:
            btn.configure(state=state)
    
    def _run_step0(self):
        if self._is_running:
            return
        if not self._validate_input():
            return
        self._is_running = True
        self._set_buttons_state("disabled")
        threading.Thread(target=self._do_step0, daemon=True).start()
    
    def _do_step0(self):
        from src.pipeline import VoiceBankPipeline
        config = self._get_pipeline_config()
        pipeline = VoiceBankPipeline(config, self.log_callback)
        
        self.log_callback("=" * 50)
        self.log_callback("【步骤0】音频预处理 (VAD切片 + Whisper转录)")
        success, msg, _ = pipeline.step0_preprocess()
        
        if success:
            self.log_callback(f"✅ {msg}")
        else:
            self.log_callback(f"❌ {msg}")
        self.log_callback("=" * 50)
        
        self.after(0, lambda: self._set_buttons_state("normal"))
        self._is_running = False
    
    def _run_step1(self):
        if self._is_running:
            return
        if not self._validate_source_name():
            return
        self._is_running = True
        self._set_buttons_state("disabled")
        threading.Thread(target=self._do_step1, daemon=True).start()
    
    def _do_step1(self):
        from src.pipeline import VoiceBankPipeline
        config = self._get_pipeline_config()
        pipeline = VoiceBankPipeline(config, self.log_callback)
        
        self.log_callback("=" * 50)
        self.log_callback("【步骤1】MFA语音对齐")
        success, msg = pipeline.step1_mfa_align()
        
        if success:
            self.log_callback(f"✅ {msg}")
        else:
            self.log_callback(f"❌ {msg}")
        self.log_callback("=" * 50)
        
        self.after(0, lambda: self._set_buttons_state("normal"))
        self._is_running = False
    
    def _run_full(self):
        if self._is_running:
            return
        if not self._validate_input():
            return
        self._is_running = True
        self._set_buttons_state("disabled")
        threading.Thread(target=self._do_full, daemon=True).start()
    
    def _do_full(self):
        from src.pipeline import VoiceBankPipeline
        config = self._get_pipeline_config()
        pipeline = VoiceBankPipeline(config, self.log_callback)
        
        success, msg = pipeline.run_make_pipeline()
        
        if not success:
            self.log_callback(f"❌ 流程中断: {msg}")
        
        self.after(0, lambda: self._set_buttons_state("normal"))
        self._is_running = False
    
    def _validate_input(self) -> bool:
        """验证输入"""
        if not self.source_name_var.get().strip():
            messagebox.showerror("错误", "请输入音源名称")
            return False
        if not self.input_audio_var.get().strip():
            messagebox.showerror("错误", "请选择输入音频")
            return False
        if not self.output_dir_var.get().strip():
            messagebox.showerror("错误", "请选择输出目录")
            return False
        return True
    
    def _validate_source_name(self) -> bool:
        """验证音源名称"""
        if not self.source_name_var.get().strip():
            messagebox.showerror("错误", "请输入音源名称")
            return False
        return True


class ExportSettingsDialog(ctk.CTkToplevel):
    """导出设置弹窗"""
    
    def __init__(self, master, plugin, voice_bank: str, bank_dir: str, log_callback):
        super().__init__(master)
        self.plugin = plugin
        self.voice_bank = voice_bank
        self.bank_dir = bank_dir
        self.log_callback = log_callback
        self._option_widgets = {}
        self._is_running = False
        
        self.title(f"导出设置 - {plugin.name}")
        self.geometry("500x400")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()
        
        self._setup_ui()
        self._center_window()
    
    def _center_window(self):
        """居中显示"""
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
    
    def _setup_ui(self):
        from src.export_plugins import OptionType
        
        # 标题
        header = ctk.CTkFrame(self)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(
            header, text=self.plugin.name,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text=self.plugin.description,
            text_color="gray"
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text=f"音源: {self.voice_bank}",
            text_color="gray"
        ).pack(anchor="w")
        
        # 选项区域（可滚动）
        self.options_frame = ctk.CTkScrollableFrame(self)
        self.options_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 动态生成选项控件
        for opt in self.plugin.get_options():
            self._create_option_widget(opt)
        
        # 底部按钮
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        self.cancel_btn = ctk.CTkButton(
            btn_frame, text="取消", width=80,
            fg_color="gray", command=self.destroy
        )
        self.cancel_btn.pack(side="left", padx=5)
        
        self.reset_btn = ctk.CTkButton(
            btn_frame, text="恢复默认", width=100,
            fg_color="#607D8B", command=self._reset_defaults
        )
        self.reset_btn.pack(side="left", padx=5)
        
        self.export_btn = ctk.CTkButton(
            btn_frame, text="导出", width=100,
            fg_color="#6a9a7a", hover_color="#588868", command=self._do_export
        )
        self.export_btn.pack(side="right", padx=5)
    
    def _create_option_widget(self, opt):
        """创建选项控件"""
        from src.export_plugins import OptionType
        
        frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        if opt.option_type == OptionType.LABEL:
            ctk.CTkLabel(frame, text=opt.label, text_color="gray").pack(anchor="w")
            return
        
        ctk.CTkLabel(frame, text=opt.label).pack(anchor="w")
        
        if opt.option_type == OptionType.TEXT:
            var = ctk.StringVar(value=str(opt.default or ""))
            widget = ctk.CTkEntry(frame, textvariable=var, width=300)
            widget.pack(anchor="w", pady=2)
            self._option_widgets[opt.key] = ("text", var)
            
        elif opt.option_type == OptionType.NUMBER:
            var = ctk.StringVar(value=str(opt.default or 0))
            widget = ctk.CTkEntry(frame, textvariable=var, width=150)
            widget.pack(anchor="w", pady=2)
            self._option_widgets[opt.key] = ("number", var, opt.min_value, opt.max_value)
            
        elif opt.option_type == OptionType.SWITCH:
            var = ctk.BooleanVar(value=bool(opt.default))
            widget = ctk.CTkSwitch(frame, text="", variable=var)
            widget.pack(anchor="w", pady=2)
            self._option_widgets[opt.key] = ("switch", var)
            
        elif opt.option_type == OptionType.COMBO:
            var = ctk.StringVar(value=str(opt.default or ""))
            widget = ctk.CTkComboBox(frame, values=opt.choices, variable=var, width=200)
            widget.pack(anchor="w", pady=2)
            self._option_widgets[opt.key] = ("combo", var)
            
        elif opt.option_type == OptionType.FILE:
            var = ctk.StringVar(value=str(opt.default or ""))
            entry_frame = ctk.CTkFrame(frame, fg_color="transparent")
            entry_frame.pack(anchor="w", pady=2)
            entry = ctk.CTkEntry(entry_frame, textvariable=var, width=250)
            entry.pack(side="left")
            btn = ctk.CTkButton(
                entry_frame, text="浏览", width=60,
                command=lambda v=var, ft=opt.file_types: self._browse_file(v, ft)
            )
            btn.pack(side="left", padx=5)
            self._option_widgets[opt.key] = ("file", var)
            
        elif opt.option_type == OptionType.FOLDER:
            var = ctk.StringVar(value=str(opt.default or ""))
            entry_frame = ctk.CTkFrame(frame, fg_color="transparent")
            entry_frame.pack(anchor="w", pady=2)
            entry = ctk.CTkEntry(entry_frame, textvariable=var, width=250)
            entry.pack(side="left")
            btn = ctk.CTkButton(
                entry_frame, text="浏览", width=60,
                command=lambda v=var: self._browse_folder(v)
            )
            btn.pack(side="left", padx=5)
            self._option_widgets[opt.key] = ("folder", var)
        
        if opt.description:
            ctk.CTkLabel(
                frame, text=opt.description,
                text_color="gray", font=ctk.CTkFont(size=11)
            ).pack(anchor="w")
    
    def _browse_file(self, var, file_types):
        ft = file_types if file_types else [("所有文件", "*.*")]
        path = filedialog.askopenfilename(filetypes=ft)
        if path:
            var.set(path)
    
    def _browse_folder(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
    
    def _get_options_values(self) -> dict:
        values = {}
        for key, widget_info in self._option_widgets.items():
            widget_type = widget_info[0]
            var = widget_info[1]
            
            if widget_type == "number":
                try:
                    val = float(var.get())
                    min_val = widget_info[2]
                    max_val = widget_info[3]
                    if min_val is not None:
                        val = max(min_val, val)
                    if max_val is not None:
                        val = min(max_val, val)
                    values[key] = int(val) if val == int(val) else val
                except ValueError:
                    values[key] = 0
            elif widget_type == "switch":
                values[key] = var.get()
            else:
                values[key] = var.get()
        
        return values
    
    def _reset_defaults(self):
        for opt in self.plugin.get_options():
            if opt.key in self._option_widgets:
                widget_info = self._option_widgets[opt.key]
                var = widget_info[1]
                if widget_info[0] == "switch":
                    var.set(bool(opt.default))
                else:
                    var.set(str(opt.default or ""))
    
    def _do_export(self):
        if self._is_running:
            return
        
        self._is_running = True
        self._set_buttons_state("disabled")
        
        options = self._get_options_values()
        threading.Thread(target=self._run_export, args=(options,), daemon=True).start()
    
    def _run_export(self, options: dict):
        self.log_callback("=" * 50)
        self.log_callback(f"【{self.plugin.name}】音源: {self.voice_bank}")
        
        self.plugin.set_progress_callback(self.log_callback)
        success, msg = self.plugin.export(self.voice_bank, self.bank_dir, options)
        
        if success:
            self.log_callback(f"✅ {msg}")
        else:
            self.log_callback(f"❌ {msg}")
        self.log_callback("=" * 50)
        
        self.after(0, self._on_export_complete)
    
    def _on_export_complete(self):
        self._is_running = False
        self._set_buttons_state("normal")
        messagebox.showinfo("完成", "导出完成")
    
    def _set_buttons_state(self, state: str):
        self.cancel_btn.configure(state=state)
        self.reset_btn.configure(state=state)
        self.export_btn.configure(state=state)


class ExportVoiceBankFrame(ctk.CTkFrame):
    """导出音源页面"""
    
    def __init__(self, master, log_callback, config: ConfigManager):
        super().__init__(master)
        self.log_callback = log_callback
        self.config = config
        self._plugins = {}
        self._load_plugins()
        self._setup_ui()
        self.after(500, self._refresh_voice_banks)
    
    def _load_plugins(self):
        from src.export_plugins import load_plugins
        plugins_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "export_plugins"
        )
        self._plugins = load_plugins(plugins_dir)
    
    def _setup_ui(self):
        # 音源选择区域
        ctk.CTkLabel(
            self, text="选择音源",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        ctk.CTkLabel(self, text="音源:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.voice_bank_var = ctk.StringVar()
        self.voice_bank_combo = ctk.CTkComboBox(
            self, values=["(扫描中...)"],
            variable=self.voice_bank_var, width=250,
            command=self._on_voice_bank_change
        )
        self.voice_bank_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self, text="刷新", width=60, command=self._refresh_voice_banks,
                      fg_color="#5a6a7a", hover_color="#4a5a6a").grid(
            row=1, column=2, padx=5, pady=5, sticky="w"
        )
        
        # 音源信息
        self.info_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.info_label.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # 分隔线
        ctk.CTkFrame(self, height=2, fg_color="gray50").grid(
            row=3, column=0, columnspan=3, padx=10, pady=15, sticky="ew"
        )
        
        # 导出方式区域
        ctk.CTkLabel(
            self, text="导出方式",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=4, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        # 插件列表（可滚动）
        self.plugins_frame = ctk.CTkScrollableFrame(self, height=250)
        self.plugins_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        
        # 动态生成插件卡片
        self._create_plugin_cards()
        
        # 配置行列权重
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(5, weight=1)
    
    def _create_plugin_cards(self):
        """创建插件卡片 - 整个卡片可点击"""
        for idx, (name, plugin) in enumerate(self._plugins.items()):
            # 卡片容器 - 作为按钮
            card = ctk.CTkFrame(
                self.plugins_frame,
                fg_color=("#e8e8e8", "#2a2a2a"),
                corner_radius=8
            )
            card.pack(fill="x", pady=6, padx=4)
            card.bind("<Enter>", lambda e, c=card: c.configure(fg_color=("#d8d8d8", "#3a3a3a")))
            card.bind("<Leave>", lambda e, c=card: c.configure(fg_color=("#e8e8e8", "#2a2a2a")))
            card.bind("<Button-1>", lambda e, p=plugin: self._open_plugin_settings(p))
            
            # 内容容器
            content = ctk.CTkFrame(card, fg_color="transparent")
            content.pack(fill="x", padx=12, pady=10)
            content.bind("<Button-1>", lambda e, p=plugin: self._open_plugin_settings(p))
            
            # 插件名称 - 白色，较大，左中部
            name_label = ctk.CTkLabel(
                content, text=name,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=("#1a1a1a", "#ffffff")
            )
            name_label.pack(anchor="w")
            name_label.bind("<Button-1>", lambda e, p=plugin: self._open_plugin_settings(p))
            
            # 描述
            desc_label = ctk.CTkLabel(
                content, text=plugin.description,
                text_color="gray",
                font=ctk.CTkFont(size=12)
            )
            desc_label.pack(anchor="w", pady=(2, 0))
            desc_label.bind("<Button-1>", lambda e, p=plugin: self._open_plugin_settings(p))
            
            # 作者和版本
            if plugin.author:
                meta_label = ctk.CTkLabel(
                    content, text=f"作者: {plugin.author} | 版本: {plugin.version}",
                    text_color="gray",
                    font=ctk.CTkFont(size=10)
                )
                meta_label.pack(anchor="w", pady=(2, 0))
                meta_label.bind("<Button-1>", lambda e, p=plugin: self._open_plugin_settings(p))
    
    def _open_plugin_settings(self, plugin):
        """打开插件设置弹窗"""
        voice_bank = self.voice_bank_var.get()
        if not voice_bank or voice_bank.startswith("("):
            messagebox.showerror("错误", "请先选择有效的音源")
            return
        
        bank_dir = self.config.get("bank_dir", "bank")
        ExportSettingsDialog(self, plugin, voice_bank, bank_dir, self.log_callback)
    
    def _refresh_voice_banks(self):
        """刷新音源列表"""
        bank_dir = self.config.get("bank_dir", "bank")
        voice_banks = []
        
        if os.path.exists(bank_dir):
            for name in os.listdir(bank_dir):
                source_dir = os.path.join(bank_dir, name)
                if os.path.isdir(source_dir) and not name.startswith('.'):
                    slices_dir = os.path.join(source_dir, "slices")
                    textgrid_dir = os.path.join(source_dir, "textgrid")
                    if os.path.exists(slices_dir) or os.path.exists(textgrid_dir):
                        voice_banks.append(name)
        
        if voice_banks:
            self.voice_bank_combo.configure(values=voice_banks)
            self.voice_bank_combo.set(voice_banks[0])
            self._on_voice_bank_change(voice_banks[0])
        else:
            self.voice_bank_combo.configure(values=["(未找到音源)"])
            self.voice_bank_combo.set("(未找到音源)")
            self.info_label.configure(text="")
    
    def _on_voice_bank_change(self, choice):
        """音源选择变化"""
        if choice.startswith("("):
            self.info_label.configure(text="")
            return
        
        bank_dir = self.config.get("bank_dir", "bank")
        source_dir = os.path.join(bank_dir, choice)
        slices_dir = os.path.join(source_dir, "slices")
        textgrid_dir = os.path.join(source_dir, "textgrid")
        
        slices_count = 0
        textgrid_count = 0
        
        if os.path.exists(slices_dir):
            slices_count = len([f for f in os.listdir(slices_dir) if f.endswith('.wav')])
        if os.path.exists(textgrid_dir):
            textgrid_count = len([f for f in os.listdir(textgrid_dir) if f.endswith('.TextGrid')])
        
        self.info_label.configure(
            text=f"切片: {slices_count} 个 | TextGrid: {textgrid_count} 个"
        )


class SettingsFrame(ctk.CTkFrame):
    """设置页面"""
    
    def __init__(self, master, config: ConfigManager, on_log_toggle):
        super().__init__(master)
        self.config = config
        self.on_log_toggle = on_log_toggle
        self._setup_ui()
    
    def _setup_ui(self):
        # 标题
        ctk.CTkLabel(
            self, text="应用设置",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 20))
        
        # 日志设置区域
        log_frame = ctk.CTkFrame(self, fg_color="transparent")
        log_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            log_frame, text="界面设置",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # 显示日志开关
        log_switch_frame = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_switch_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(log_switch_frame, text="显示日志输出面板").pack(side="left")
        
        self.show_log_var = ctk.BooleanVar(value=self.config.get("show_log", False))
        self.log_switch = ctk.CTkSwitch(
            log_switch_frame, text="",
            variable=self.show_log_var,
            command=self._on_log_switch_change
        )
        self.log_switch.pack(side="right")
        
        ctk.CTkLabel(
            log_frame, text="开启后将在主界面底部显示日志输出区域",
            text_color="gray", font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(2, 0))
        
        # 分隔线
        ctk.CTkFrame(self, height=1, fg_color="gray50").pack(fill="x", padx=15, pady=20)
        
        # 关于区域
        about_frame = ctk.CTkFrame(self, fg_color="transparent")
        about_frame.pack(fill="x", padx=15, pady=10)
        
        ctk.CTkLabel(
            about_frame, text="关于",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        ctk.CTkLabel(
            about_frame, text="语音数据集处理工具",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            about_frame, text="基于 CustomTkinter 构建",
            text_color="gray", font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(2, 0))
    
    def _on_log_switch_change(self):
        """日志开关变化"""
        show_log = self.show_log_var.get()
        self.config.set("show_log", show_log)
        self.on_log_toggle(show_log)


class App(ctk.CTk):
    """主应用窗口"""
    
    def __init__(self):
        super().__init__()
        self.title("语音数据集处理工具")
        self.geometry("750x720")
        self.minsize(700, 620)
        
        self.config = ConfigManager()
        self._setup_ui()
        logger.info("应用启动")
    
    def _setup_ui(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        tab1 = self.tabview.add("模型下载")
        tab2 = self.tabview.add("制作音源")
        tab3 = self.tabview.add("导出音源")
        tab4 = self.tabview.add("设置")
        
        self.download_frame = ModelDownloadFrame(tab1, self._log, self.config)
        self.download_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.make_frame = MakeVoiceBankFrame(tab2, self._log, self.config, self.download_frame)
        self.make_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.export_frame = ExportVoiceBankFrame(tab3, self._log, self.config)
        self.export_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.settings_frame = SettingsFrame(tab4, self.config, self._toggle_log_panel)
        self.settings_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 日志区域 - 默认隐藏
        self.log_frame = ctk.CTkFrame(self)
        ctk.CTkLabel(self.log_frame, text="日志输出:").pack(anchor="w", padx=5, pady=2)
        self.log_text = ctk.CTkTextbox(self.log_frame, height=100)
        self.log_text.pack(fill="x", padx=5, pady=5)
        
        # 根据配置决定是否显示日志
        if self.config.get("show_log", False):
            self.log_frame.pack(fill="x", padx=10, pady=(0, 10))
    
    def _toggle_log_panel(self, show: bool):
        """切换日志面板显示"""
        if show:
            self.log_frame.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.log_frame.pack_forget()
    
    def _log(self, message):
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
