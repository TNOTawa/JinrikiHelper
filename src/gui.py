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


class TextGridToBankFrame(ctk.CTkFrame):
    """TextGrid转音频库功能面板"""
    
    def __init__(self, master, log_callback):
        super().__init__(master)
        self.log_callback = log_callback
        self._setup_ui()
    
    def _setup_ui(self):
        # WAV目录
        ctk.CTkLabel(self, text="① WAV文件目录:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.wav_dir_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.wav_dir_var, width=400).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_wav_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # TextGrid目录
        ctk.CTkLabel(self, text="② TextGrid目录:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.tg_dir_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.tg_dir_var, width=400).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_tg_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # 输出目录
        ctk.CTkLabel(self, text="③ 输出目录:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.save_dir_var = ctk.StringVar(value="bank")
        ctk.CTkEntry(self, textvariable=self.save_dir_var, width=400).grid(row=2, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_save_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # 执行按钮
        ctk.CTkButton(self, text="④ 开始转换", command=self._run).grid(row=3, column=1, pady=20)
    
    def _browse_wav_dir(self):
        path = filedialog.askdirectory(title="选择WAV文件目录")
        if path:
            self.wav_dir_var.set(path)
    
    def _browse_tg_dir(self):
        path = filedialog.askdirectory(title="选择TextGrid目录")
        if path:
            self.tg_dir_var.set(path)
    
    def _browse_save_dir(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.save_dir_var.set(path)
    
    def _run(self):
        wav_dir = self.wav_dir_var.get()
        tg_dir = self.tg_dir_var.get()
        save_dir = self.save_dir_var.get()
        
        if not wav_dir or not tg_dir or not save_dir:
            messagebox.showerror("错误", "请填写所有目录路径")
            return
        
        threading.Thread(target=self._process, args=(wav_dir, tg_dir, save_dir), daemon=True).start()
    
    def _process(self, wav_dir, tg_dir, save_dir):
        import textgrid
        import glob
        import audiofile
        
        self.log_callback("开始TextGrid转音频库...")
        logger.info(f"WAV目录: {wav_dir}, TextGrid目录: {tg_dir}, 输出目录: {save_dir}")
        
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            tg_files = glob.glob(os.path.join(tg_dir, '*.TextGrid'))
            total = len(tg_files)
            
            for i, path in enumerate(tg_files):
                basename = os.path.basename(path).replace('.TextGrid', '.wav')
                wav_path = os.path.join(wav_dir, basename)
                
                if not os.path.exists(wav_path):
                    self.log_callback(f"警告: 找不到对应WAV文件 {wav_path}")
                    continue
                
                tg = textgrid.TextGrid.fromFile(path)
                audio, sr = audiofile.read(wav_path)
                
                for word in tg[0]:
                    if word.mark in ['SP', 'AP', '']:
                        continue
                    
                    word_text = word.mark.split(':')[0]
                    word_dir = os.path.join(save_dir, word_text)
                    
                    if not os.path.exists(word_dir):
                        os.makedirs(word_dir)
                    
                    index = 1
                    while True:
                        filename = os.path.join(word_dir, f'{index}.wav')
                        if not os.path.exists(filename):
                            break
                        index += 1
                    
                    start_sample = int(word.minTime * sr)
                    end_sample = int(word.maxTime * sr)
                    audiofile.write(filename, audio[start_sample:end_sample], sr)
                
                self.log_callback(f"进度: {i+1}/{total} - {basename}")
            
            self.log_callback("TextGrid转音频库完成!")
            logger.info("TextGrid转音频库处理完成")
        except Exception as e:
            self.log_callback(f"错误: {str(e)}")
            logger.error(f"处理失败: {e}", exc_info=True)


class BankSortFrame(ctk.CTkFrame):
    """音频库排序功能面板"""
    
    def __init__(self, master, log_callback):
        super().__init__(master)
        self.log_callback = log_callback
        self._setup_ui()
    
    def _setup_ui(self):
        # 音频库目录
        ctk.CTkLabel(self, text="① 音频库目录:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.bank_dir_var = ctk.StringVar(value="bank")
        ctk.CTkEntry(self, textvariable=self.bank_dir_var, width=400).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_bank_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # 最大数量
        ctk.CTkLabel(self, text="② 每词最大数量:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.max_count_var = ctk.StringVar(value="100")
        ctk.CTkEntry(self, textvariable=self.max_count_var, width=100).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 执行按钮
        ctk.CTkButton(self, text="③ 开始排序", command=self._run).grid(row=2, column=1, pady=20)
    
    def _browse_bank_dir(self):
        path = filedialog.askdirectory(title="选择音频库目录")
        if path:
            self.bank_dir_var.set(path)
    
    def _run(self):
        bank_dir = self.bank_dir_var.get()
        try:
            max_count = int(self.max_count_var.get())
        except ValueError:
            messagebox.showerror("错误", "最大数量必须是整数")
            return
        
        if not bank_dir:
            messagebox.showerror("错误", "请选择音频库目录")
            return
        
        threading.Thread(target=self._process, args=(bank_dir, max_count), daemon=True).start()
    
    def _process(self, bank_dir, max_count):
        import glob
        import audiofile
        import shutil
        
        self.log_callback("开始音频库排序...")
        logger.info(f"音频库目录: {bank_dir}, 最大数量: {max_count}")
        
        try:
            stats = {}
            wav_files = glob.glob(os.path.join(bank_dir, '**', '*.wav'), recursive=True)
            
            self.log_callback(f"扫描到 {len(wav_files)} 个WAV文件")
            
            for path in wav_files:
                rel_path = os.path.relpath(path, bank_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    word = parts[0]
                    filename = parts[-1]
                    if word not in stats:
                        stats[word] = []
                    stats[word].append((path, audiofile.duration(path)))
            
            self.log_callback(f"统计到 {len(stats)} 个词条")
            
            for word in stats:
                sorted_files = sorted(stats[word], key=lambda x: -x[1])
                for index, (src_path, duration) in enumerate(sorted_files):
                    if index >= max_count:
                        break
                    dst_path = os.path.join(bank_dir, f'{word}_{index}.wav')
                    shutil.copyfile(src_path, dst_path)
                self.log_callback(f"处理词条: {word} ({min(len(sorted_files), max_count)} 个文件)")
            
            self.log_callback("音频库排序完成!")
            logger.info("音频库排序处理完成")
        except Exception as e:
            self.log_callback(f"错误: {str(e)}")
            logger.error(f"处理失败: {e}", exc_info=True)


class ModelDownloadFrame(ctk.CTkFrame):
    """模型配置功能面板"""
    
    # Whisper 模型选项
    WHISPER_MODELS = {
        "whisper-small": {
            "name": "openai/whisper-small",
            "desc": "小型模型，约500MB，速度快",
            "size": "~500MB"
        },
        "whisper-medium": {
            "name": "openai/whisper-medium",
            "desc": "中型模型，约1.5GB，精度更高",
            "size": "~1.5GB"
        }
    }
    
    # 配置文件路径
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    
    def __init__(self, master, log_callback):
        super().__init__(master)
        self.log_callback = log_callback
        self.whisper_pipe = None
        self._download_thread = None
        self._load_config()
        self._setup_ui()
    
    def _get_default_models_dir(self):
        """获取默认模型目录"""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    def _load_config(self):
        """加载配置"""
        self.config = {
            "whisper_model": "whisper-small",
            "models_dir": self._get_default_models_dir(),
            "mfa_dir": os.path.join(self._get_default_models_dir(), "mfa")
        }
        
        if os.path.exists(self.CONFIG_FILE):
            try:
                import json
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.config.update(saved)
                logger.info(f"已加载配置: {self.CONFIG_FILE}")
            except Exception as e:
                logger.warning(f"加载配置失败: {e}")
    
    def _save_config(self):
        """保存配置"""
        try:
            import json
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存: {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _setup_ui(self):
        # ========== Whisper 模型区域 ==========
        whisper_label = ctk.CTkLabel(
            self,
            text="Whisper 语音识别模型",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        whisper_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        # 模型选择
        ctk.CTkLabel(self, text="模型版本:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.whisper_model_var = ctk.StringVar(value=self.config["whisper_model"])
        self.model_dropdown = ctk.CTkComboBox(
            self,
            values=list(self.WHISPER_MODELS.keys()),
            variable=self.whisper_model_var,
            width=200,
            command=self._on_model_change
        )
        self.model_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 模型说明
        self.model_desc_label = ctk.CTkLabel(
            self,
            text=self._get_model_desc(),
            text_color="gray"
        )
        self.model_desc_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        
        # 下载目录
        ctk.CTkLabel(self, text="下载目录:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.models_dir_var = ctk.StringVar(value=self.config["models_dir"])
        ctk.CTkEntry(self, textvariable=self.models_dir_var, width=320).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_models_dir).grid(row=2, column=2, padx=5, pady=5, sticky="w")
        
        # Whisper 状态和按钮
        ctk.CTkLabel(self, text="状态:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.whisper_status = ctk.CTkLabel(self, text="⏳ 未加载", text_color="gray")
        self.whisper_status.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        self.whisper_btn = ctk.CTkButton(
            self,
            text="下载 / 加载模型",
            command=self._download_whisper,
            width=140
        )
        self.whisper_btn.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        
        # 下载进度
        self.progress_label = ctk.CTkLabel(self, text="", text_color="gray")
        self.progress_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        # ========== MFA 模型区域 ==========
        mfa_label = ctk.CTkLabel(
            self,
            text="MFA 声学模型",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        mfa_label.grid(row=5, column=0, columnspan=3, padx=10, pady=(20, 5), sticky="w")
        
        mfa_desc = ctk.CTkLabel(
            self,
            text="Montreal Forced Aligner 模型，用于语音对齐",
            text_color="gray"
        )
        mfa_desc.grid(row=6, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w")
        
        # MFA 模型目录
        ctk.CTkLabel(self, text="模型目录:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.mfa_dir_var = ctk.StringVar(value=self.config["mfa_dir"])
        ctk.CTkEntry(self, textvariable=self.mfa_dir_var, width=320).grid(row=7, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_mfa_dir).grid(row=7, column=2, padx=5, pady=5)
        
        # MFA 语言选择
        ctk.CTkLabel(self, text="选择语言:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.mfa_lang_var = ctk.StringVar(value="mandarin")
        self.mfa_lang_dropdown = ctk.CTkComboBox(
            self,
            values=["mandarin", "japanese"],
            variable=self.mfa_lang_var,
            width=200,
            command=self._on_mfa_lang_change
        )
        self.mfa_lang_dropdown.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        
        self.mfa_lang_desc = ctk.CTkLabel(self, text="中文 (普通话)", text_color="gray")
        self.mfa_lang_desc.grid(row=8, column=2, padx=5, pady=5, sticky="w")
        
        # MFA 下载按钮和状态
        ctk.CTkLabel(self, text="状态:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.mfa_status = ctk.CTkLabel(self, text="⏳ 未下载", text_color="gray")
        self.mfa_status.grid(row=9, column=1, padx=5, pady=5, sticky="w")
        
        self.mfa_download_btn = ctk.CTkButton(
            self,
            text="下载模型",
            command=self._download_mfa_models,
            width=140
        )
        self.mfa_download_btn.grid(row=9, column=2, padx=5, pady=5, sticky="w")
        
        # MFA 文件列表
        ctk.CTkLabel(self, text="已有文件:").grid(row=10, column=0, padx=10, pady=(10, 5), sticky="nw")
        self.mfa_files_text = ctk.CTkTextbox(self, height=70, width=400)
        self.mfa_files_text.grid(row=10, column=1, columnspan=2, padx=5, pady=(10, 5), sticky="w")
        self.mfa_files_text.insert("end", "选择目录后显示文件列表")
        self.mfa_files_text.configure(state="disabled")
        
        # 初始扫描
        self._scan_mfa_dir()
    
    def _get_model_desc(self):
        """获取当前选中模型的描述"""
        model_key = self.whisper_model_var.get()
        info = self.WHISPER_MODELS.get(model_key, {})
        return f"{info.get('desc', '')} ({info.get('size', '')})"
    
    def _on_model_change(self, choice):
        """模型选择变更"""
        self.model_desc_label.configure(text=self._get_model_desc())
        self.config["whisper_model"] = choice
        self._save_config()
        # 重置状态
        self.whisper_status.configure(text="⏳ 未加载", text_color="gray")
        self.whisper_pipe = None
    
    def _browse_models_dir(self):
        """浏览选择模型下载目录"""
        path = filedialog.askdirectory(title="选择模型下载目录")
        if path:
            self.models_dir_var.set(path)
            self.config["models_dir"] = path
            self._save_config()
    
    def _browse_mfa_dir(self):
        """浏览选择 MFA 模型目录"""
        path = filedialog.askdirectory(title="选择 MFA 模型目录")
        if path:
            self.mfa_dir_var.set(path)
            self.config["mfa_dir"] = path
            self._save_config()
            self._scan_mfa_dir()
    
    def _on_mfa_lang_change(self, choice):
        """MFA 语言选择变更"""
        from src.mfa_model_downloader import get_available_languages
        langs = get_available_languages()
        self.mfa_lang_desc.configure(text=langs.get(choice, ""))
    
    def _download_mfa_models(self):
        """下载 MFA 模型"""
        if self._download_thread and self._download_thread.is_alive():
            return
        
        self.mfa_download_btn.configure(state="disabled")
        self.mfa_status.configure(text="⏳ 下载中...", text_color="gray")
        self._download_thread = threading.Thread(target=self._do_download_mfa, daemon=True)
        self._download_thread.start()
    
    def _do_download_mfa(self):
        """执行 MFA 模型下载（后台线程）"""
        from src.mfa_model_downloader import download_language_models
        
        language = self.mfa_lang_var.get()
        output_dir = self.mfa_dir_var.get()
        
        # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.log_callback(f"开始下载 MFA 模型: {language}")
        
        success, acoustic_path, dict_path = download_language_models(
            language=language,
            output_dir=output_dir,
            progress_callback=self.log_callback
        )
        
        if success:
            self.after(0, lambda: self.mfa_status.configure(text="✅ 已下载", text_color="green"))
            self.log_callback(f"声学模型: {acoustic_path}")
            self.log_callback(f"字典文件: {dict_path}")
        else:
            self.after(0, lambda: self.mfa_status.configure(text="❌ 下载失败", text_color="red"))
        
        self.after(0, lambda: self.mfa_download_btn.configure(state="normal"))
        self.after(0, self._scan_mfa_dir)
    
    def _scan_mfa_dir(self):
        """扫描 MFA 模型目录"""
        mfa_dir = self.mfa_dir_var.get()
        
        self.mfa_files_text.configure(state="normal")
        self.mfa_files_text.delete("1.0", "end")
        
        if not os.path.exists(mfa_dir):
            self.mfa_files_text.insert("end", "目录不存在")
        else:
            files = []
            for f in os.listdir(mfa_dir):
                if f.endswith(('.zip', '.dict', '.txt')):
                    fpath = os.path.join(mfa_dir, f)
                    size = os.path.getsize(fpath)
                    size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.0f}KB"
                    files.append(f"• {f} ({size_str})")
            
            if files:
                self.mfa_files_text.insert("end", "\n".join(files))
            else:
                self.mfa_files_text.insert("end", "目录为空，请手动放入 MFA 模型文件")
        
        self.mfa_files_text.configure(state="disabled")
    
    def _download_whisper(self):
        """下载/加载 Whisper 模型"""
        if self._download_thread and self._download_thread.is_alive():
            return
        
        self.whisper_btn.configure(state="disabled")
        self.whisper_status.configure(text="⏳ 加载中...", text_color="gray")
        self._download_thread = threading.Thread(target=self._do_download_whisper, daemon=True)
        self._download_thread.start()
    
    def _do_download_whisper(self):
        """执行 Whisper 模型下载（后台线程）"""
        try:
            self._update_progress("正在加载 transformers 库...")
            from transformers import pipeline
            import torch
            
            model_key = self.whisper_model_var.get()
            model_name = self.WHISPER_MODELS[model_key]["name"]
            cache_dir = os.path.join(self.models_dir_var.get(), "whisper")
            
            # 确保目录存在
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            self._update_progress(f"正在下载/加载 {model_key}...")
            self.log_callback(f"开始加载 Whisper 模型: {model_name}")
            self.log_callback(f"缓存目录: {cache_dir}")
            logger.info(f"加载 Whisper 模型: {model_name}, 缓存目录: {cache_dir}")
            
            # 设置环境变量指定缓存目录
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            
            # 加载模型
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
            logger.info("Whisper 模型加载成功")
            
        except Exception as e:
            error_msg = str(e)
            self._update_progress("")
            self.after(0, lambda: self.whisper_status.configure(text="❌ 加载失败", text_color="red"))
            self.after(0, lambda: self.whisper_btn.configure(state="normal"))
            self.log_callback(f"Whisper 模型加载失败: {error_msg}")
            logger.error(f"Whisper 模型加载失败: {e}", exc_info=True)
    
    def _update_progress(self, text):
        """更新进度文本（线程安全）"""
        self.after(0, lambda: self.progress_label.configure(text=text))
    
    def get_whisper_pipeline(self):
        """获取 Whisper pipeline（供其他模块调用）"""
        return self.whisper_pipe
    
    def get_mfa_dir(self):
        """获取 MFA 模型目录路径（供其他模块调用）"""
        return self.mfa_dir_var.get()


class MakeDatasetFrame(ctk.CTkFrame):
    """批量制作数据集功能面板"""
    
    def __init__(self, master, log_callback):
        super().__init__(master)
        self.log_callback = log_callback
        self._is_running = False
        self._setup_ui()
        self._check_mfa_status()
    
    def _setup_ui(self):
        # MFA 状态提示
        self.mfa_status_label = ctk.CTkLabel(
            self, 
            text="⏳ 检查 MFA 环境...",
            font=ctk.CTkFont(size=12)
        )
        self.mfa_status_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        # 数据集原始目录
        ctk.CTkLabel(self, text="① 切片及LAB目录:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.raw_dir_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.raw_dir_var, width=400).grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_raw_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # 输出目录
        ctk.CTkLabel(self, text="② TextGrid输出目录:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_var = ctk.StringVar()
        ctk.CTkEntry(self, textvariable=self.output_dir_var, width=400).grid(row=2, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_output_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # 字典路径
        ctk.CTkLabel(self, text="③ 字典文件:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.dict_path_var = ctk.StringVar(value="models/mfa/mandarin_china_mfa.dict")
        ctk.CTkEntry(self, textvariable=self.dict_path_var, width=400).grid(row=3, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_dict).grid(row=3, column=2, padx=5, pady=5)
        
        # MFA模型路径
        ctk.CTkLabel(self, text="④ MFA模型文件:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.mfa_model_var = ctk.StringVar(value="models/mfa/mandarin_mfa.zip")
        ctk.CTkEntry(self, textvariable=self.mfa_model_var, width=400).grid(row=4, column=1, padx=5, pady=5)
        ctk.CTkButton(self, text="浏览", width=60, command=self._browse_mfa).grid(row=4, column=2, padx=5, pady=5)
        
        # 选项
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        
        self.single_speaker_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame, 
            text="单说话人模式", 
            variable=self.single_speaker_var
        ).pack(side="left", padx=10)
        
        self.clean_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_frame, 
            text="清理旧缓存", 
            variable=self.clean_var
        ).pack(side="left", padx=10)
        
        # 执行按钮
        self.run_btn = ctk.CTkButton(self, text="⑤ 开始对齐", command=self._run)
        self.run_btn.grid(row=6, column=1, pady=20)
    
    def _check_mfa_status(self):
        """检查 MFA 环境状态"""
        from src.mfa_runner import check_mfa_available
        
        if check_mfa_available():
            self.mfa_status_label.configure(
                text="✅ MFA 外挂环境已就绪 (tools/mfa_engine)",
                text_color="green"
            )
        else:
            self.mfa_status_label.configure(
                text="❌ MFA 外挂环境不可用，请检查 tools/mfa_engine 目录",
                text_color="red"
            )
    
    def _browse_raw_dir(self):
        path = filedialog.askdirectory(title="选择切片及LAB目录")
        if path:
            self.raw_dir_var.set(path)
    
    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="选择TextGrid输出目录")
        if path:
            self.output_dir_var.set(path)
    
    def _browse_dict(self):
        path = filedialog.askopenfilename(
            title="选择字典文件", 
            filetypes=[("字典文件", "*.dict *.txt"), ("所有文件", "*.*")]
        )
        if path:
            self.dict_path_var.set(path)
    
    def _browse_mfa(self):
        path = filedialog.askopenfilename(
            title="选择MFA模型", 
            filetypes=[("ZIP文件", "*.zip"), ("所有文件", "*.*")]
        )
        if path:
            self.mfa_model_var.set(path)
    
    def _run(self):
        if self._is_running:
            return
        
        raw_dir = self.raw_dir_var.get()
        output_dir = self.output_dir_var.get()
        dict_path = self.dict_path_var.get()
        mfa_model = self.mfa_model_var.get()
        
        if not raw_dir or not output_dir:
            messagebox.showerror("错误", "请填写输入目录和输出目录")
            return
        
        self._is_running = True
        self.run_btn.configure(state="disabled", text="对齐中...")
        
        threading.Thread(
            target=self._process,
            args=(raw_dir, output_dir, dict_path, mfa_model),
            daemon=True
        ).start()
    
    def _process(self, raw_dir, output_dir, dict_path, mfa_model):
        """执行 MFA 对齐（后台线程）"""
        from src.mfa_runner import run_mfa_alignment
        
        self.log_callback("=" * 50)
        self.log_callback("开始 MFA 对齐任务")
        
        success, message = run_mfa_alignment(
            corpus_dir=raw_dir,
            output_dir=output_dir,
            dict_path=dict_path if dict_path else None,
            model_path=mfa_model if mfa_model else None,
            single_speaker=self.single_speaker_var.get(),
            clean=self.clean_var.get(),
            progress_callback=self.log_callback
        )
        
        if success:
            self.log_callback("✅ MFA 对齐任务完成!")
            self.log_callback(f"TextGrid 文件已输出到: {output_dir}")
        else:
            self.log_callback(f"❌ MFA 对齐失败: {message}")
        
        self.log_callback("=" * 50)
        
        # 恢复按钮状态
        self.after(0, lambda: self.run_btn.configure(state="normal", text="⑤ 开始对齐"))
        self._is_running = False


class App(ctk.CTk):
    """主应用窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.title("语音数据集处理工具")
        self.geometry("700x600")
        self.minsize(600, 500)
        
        self._setup_ui()
        logger.info("应用启动")
    
    def _setup_ui(self):
        # 标签页
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 添加标签页（按工作流程顺序排列）
        tab1 = self.tabview.add("1. 模型下载")
        tab2 = self.tabview.add("2. 批量制作数据集")
        tab3 = self.tabview.add("3. TextGrid转音频库")
        tab4 = self.tabview.add("4. 音频库排序")
        
        # 各功能面板
        self.download_frame = ModelDownloadFrame(tab1, self._log)
        self.download_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.dataset_frame = MakeDatasetFrame(tab2, self._log)
        self.dataset_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.tg_frame = TextGridToBankFrame(tab3, self._log)
        self.tg_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.sort_frame = BankSortFrame(tab4, self._log)
        self.sort_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 日志区域
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(log_frame, text="日志输出:").pack(anchor="w", padx=5, pady=2)
        
        self.log_text = ctk.CTkTextbox(log_frame, height=150)
        self.log_text.pack(fill="x", padx=5, pady=5)
    
    def _log(self, message):
        """添加日志消息"""
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")


def main():
    """程序入口"""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
