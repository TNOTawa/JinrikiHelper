# 语音数据集处理工具 - 流程文档 （AI用）

## 项目概述

本项目是一个基于 [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) 开发的语音数据集处理工具，用于自动化制作语音音源库。主要功能包括音频切片、语音识别、强制对齐和音源导出。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI 界面 (gui.py)                         │
│                    基于 Gradio 6.2.0 构建                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  模型下载页   │  │  制作音源页   │  │     导出音源页        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    流水线核心 (pipeline.py)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  VoiceBankPipeline - 音源制作流水线                        │   │
│  │  • step0_preprocess() - VAD切片 + Whisper转录             │   │
│  │  • step1_mfa_align()  - MFA语音对齐                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Silero VAD     │  │  Whisper ASR    │  │  MFA 对齐引擎    │
│  语音活动检测    │  │  语音识别       │  │  (外挂模式)      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 目录结构

```
项目根目录/
├── main.py                 # 程序入口 (Web UI)
├── main_local.py           # 本地桌面入口 (CustomTkinter GUI)
├── config.json             # 全局配置文件
├── bank/                   # 音源库目录
│   └── [音源名称]/
│       ├── meta.json       # 音源元信息
│       ├── slices/         # 切片文件 (.wav + .lab)
│       └── textgrid/       # MFA对齐结果 (.TextGrid)
├── export/                 # 导出目录
│   └── [音源名称]/
│       └── simple_export/  # 简单导出结果
├── models/                 # 模型目录
│   ├── mfa/                # MFA声学模型和字典
│   ├── silero_vad/         # Silero VAD模型
│   └── whisper/            # Whisper模型缓存
├── tools/
│   └── mfa_engine/         # MFA独立运行环境 (Sidecar)
└── src/                    # 源代码目录
    ├── gui.py              # Web UI (Gradio 6.2.0)
    ├── gui_old.py          # 旧版桌面 GUI (CustomTkinter)
    ├── pipeline.py         # 流水线核心
    ├── audio_processor.py  # 音频处理
    ├── text_processor.py   # 文本处理
    ├── mfa_runner.py       # MFA对齐运行器
    ├── mfa_model_downloader.py    # MFA模型下载
    ├── silero_vad_downloader.py   # VAD模型下载
    └── export_plugins/     # 导出插件目录
```

## 核心流程

### 完整制作流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           音源制作完整流程                                │
└─────────────────────────────────────────────────────────────────────────┘

输入: 原始音频文件 (.wav/.mp3/.flac/.ogg/.m4a)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤0: 音频预处理 (step0_preprocess)                                     │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 1. Silero VAD 语音活动检测                                           │ │
│ │    • 加载 ONNX 格式 VAD 模型                                         │ │
│ │    • 检测语音片段时间戳                                              │ │
│ │    • 按语音片段切分音频                                              │ │
│ │    • 输出: 16bit 44.1kHz 单声道 WAV                                  │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 2. Whisper 语音识别                                                  │ │
│ │    • 加载 Whisper 模型 (small/medium)                                │ │
│ │    • 对每个切片进行转录                                              │ │
│ │    • 生成对应的 .lab 标注文件                                        │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ 输出: bank/[音源名称]/slices/                                           │
│       ├── test_0000.wav                                                 │
│       ├── test_0000.lab  (中文文本)                                     │
│       ├── test_0001.wav                                                 │
│       └── test_0001.lab                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤1: MFA语音对齐 (step1_mfa_align)                                     │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Montreal Forced Aligner                                              │ │
│ │    • 使用外挂模式调用 tools/mfa_engine                               │ │
│ │    • 输入: .wav 音频 + .lab 中文文本                                 │ │
│ │    • 字典: 汉字 → 音素映射 (mandarin_china_mfa.dict)                 │ │
│ │    • 声学模型: mandarin_mfa.zip                                      │ │
│ │    • 输出: 音素级时间对齐的 TextGrid 文件                            │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ 输出: bank/[音源名称]/textgrid/                                         │
│       ├── test_0000.TextGrid                                            │
│       └── test_0001.TextGrid                                            │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤2: 导出音源 (导出插件系统)                                           │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 简单单字导出插件 (SimpleExportPlugin)                                │ │
│ │    1. 从 TextGrid 提取每个汉字的时间边界                             │ │
│ │    2. 将汉字转换为拼音 (pypinyin)                                    │ │
│ │    3. 按拼音分类切出音频片段                                         │ │
│ │    4. 按时长排序，保留最佳样本                                       │ │
│ │    5. 按命名规则导出 (如: ba.wav, ba1.wav, ba2.wav)                  │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ UTAU oto.ini 导出插件 (UTAUOtoExportPlugin)                          │ │
│ │    1. 从 TextGrid phones 层提取音素时间边界                          │ │
│ │    2. 识别辅音+元音对，计算 oto.ini 六参数                           │ │
│ │       • Offset: 音频开始位置                                         │ │
│ │       • Consonant: 不被拉伸的区域                                    │ │
│ │       • Cutoff: 音频结束位置（负值从末尾算）                         │ │
│ │       • Preutterance: 与节拍对齐位置                                 │ │
│ │       • Overlap: 交叉淡化区域                                        │ │
│ │    3. IPA 音素转换为拼音/罗马音别名                                  │ │
│ │    4. 生成 oto.ini 配置文件                                          │ │
│ │    5. 生成 character.txt（支持自定义角色名）                         │ │
│ │    6. 自动检测文件名编码兼容性，不合法时转拼音                       │ │
│ │    7. 自动拼字功能（可选）:                                          │ │
│ │       • 收集已有的高质量辅音和元音片段                               │ │
│ │       • 排列组合生成缺失的音素组合                                   │ │
│ │       • 交叉淡化拼接音频并保存                                       │ │
│ │       • 自动生成对应的 oto 配置条目                                  │ │
│ │    8. 模糊拼字功能（可选，仅中文）:                                  │ │
│ │       • 在自动拼字基础上，用近似音素替代缺失音素                     │ │
│ │       • 声母近似组: sh↔s, zh↔z, ch↔c, l↔n↔r, f↔h                    │ │
│ │       • 韵母近似组: an↔ang, en↔eng↔ong, in↔ing, ian↔iang, uan↔uang │ │
│ │       • 同组内音素互为替代，按组内顺序优先匹配                       │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ 输出: export/[音源名称]/simple_export/                                  │
│       ├── ba.wav                                                        │
│       ├── ba1.wav                                                       │
│       ├── ni.wav                                                        │
│       └── ...                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. 模型管理模块

| 模块 | 文件 | 功能 |
|------|------|------|
| Silero VAD 下载 | `silero_vad_downloader.py` | 下载语音活动检测模型（支持多镜像源） |
| MFA 模型下载 | `mfa_model_downloader.py` | 下载声学模型和字典（带完整性校验） |
| Whisper 模型 | 通过 HuggingFace 自动下载 | 语音识别模型 |

支持的语言:
- 中文 (普通话): `mandarin_mfa.zip` + `mandarin_china_mfa.dict`
- 日文: `japanese_mfa.zip` + `japanese_mfa.dict`

Silero VAD 下载镜像源（按优先级）:
- HuggingFace deepghs/silero-vad-onnx（国内云环境推荐）
- HuggingFace onnx-community/silero-vad
- hf-mirror.com 镜像站
- ghproxy.com 镜像（GitHub 加速）
- jsdelivr CDN
- GitHub 原始地址（备选）

MFA 字典文件完整性校验:
- 下载完成后计算 SHA256 哈希并保存为 `.sha256` 文件
- 后续启动时校验哈希值，损坏则自动重新下载
- 检查字典文件最少行数（中文 8 万行，日文 10 万行）
- 自动清理字典文件中的空行（MFA 3.x 不支持空行）

### 2. 音频处理模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 音频处理器 | `audio_processor.py` | VAD切片 + Whisper转录封装 |
| 文本处理器 | `text_processor.py` | 中文转拼音、日文转罗马音 |

### 3. MFA 对齐模块

| 模块 | 文件 | 功能 |
|------|------|------|
| MFA 运行器 | `mfa_runner.py` | 跨平台调用 MFA 引擎 |

MFA 支持两种运行模式:
- **Windows**: Sidecar Pattern，通过 subprocess 调用独立的 Python 环境 (`tools/mfa_engine`)
- **Linux**: 直接调用系统安装的 `mfa` 命令 (pip install montreal-forced-aligner)

### 4. 导出插件系统

| 模块 | 文件 | 功能 |
|------|------|------|
| 插件基类 | `export_plugins/base.py` | 定义插件接口、配置选项和公共方法 |
| 插件加载器 | `export_plugins/loader.py` | 扫描和加载插件 |
| 简单导出 | `export_plugins/simple_export.py` | 按拼音分类导出单字音频，支持质量评估 |
| UTAU 导出 | `export_plugins/utau_oto_export.py` | 生成 UTAU 音源配置文件 (oto.ini) |
| 质量评分 | `quality_scorer.py` | 音频质量多维度评估 |

插件配置选项类型:
- `TEXT`: 文本输入
- `NUMBER`: 数字输入
- `SWITCH`: 开关
- `COMBO`: 下拉选择
- `MULTI_SELECT`: 多选框
- `FILE`/`FOLDER`: 文件/文件夹选择

基类公共方法 (`ExportPlugin`):
- `load_language_from_meta()`: 从 meta.json 加载语言设置
- `parse_quality_metrics()`: 解析质量评估维度选项
- `apply_naming_rule()`: 应用命名规则生成文件名/别名
- `get_source_paths()`: 获取音源相关路径
- `get_export_dir()`: 获取导出目录路径
- `get_quality_scorer()`: 获取质量评分器实例

### 5. 音源质量评分模块

`src/quality_scorer.py` 提供多维度音频质量评估:

| 评估维度 | 函数 | 说明 | 耗时 |
|---------|------|------|------|
| 时长 | `duration_score()` | 适中时长得分高 (0.3~0.8s 最佳) | <1ms |
| 音量稳定性 | `rms_variance_score()` | RMS 方差越小越好 | ~5ms |
| 音高稳定性 | `f0_variance_score()` | F0 方差越小越好 | ~50-200ms |

使用方式:
```python
from src.quality_scorer import QualityScorer

scorer = QualityScorer(enabled_metrics=["duration", "f0"])
scores = scorer.score_from_file("audio.wav")
# 返回: {"duration": 0.85, "f0": 0.91, "combined": 0.88}
```

导出插件质量评估选项:
- `duration`: 仅时长评估（默认，最快）
- `duration+rms`: 时长 + 音量稳定性
- `duration+f0`: 时长 + 音高稳定性
- `all`: 全部维度（耗时较长）

已集成质量评估的插件:
- **简单单字导出**: 默认仅评估时长，可选启用 RMS/F0 评估
- **UTAU oto.ini 导出**: 默认评估时长+RMS，可选启用 F0 评估

### 5. MFA 跨平台支持

MFA 支持三种运行模式:
- **Windows (外挂模式)**: 通过 subprocess 调用独立的 Python 环境 (`tools/mfa_engine`)
- **Linux (系统命令)**: 直接调用系统 PATH 中的 `mfa` 命令
- **Linux (Python 模块)**: 通过 `python -m montreal_forced_aligner` 调用

`mfa_runner.py` 会自动检测可用的调用方式，优先级:
1. 系统 `mfa` 命令 (shutil.which)
2. Python 模块方式 (`sys.executable -m montreal_forced_aligner`)
3. Windows 外挂模式 (`tools/mfa_engine/python.exe`)

## 数据流

```
原始音频
    │
    ├──[VAD切片]──→ slices/*.wav (44.1kHz 16bit 单声道)
    │
    ├──[Whisper]──→ slices/*.lab (中文文本)
    │
    ├──[MFA对齐]──→ textgrid/*.TextGrid (音素级时间标注)
    │
    └──[导出插件]──→ export/[音源]/[插件名]/*.wav (按拼音分类)
```

## 元信息文件

每个音源目录包含 `meta.json` 文件，记录制作参数:

```json
{
  "source_name": "my_voice",
  "created_at": "2026-01-31T10:00:00",
  "updated_at": "2026-01-31T12:00:00",
  "whisper_model": "openai/whisper-small",
  "mfa_dict": "mandarin_china_mfa.dict",
  "mfa_acoustic": "mandarin_mfa.zip",
  "language": "chinese",
  "single_speaker": true,
  "slice_count": 13,
  "textgrid_count": 13
}
```

## 使用流程

### 方式一: 本地 Web UI (Gradio)

1. 运行 `python main.py` 启动 Web UI
2. 浏览器自动打开 http://127.0.0.1:7860
3. 在「📦 模型下载」页下载所需模型:
   - Whisper 语音识别模型 (small/medium)
   - Silero VAD 语音活动检测模型
   - MFA 声学模型和字典 (中文/日文)
4. 在「🎵 制作音源」页:
   - 设置音源名称和转录语言
   - 上传或输入音频文件路径
   - 选择 Whisper 模型和 MFA 模型
   - 点击「一键执行全部流程」或分步执行
5. 在「📤 导出音源」页:
   - 选择已制作的音源
   - 选择导出插件
   - 配置导出选项并执行
   - 点击下载按钮获取结果

> 注: 旧版 CustomTkinter 桌面 GUI 已移至 `src/gui_old.py`，可通过 `python main_local.py` 启动

### 方式二: 本地桌面 GUI (CustomTkinter)

1. 运行 `python main_local.py` 启动桌面应用
2. 使用原生窗口界面操作，功能与 Web UI 相同
3. 适合不需要浏览器的本地独立运行场景

### 方式三: 云端部署 (HF Spaces / 魔塔社区)

1. 使用 `app.py` 作为入口文件
2. 云端环境自动安装 MFA 和下载模型
3. 处理完成后通过下载按钮获取结果 (云端数据不持久)

支持的云平台:
- Hugging Face Spaces (Gradio SDK)
- 魔塔社区 ModelScope (推荐，国内访问快)

### 方式四: 命令行/脚本

```python
from src.pipeline import PipelineConfig, VoiceBankPipeline

config = PipelineConfig(
    source_name="my_voice",
    input_path="path/to/audio.wav",
    output_base_dir="bank",
    models_dir="models",
    whisper_model="openai/whisper-small",
    mfa_dict_path="models/mfa/mandarin_china_mfa.dict",
    mfa_model_path="models/mfa/mandarin_mfa.zip",
    language="chinese"
)

pipeline = VoiceBankPipeline(config, progress_callback=print)
success, msg = pipeline.run_make_pipeline()
```

## 依赖说明

核心依赖:
- `gradio`: Web UI 框架 (6.2.0)
- `transformers`: Whisper 模型
- `torch` + `torchaudio` + `torchvision`: PyTorch 生态（版本需匹配）
- `silero-vad`: 语音活动检测
- `textgrid`: TextGrid 文件解析
- `soundfile`: 音频读写
- `pypinyin`: 中文转拼音
- `pykakasi`: 日文转罗马音 (可选)

> 注: 旧版桌面 GUI 使用 `customtkinter`，代码保留在 `src/gui_old.py`

> 注: `torchvision` 必须与 `torch` 版本匹配，否则会导致 transformers 导入失败

MFA 环境:
- **Windows**: 独立打包在 `tools/mfa_engine/`，包含 Python 3.11 + montreal-forced-aligner
- **Linux**: 通过 pip 安装 `montreal-forced-aligner`

## 云端部署说明

### 目录结构 (云端)

```
项目根目录/
├── app.py                  # 云端入口 (使用 gui_cloud.py)
├── main.py                 # 本地入口 (使用 gui.py)
├── requirements.txt
├── src/
│   ├── gui.py              # 本地 Web UI (完整功能)
│   ├── gui_cloud.py        # 云端 Web UI (上传→处理→下载)
│   ├── gui_old.py          # 旧版桌面 GUI (CustomTkinter)
│   ├── mfa_runner.py       # 跨平台 MFA 调用
│   └── ...
└── ...
```

### 云端 GUI 特点 (gui_cloud.py)

- **制作音源**: 上传音频文件 → VAD切片 + Whisper转录 + MFA对齐 → 下载音源包
- **导出音源**: 上传音源包 → 选择导出插件 → 下载导出结果
- 使用临时工作空间，处理完成后自动清理
- 无需本地持久化存储

**用户体验优化**:
- 音频未上传完成时禁用「开始制作」按钮，防止误操作
- 导出页面提供「使用刚制作的音源」按钮，避免重复上传
- Whisper 模型选项标注速度参考：small 约 4 秒/句，medium 约 12 秒/句（慢 2-3 倍但更准确）
- **导出插件动态选项系统**: 
  - 插件选项完全动态化，根据 `ExportPlugin.get_options()` 自动生成 UI 组件
  - 切换插件时自动显示/隐藏对应的配置选项组
  - 支持的选项类型: TEXT(文本)、NUMBER(数字)、SWITCH(开关)、COMBO(下拉)、MULTI_SELECT(多选)、LABEL(说明文字)
  - 新增插件无需修改 GUI 代码，只需在插件中定义 `get_options()` 即可自动生成界面

### 平台差异

| 功能 | 本地 (Windows) | 云端 (Linux) |
|------|----------------|--------------|
| MFA 调用 | tools/mfa_engine 外挂 | 系统 mfa 命令 |
| 数据存储 | 本地持久化 | 临时目录，需下载 |
| GPU 加速 | 本地显卡 | 取决于平台配置 |
| 模型缓存 | models/ 目录 | 首次运行自动下载 |

### 魔搭创空间部署配置

部署配置文件 `ms_deploy.json`:
```json
{
  "sdk_type": "gradio",
  "sdk_version": "6.2.0",
  "resource_configuration": "platform/2v-cpu-16g-mem",
  "base_image": "ubuntu22.04-py311-torch2.3.1-modelscope1.31.0"
}
```

云端依赖文件 `requirements_cloud.txt`:
- 移除 Windows 专用依赖 (customtkinter)
- 添加 `montreal-forced-aligner` (Linux pip 安装)
- 保持 gradio 版本与 sdk_version 一致

### 云端 MFA 中文分词依赖

MFA 中文对齐需要 `spacy-pkuseg` 分词器，该分词器会从 GitHub 下载模型文件。由于魔搭创空间无法直接访问 GitHub，`app.py` 会在启动时预下载模型：

**pkuseg 模型文件**:
- `spacy_ontonotes.zip` - 中文分词模型 (约 30MB)
- `postag.zip` - 词性标注模型 (约 50MB)

**下载策略**:
1. 使用 GitHub 镜像 `ghfast.top` 加速下载
2. 模型保存到持久化目录 `/home/studio_service/models/pkuseg/`
3. 必须保留 zip 文件（pkuseg 使用 `torch.hub.download_url_to_file` 检查 zip 是否存在）
4. 设置 `PKUSEG_HOME` 环境变量指向模型目录

**环境变量传递**:
- `app.py` 启动时设置 `os.environ["PKUSEG_HOME"]`
- `mfa_runner.py` 的 `_build_mfa_env()` 函数在 Linux 环境下也设置该变量
- 确保 MFA 子进程能正确找到预下载的模型
