# 语音数据集处理工具 - 流程文档 （AI用）

## 项目概述

本项目是一个基于 [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) 开发的语音数据集处理工具，用于自动化制作语音音源库。主要功能包括音频切片、语音识别、强制对齐和音源导出。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI 界面 (gui.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  模型配置页   │  │  制作音源页   │  │     导出音源页        │   │
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
├── main.py                 # 程序入口
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
| Silero VAD 下载 | `silero_vad_downloader.py` | 下载语音活动检测模型 |
| MFA 模型下载 | `mfa_model_downloader.py` | 下载声学模型和字典 |
| Whisper 模型 | 通过 HuggingFace 自动下载 | 语音识别模型 |

支持的语言:
- 中文 (普通话): `mandarin_mfa.zip` + `mandarin_china_mfa.dict`
- 日文: `japanese_mfa.zip` + `japanese_mfa.dict`

### 2. 音频处理模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 音频处理器 | `audio_processor.py` | VAD切片 + Whisper转录封装 |
| 文本处理器 | `text_processor.py` | 中文转拼音、日文转罗马音 |

### 3. MFA 对齐模块

| 模块 | 文件 | 功能 |
|------|------|------|
| MFA 运行器 | `mfa_runner.py` | 外挂模式调用 MFA 引擎 |

MFA 采用 Sidecar Pattern，通过 subprocess 调用独立的 Python 环境 (`tools/mfa_engine`)，避免依赖冲突。

### 4. 导出插件系统

| 模块 | 文件 | 功能 |
|------|------|------|
| 插件基类 | `export_plugins/base.py` | 定义插件接口和配置选项 |
| 插件加载器 | `export_plugins/loader.py` | 扫描和加载插件 |
| 简单导出 | `export_plugins/simple_export.py` | 按拼音分类导出单字音频 |

插件配置选项类型:
- `TEXT`: 文本输入
- `NUMBER`: 数字输入
- `SWITCH`: 开关
- `COMBO`: 下拉选择
- `FILE`/`FOLDER`: 文件/文件夹选择

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

### 方式一: GUI 界面

1. 运行 `python main.py` 启动 GUI
2. 在「模型配置」页下载所需模型
3. 在「制作音源」页:
   - 设置音源名称
   - 选择输入音频
   - 选择模型和语言
   - 点击「一键执行全部流程」或分步执行
4. 在「导出音源」页选择导出插件并执行

### 方式二: 命令行/脚本

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
- `customtkinter`: GUI 框架
- `transformers`: Whisper 模型
- `silero-vad`: 语音活动检测
- `textgrid`: TextGrid 文件解析
- `soundfile`: 音频读写
- `torchaudio`: 音频重采样
- `pypinyin`: 中文转拼音
- `pykakasi`: 日文转罗马音 (可选)

MFA 环境:
- 独立打包在 `tools/mfa_engine/`
- 包含 Python 3.11 + montreal-forced-aligner
