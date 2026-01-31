# 人力V助手 (JinrikiHelper)

> 基于 [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) 开发的语音数据集处理工具

作者：TNOT

## 简介

本项目是一个自动化语音音源库制作工具，主要功能包括：

- 音频切片：使用 Silero VAD 进行语音活动检测和自动切片
- 语音识别：使用 Whisper 模型进行语音转文字
- 强制对齐：使用 MFA (Montreal Forced Aligner) 进行音素级时间对齐
- 音源导出：支持插件化导出，按拼音分类导出单字音频

支持语言：中文（普通话）、日文

## 环境要求

- Python 3.8+
- Windows 系统（MFA 引擎已预打包）

## 安装

### 1. 安装 Python 依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载 MFA 引擎

MFA 引擎体积较大，未包含在仓库中，请从以下地址下载：

| 网盘 | 链接 |
|------|------|
| 百度网盘 | [待补充] |
| Google Drive | [待补充] |

下载后解压到项目根目录的 `tools/` 文件夹下，确保目录结构如下：

```
tools/
└── mfa_engine/
    ├── python.exe
    ├── Scripts/
    │   └── mfa.exe
    └── ...
```

## 使用方法

### GUI 界面（推荐）

```bash
python main.py
```

1. 在「模型配置」页下载所需模型（Silero VAD、MFA 声学模型和字典）
2. 在「制作音源」页：
   - 设置音源名称
   - 选择输入音频文件（支持 .wav/.mp3/.flac/.ogg/.m4a）
   - 选择 Whisper 模型和语言
   - 点击「一键执行全部流程」或分步执行
3. 在「导出音源」页选择导出插件并执行

### 脚本调用

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

## 项目结构

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
│       └── [插件名]/       # 导出结果
├── models/                 # 模型目录
│   ├── mfa/                # MFA声学模型和字典
│   ├── silero_vad/         # Silero VAD模型
│   └── whisper/            # Whisper模型缓存
├── tools/
│   └── mfa_engine/         # MFA独立运行环境
├── src/                    # 源代码目录
│   ├── gui.py              # GUI界面
│   ├── pipeline.py         # 流水线核心
│   ├── audio_processor.py  # 音频处理（VAD+Whisper）
│   ├── mfa_runner.py       # MFA运行器
│   ├── text_processor.py   # 文本处理（拼音转换）
│   └── export_plugins/     # 导出插件目录
├── docs/                   # 项目文档
└── tests/                  # 测试代码
```

## 依赖项

核心依赖：
- `customtkinter` - GUI 框架
- `transformers` - Whisper 模型
- `onnxruntime` - Silero VAD 推理
- `textgrid` - TextGrid 文件解析
- `soundfile` - 音频读写
- `torchaudio` - 音频重采样
- `pypinyin` - 中文转拼音
- `pykakasi` - 日文转罗马音

MFA 环境需单独下载，详见安装说明。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

本工具集成 Montreal Forced Aligner (MIT License)。

原项目版权归 心水湛清 所有。

## 致谢

- [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) - 原项目作者
- [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) - 语音对齐工具
- [Silero VAD](https://github.com/snakers4/silero-vad) - 语音活动检测
- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别模型
