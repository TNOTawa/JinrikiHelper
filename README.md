# 项目名称

> 基于 [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) 开发的语音数据集处理工具

## 简介

本项目用于处理语音数据集，支持以下功能：

- TextGrid 标注文件转音频库
- 音频文件排序与筛选
- 批量制作 UTAU 人力音源

## 环境要求

- Python 3.8+
- MFA（Montreal Forced Aligner）

## 安装

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 依赖项

主要依赖：

- textgrid
- audiofile
- tqdm

## 使用方法

### TextGrid 转音频库

```bash
python textgrid2bank.py
```

### 音频库排序

```bash
python bank_sort.py
```

### 批量制作数据集

```bash
python make_dataset_batch.py
```

## 项目结构

```
TODO
```

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。

原项目版权归 心水湛清 所有。

## 致谢

- [xszqxszq/auto-voice-bank](https://github.com/xszqxszq/auto-voice-bank) - 原项目作者
