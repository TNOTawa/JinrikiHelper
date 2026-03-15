# FFmpeg 环境变量问题排查指南

## 问题现象

用户已安装 FFmpeg 并设置了系统环境变量，但运行便携版时仍报错：

```
FileNotFoundError: [WinError 2] 系统找不到指定的档案。
```

错误发生在 `pipeline.py` 调用 FFmpeg 进行音频处理时。

## 原因分析

### 便携版环境隔离

便携版通过 `run_portable.bat` 启动，使用内嵌的 `python\python.exe`。该 Python 环境可能存在以下问题：

1. **PATH 环境变量未正确继承** - 便携版 Python 可能无法访问系统 PATH 中的 FFmpeg
2. **CMD 窗口环境变量刷新问题** - 新设置的环境变量需要重启 CMD 窗口才能生效

## 解决方案

### 方案一：重启命令提示符（推荐先尝试）

如果刚刚设置完 FFmpeg 环境变量，需要：

1. **关闭所有 CMD 窗口**
2. **重新打开 CMD 窗口**
3. 再次运行 `run_portable.bat`

> 环境变量修改后，已打开的 CMD 窗口不会自动刷新，必须重新打开。

### 方案二：验证 FFmpeg 是否正确安装

在 CMD 中执行以下命令验证：

```cmd
where ffmpeg
ffmpeg -version
```

如果显示 "找不到文件" 或报错，说明环境变量设置有问题。

### 方案三：检查环境变量设置

1. 按 `Win + R`，输入 `sysdm.cpl`，回车
2. 点击「高级」→「环境变量」
3. 在「系统变量」或「用户变量」中找到 `Path`
4. 确认 FFmpeg 的 `bin` 目录已添加，例如：
   ```
   C:\ffmpeg\bin
   ```
5. 点击确定保存，然后**重新打开 CMD 窗口**

### 方案四：在便携版脚本中显式指定 FFmpeg 路径

如果上述方案无效，可以修改 `run_portable.bat`，在启动前手动添加 FFmpeg 路径：

```bat
@echo off
chcp 65001 >nul
echo 启动人力V助手 (便携版)...

REM 添加 FFmpeg 到 PATH（请修改为你的实际路径）
set PATH=%PATH%;C:\ffmpeg\bin

set PYTHONPATH=%~dp0
"%~dp0python\python.exe" "%~dp0main.py"
pause
```

将 `C:\ffmpeg\bin` 替换为你的 FFmpeg 实际安装路径。

### 方案五：将 FFmpeg 放入便携版目录

将 `ffmpeg.exe` 和 `ffprobe.exe` 直接复制到便携版根目录（与 `main.py` 同级），程序会优先使用当前目录下的可执行文件。

## 快速诊断命令

在 `run_portable.bat` 所在目录打开 CMD，执行：

```cmd
REM 检查系统 FFmpeg
where ffmpeg

REM 检查便携版 Python 能否找到 FFmpeg
python\python.exe -c "import subprocess; subprocess.run(['ffmpeg', '-version'])"
```

如果第一条命令成功但第二条失败，说明便携版 Python 环境与系统环境隔离，请使用方案四或方案五。

## 相关文件

- `run_portable.bat` - 便携版启动脚本
- `src/pipeline.py` - 音频处理流水线，调用 FFmpeg 的位置
