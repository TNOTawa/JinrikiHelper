# MFA 引擎本地安装说明

本文档介绍如何在本地部署 MFA (Montreal Forced Aligner) 引擎，适用于想从源码运行项目的用户。

## 前提条件

- Windows 系统
- 已安装 conda 或 micromamba

## 安装步骤

### 1. 创建 MFA 环境

使用 conda：
```bash
conda create -n mfa_engine -c conda-forge montreal-forced-aligner
```

或使用 micromamba （推荐）：
```bash
micromamba create -n mfa_engine -c conda-forge montreal-forced-aligner
```

### 2. 提取环境到项目目录

安装完成后，将环境目录复制到项目的 `tools` 文件夹中：

conda 默认路径：
```
%USERPROFILE%\anaconda3\envs\mfa_engine
或
%USERPROFILE%\miniconda3\envs\mfa_engine
```

micromamba 默认路径：
```
%USERPROFILE%\micromamba\envs\mfa_engine
```

将整个 `mfa_engine` 文件夹复制到项目的 `tools/` 目录下，最终结构：

```
项目根目录/
└── tools/
    └── mfa_engine/
        ├── python.exe
        ├── Scripts/
        │   └── mfa.exe
        └── ...
```

### 3. 验证安装

运行以下命令验证 MFA 是否可用：

```bash
tools\mfa_engine\Scripts\mfa.exe version
```

如果正确输出版本号，说明安装成功。

## 常见问题

安装过程中如遇到问题或报错，建议将错误信息提供给 AI 助手寻求帮助。
