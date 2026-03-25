| 模块 | 文件 | 功能 |
|------|------|------|
| MFA 运行器 | `mfa_runner.py` | 跨平台调用 MFA 引擎 |

MFA 支持两种运行模式:
- **Windows**: Sidecar Pattern，通过 subprocess 调用独立的 Python 环境 (`tools/mfa_engine`)
    （本人使用micromamba安装，并复制 micromamba\envs\mfa_engine 到 tools 文件夹。）
- **Linux**: 直接调用系统安装的 `mfa` 命令（官方推荐 conda-forge 安装: `conda install -c conda-forge montreal-forced-aligner`）

**Windows 中文用户名兼容性**:
MFA 底层使用的 OpenFST 库不支持非 ASCII 路径。当 Windows 用户名包含中文时，MFA 默认数据目录 `C:\Users\用户名\Documents\MFA\` 会导致写入失败。解决方案是通过 `MFA_ROOT_DIR` 环境变量将数据目录重定向到项目目录下的 `mfa_data/` 文件夹（纯 ASCII 路径）。