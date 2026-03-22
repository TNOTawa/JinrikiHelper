# 并发队列配置指南

## 概述
云端部署支持 **4 个独立的任务队列**，分别控制不同模块的并发数。针对 **2核 CPU、16GB 内存** 的配置，提供了合理的默认值。

## 任务队列架构

| 队列名称 | 关键词 | 资源消耗 | 默认值 | 环境变量 | 说明 |
|---------|-------|--------|-------|---------|------|
| **制作** | `make` | 6-10GB | 1 | `JINRIKI_MAX_MAKE_JOBS` | VAD切片 → Whisper转录 → MFA对齐 → 打包 |
| **识别** | `whisper` | 3-5GB | 0(不限) | `JINRIKI_MAX_WHISPER_JOBS` | 单独的语音识别任务（可选） |
| **对齐** | `mfa` | 2-4GB | 1 | `JINRIKI_MAX_MFA_JOBS` | 单独的 MFA 对齐任务 |
| **导出** | `export` | 1-2GB | 2 | `JINRIKI_MAX_EXPORT_JOBS` | 音源导出（I/O密集） |

## 默认配置（2核 CPU）

```bash
# 推荐配置（平衡吞吐和稳定性）
export JINRIKI_MAX_MAKE_JOBS=1        # 综合任务：1个
export JINRIKI_MAX_WHISPER_JOBS=0     # 识别任务：不限制（仅供高级用户）
export JINRIKI_MAX_MFA_JOBS=1         # 对齐任务：1个
export JINRIKI_MAX_EXPORT_JOBS=2      # 导出任务：2个（I/O密集，可以并发）
export JINRIKI_MAX_JOB_SECONDS=1800   # 单任务超时：30分钟
```

### 为什么是这样的配置？

| 队列 | 资源考量 | 建议 |
|------|--------|------|
| `make` | 包含 Whisper + MFA，CPU+GPU 密集 | **限制为 1**：避免内存溢出，16GB 无法同时运行 2 个 |
| `whisper` | 单独的语音识别，仅当不用 `make` 时 | **默认不限制(0)**：非主流程，不配置 |
| `mfa` | MFA 自动利用 2 核 CPU（`num_jobs=2`） | **限制为 1**：避免与 `make` 中的 MFA 冲突 |
| `export` | I/O 密集，内存占用少（仅 1-2GB） | **限制为 2**：可与其他任务并行 |

## 调整方案

### 方案 A：重视吞吐（高风险）
```bash
export JINRIKI_MAX_MAKE_JOBS=2        # 可能导致内存溢出
export JINRIKI_MAX_MFA_JOBS=1
export JINRIKI_MAX_EXPORT_JOBS=3
```
⚠️ **风险**：需要严格的内存监控，推荐仅在 16GB 内存充足且无其他应用时使用

### 方案 B：稳定性优先（低风险）
```bash
export JINRIKI_MAX_MAKE_JOBS=1        # 同上
export JINRIKI_MAX_MFA_JOBS=1
export JINRIKI_MAX_EXPORT_JOBS=1      # 所有队列均串行
```
✓ **优势**：最稳定，适合生产环境

### 方案 C：仅导出加速
```bash
export JINRIKI_MAX_MAKE_JOBS=1
export JINRIKI_MAX_MFA_JOBS=1
export JINRIKI_MAX_EXPORT_JOBS=4      # 导出可以更多并发
```
✓ **用场景**：已制作的音源包仅需反复导出不同格式

## 部署方式

### 1. Modelscope 魔塔创空间

#### 方式A：环境变量（推荐）
在创空间的「**启动命令**」中设置：

```bash
export JINRIKI_MAX_MAKE_JOBS=1 && \
export JINRIKI_MAX_MFA_JOBS=1 && \
export JINRIKI_MAX_EXPORT_JOBS=2 && \
python app.py
```

#### 方式B：配置文件
创建 `.env` 文件：
```
JINRIKI_MAX_MAKE_JOBS=1
JINRIKI_MAX_MFA_JOBS=1
JINRIKI_MAX_EXPORT_JOBS=2
JINRIKI_MAX_JOB_SECONDS=1800
```

### 2. HuggingFace Spaces
同样支持在 Space 的 **Secrets** 中设置环境变量

### 3. 本地开发
```bash
# Windows PowerShell
$env:JINRIKI_MAX_MAKE_JOBS=1
$env:JINRIKI_MAX_EXPORT_JOBS=2
python app.py

# Linux/macOS
export JINRIKI_MAX_MAKE_JOBS=1
export JINRIKI_MAX_EXPORT_JOBS=2
python app.py
```

## 队列状态监控

在 Web UI 的「**关于**」页面可以实时查看队列状态：

```
【任务队列】 制作: 0/1 | 识别: 0/∞ | 对齐: 1/1 | 导出: 2/2
【运行中】 make:abc123 | export:def456 | export:ghi789 ...共3个
```

- `∞` 表示识别队列不限制（默认配置）
- 当队列满时，新任务会被拒绝并提示"服务繁忙"

## 常见问题

### Q1: 如何同时支持更多并发？
**A**: 需要更高的硬件配置。目前针对 2核 CPU 的推荐是单 `make` 任务 + 多 `export` 任务。升级至 4核+可考虑：
```bash
export JINRIKI_MAX_MAKE_JOBS=2
export JINRIKI_MAX_EXPORT_JOBS=3
```

### Q2: 为什么 `JINRIKI_MAX_WHISPER_JOBS=0` 不限制？
**A**: `whisper` 队列仅用于 `make` 流程外的单独转录任务，这是高级功能。默认不限制以免阻碍非主流程。生产场景建议改为：
```bash
export JINRIKI_MAX_WHISPER_JOBS=1
```

### Q3: 任务超时后会怎样？
**A**: 超时的任务会被标记为取消，释放队列位置，用户会收到"已超时"的提示。默认 1800 秒（30 分钟）。

### Q4: 能否设置某个队列为 0（禁用）？
**A**: 不推荐。系统最少保持 `min=1` 的限制以避免完全阻塞。若确实要禁用某个功能，应改代码而非配置文件。

## 监控与调优

### 查看当前内存占用
```bash
# Linux
free -h
# macOS
vm_stat

# Windows PowerShell
Get-WmiObject Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum
```

### 推荐的监控指标
- **内存使用率**: < 80%（预留余量防止 OOM）
- **任务排队时间**: < 300 秒（5 分钟）
- **任务成功率**: ≥ 99%

### 根据状态调整

| 现象 | 原因 | 调整 |
|------|------|------|
| 内存持续 > 90% | 并发过高 | 降低 `MAX_MAKE_JOBS` |
| 任务频繁超时 | CPU/GPU 资源不足 | 降低 `MAX_EXPORT_JOBS` |
| 队列堆积 > 10 | 硬件不足以支撑 | 等待配置升级或降低并发 |

## 参考资源
- MFA 对齐并发：支持自动检测 `num_jobs` (2核 = 2 并发)
- Whisper 转录：单线程（不支持内部并发）
- 音源导出：纯 I/O 密集，可使用线程池（当前实现为单线程）

---

**最后更新**: 2026.03.22 | **默认配置**: 2核 CPU + 16GB 内存 + Modelscope 魔塔
