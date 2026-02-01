# UTAU oto.ini 导出功能设计文档

## oto.ini 格式规范

### 文件格式
- 纯文本文件，编码通常为 Shift-JIS（日语）或 UTF-8
- 每行一条配置，格式为：
```
文件名.wav=别名,Offset,Consonant,Cutoff,Preutterance,Overlap
```

### 六个参数说明

| 参数 | 中文名 | 单位 | 说明 |
|-----|-------|-----|------|
| 文件名 | - | - | wav 音频文件名 |
| Alias | 别名 | - | 音素/音节名称，UTAU 调用时使用 |
| Offset | 左空白 | ms | 音频开始播放的位置，跳过前面的静音 |
| Consonant | 辅音区 | ms | 不被拉伸的区域长度（从 Offset 开始计算） |
| Cutoff | 右空白 | ms | 音频停止播放的位置（正值从开头算，负值从末尾算） |
| Preutterance | 先行发声 | ms | 与节拍对齐的位置（通常是辅音结束/元音开始处） |
| Overlap | 重叠 | ms | 与前一音符交叉淡化的区域长度 |

### 参数可视化
```
|<-- Offset -->|<-- Consonant -->|<-- 可拉伸区域 -->|<-- Cutoff(负值) -->|
[    蓝色      ][      粉色      ][      白色       ][       蓝色        ]
               ^                 ^
               |                 |
            Overlap         Preutterance
```

### 示例
```
あ.wav=あ,100,150,-200,120,30
ka.wav=か,50,80,-150,70,20
```

## 从 TextGrid 生成 oto.ini 的可行性分析

### TextGrid 数据结构
项目的 TextGrid 包含两个层级：
- **words 层**：词/字级别的时间标注
- **phones 层**：音素级别的时间标注（IPA 符号）

### 中文示例
```
# words 层：「你」0.03s - 0.14s
# phones 层：
intervals [2]: xmin = 0.03, xmax = 0.08, text = "ɲ"      # 辅音 50ms
intervals [3]: xmin = 0.08, xmax = 0.14, text = "i˨˩˦"   # 元音 60ms
```

### 日语示例
```
# words 层：「地」0.9s - 1.04s
# phones 层：
intervals [9]:  xmin = 0.9,  xmax = 0.98, text = "dʑ"    # 辅音 80ms
intervals [10]: xmin = 0.98, xmax = 1.04, text = "i"     # 元音 60ms
```

### 参数计算方法

| oto.ini 参数 | 计算方式 |
|-------------|---------|
| Offset | `辅音.xmin * 1000`（转毫秒） |
| Consonant | `(辅音.xmax - 辅音.xmin) * 1000` |
| Cutoff | `-(文件时长 - 元音.xmax) * 1000`（负值，从末尾算） |
| Preutterance | `(辅音.xmax - 辅音.xmin) * 1000`（辅音时长，相对于 Offset） |
| Overlap | `Preutterance * 0.25 ~ 0.33`（经验值，取 1/4 到 1/3） |

### 结论
**TextGrid 数据完全足够生成 oto.ini**，phones 层已提供辅音/元音的精确时间边界。

## 实现要点

### 1. 音素类型识别
需要建立 IPA 符号到辅音/元音的映射表：

**中文辅音示例**：
`ɲ`, `n`, `ŋ`, `ʂ`, `w`, `ʈʂ`, `j`, `tɕ`, `m`, `k`, `t`, `p`, `pʲ`, `f`, `ʔ`, `s`, `kʰ`, `ʐ`, `l`, `x`

**中文元音示例**：
`i`, `o`, `a`, `ə`, `aw`, `ej`, `ow`, `e` （可能带声调标记如 `˥`, `˨˩˦`）

**日语辅音示例**：
`h`, `ɕ`, `t`, `dʑ`, `ɡ`, `k`, `n`, `ɲ`

**日语元音示例**：
`a`, `i`, `ɯ`, `e`, `o`, `aː`, `iː`, `oː`（长音用 `ː` 标记）

### 2. 特殊情况处理
- **纯元音音节**：如「あ」只有元音，无辅音
- **长音**：日语的 `oː`, `aː` 等
- **拨音**：日语的 `n` 单独成拍
- **促音**：日语的小「っ」
- **spn 标记**：MFA 的噪音/无法识别标记，需跳过

### 3. 别名生成策略
- 可使用 words 层的文字作为别名
- 或使用 phones 层的 IPA 转换为假名/拼音
- 支持 CV（辅音+元音）和 VCV 等不同录音方式

## 音源质量评分方案

### 问题背景
简单导出仅按时长排序，但时长较长的音频可能存在以下问题：
- 多个字合并（MFA 对齐错误）
- 音调转变大（不适合 UTAU 拉伸）
- 音量波动大（录音不稳定）

对于 UTAU 音源，需要更精细的质量评估。

### 评分维度

| 维度 | 指标 | 计算方式 | 理想值 | 权重建议 |
|-----|------|---------|-------|---------|
| 时长 | duration | 音频时长（秒） | 适中（0.3~0.8s） | 0.3 |
| 音量稳定性 | rms_variance | RMS 能量的方差 | 越小越好 | 0.3 |
| 音高稳定性 | f0_variance | 基频的方差 | 越小越好 | 0.4 |

### 各维度详细说明

#### 1. 时长评分 (Duration Score)
```python
def duration_score(duration: float) -> float:
    """
    时长评分：适中时长得分最高
    
    - 过短（<0.2s）：发音不完整
    - 过长（>1.0s）：可能包含多字或拖音
    - 最佳范围：0.3~0.8s
    """
    if duration < 0.2:
        return duration / 0.2 * 0.5  # 0~0.5分
    elif duration <= 0.8:
        return 1.0  # 满分
    elif duration <= 1.2:
        return 1.0 - (duration - 0.8) / 0.4 * 0.3  # 0.7~1.0分
    else:
        return max(0.3, 0.7 - (duration - 1.2) * 0.2)  # 递减
```

#### 2. 音量稳定性评分 (RMS Variance Score)
```python
def rms_variance_score(audio: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """
    音量稳定性评分：RMS 方差越小越好
    
    计算步骤：
    1. 将音频分帧（默认20ms一帧）
    2. 计算每帧的 RMS 能量
    3. 计算 RMS 序列的方差
    4. 归一化到 0~1 分数
    """
    frame_size = int(sr * frame_ms / 1000)
    frames = len(audio) // frame_size
    
    rms_values = []
    for i in range(frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)
    
    if len(rms_values) < 2:
        return 0.5  # 太短无法评估
    
    variance = np.var(rms_values)
    # 归一化：方差越小分数越高
    # 经验阈值：方差 < 0.01 为优秀，> 0.1 为较差
    score = max(0, 1.0 - variance * 10)
    return score
```

#### 3. 音高稳定性评分 (F0 Variance Score)
```python
def f0_variance_score(audio: np.ndarray, sr: int) -> float:
    """
    音高稳定性评分：F0 方差越小越好
    
    计算步骤：
    1. 使用 pyin/crepe/parselmouth 提取 F0
    2. 过滤无声帧（F0=0 或 NaN）
    3. 计算有效 F0 的方差
    4. 归一化到 0~1 分数
    
    依赖：librosa.pyin 或 parselmouth
    """
    import librosa
    
    # 提取 F0（使用 pyin 算法）
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'),  # ~65Hz
        fmax=librosa.note_to_hz('C6'),  # ~1047Hz
        sr=sr
    )
    
    # 过滤无效值
    valid_f0 = f0[~np.isnan(f0)]
    
    if len(valid_f0) < 3:
        return 0.5  # 无法评估
    
    # 转换为音分（cents）计算方差，避免频率绝对值影响
    # cents = 1200 * log2(f / f_ref)
    f0_cents = 1200 * np.log2(valid_f0 / np.median(valid_f0))
    variance = np.var(f0_cents)
    
    # 归一化：方差 < 100 cents² 为优秀，> 10000 cents² 为较差
    # 100 cents ≈ 1个半音
    score = max(0, 1.0 - variance / 10000)
    return score
```

### 综合评分计算

```python
def calculate_quality_score(
    audio: np.ndarray,
    sr: int,
    weights: dict = None,
    enabled_metrics: list = None
) -> float:
    """
    综合质量评分
    
    参数：
        audio: 音频数据
        sr: 采样率
        weights: 各维度权重，如 {"duration": 0.3, "rms": 0.3, "f0": 0.4}
        enabled_metrics: 启用的评分维度，如 ["duration", "rms", "f0"]
    
    返回：
        0~1 的综合分数
    """
    default_weights = {"duration": 0.3, "rms": 0.3, "f0": 0.4}
    weights = weights or default_weights
    enabled_metrics = enabled_metrics or ["duration", "rms", "f0"]
    
    scores = {}
    duration = len(audio) / sr
    
    if "duration" in enabled_metrics:
        scores["duration"] = duration_score(duration)
    
    if "rms" in enabled_metrics:
        scores["rms"] = rms_variance_score(audio, sr)
    
    if "f0" in enabled_metrics:
        scores["f0"] = f0_variance_score(audio, sr)
    
    # 加权平均（仅计算启用的维度）
    total_weight = sum(weights[k] for k in scores.keys())
    final_score = sum(scores[k] * weights[k] for k in scores.keys()) / total_weight
    
    return final_score
```

### 用户配置选项

```python
# 插件选项设计
PluginOption(
    key="quality_metrics",
    label="质量评估维度",
    option_type=OptionType.MULTI_SELECT,
    default=["duration"],
    choices=[
        ("duration", "时长（快速）"),
        ("rms", "音量稳定性（中速）"),
        ("f0", "音高稳定性（较慢）")
    ],
    description="选择用于排序的质量指标，多选时综合评分"
)

PluginOption(
    key="duration_weight",
    label="时长权重",
    option_type=OptionType.SLIDER,
    default=0.3,
    min_value=0,
    max_value=1,
    step=0.1,
    visible_when={"quality_metrics": "contains:duration"}
)

# 类似地添加 rms_weight 和 f0_weight
```

### 性能考虑

| 评估维度 | 耗时估算（每文件） | 依赖 |
|---------|------------------|------|
| duration | <1ms | 无 |
| rms | ~5ms | numpy |
| f0 | ~50-200ms | librosa 或 parselmouth |

建议：
- 默认仅启用 `duration`（兼容现有行为）
- UTAU 导出时推荐启用 `duration` + `f0`
- 完整评估启用全部三项

### 缓存策略

为避免重复计算，可将评分结果缓存到 JSON：

```json
// bank/{source}/quality_cache.json
{
  "version": "1.0",
  "metrics": ["duration", "rms", "f0"],
  "scores": {
    "segments/ba/1.wav": {
      "duration": 0.85,
      "rms": 0.72,
      "f0": 0.91,
      "combined": 0.83
    }
  }
}
```

## 待实现功能

- [x] 创建 `src/export_plugins/utau_oto_export.py`
- [x] 实现 IPA 音素分类器（辅音/元音判断）
- [x] 实现 TextGrid 解析与音素配对逻辑
- [x] 实现 oto.ini 参数计算
- [x] 支持中文和日语
- [x] 日语支持罗马音/平假名别名切换
- [x] 一个 wav 文件支持多条 oto 配置（不裁剪音频）
- [x] 每个别名最大样本数限制
- [x] 添加到 GUI 导出选项（通过 loader.py 自动注册）
- [x] 实现音源质量评分模块 `src/quality_scorer.py`
- [x] 在导出插件基类中集成质量评分接口
- [x] 为 UTAU 导出插件添加质量评分选项

## 参考资料
- [Wasteland UTAU - OTO Configuration](https://wastelandutau.neocities.org/en/config)
- [UTAU.us - OTO.ini 101](https://www.utau.us/oto.html)
- [SetParam OTOing Tutorial](https://sockhunter98.wixsite.com/socks-landing/otoing)
