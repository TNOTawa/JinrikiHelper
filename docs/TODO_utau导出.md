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

## 待实现功能

- [ ] 创建 `src/export_plugins/utau_oto_export.py`
- [ ] 实现 IPA 音素分类器（辅音/元音判断）
- [ ] 实现 TextGrid 解析与音素配对逻辑
- [ ] 实现 oto.ini 参数计算
- [ ] 支持中文和日语
- [ ] 支持自定义别名格式
- [ ] 支持批量导出整个 bank
- [ ] 添加到 GUI 导出选项

## 参考资料
- [Wasteland UTAU - OTO Configuration](https://wastelandutau.neocities.org/en/config)
- [UTAU.us - OTO.ini 101](https://www.utau.us/oto.html)
- [SetParam OTOing Tutorial](https://sockhunter98.wixsite.com/socks-landing/otoing)
