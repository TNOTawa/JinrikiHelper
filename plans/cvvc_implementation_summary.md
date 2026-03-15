# CVVC 音源导出功能实现总结

## 实现完成时间
2026-02-04

## 版本更新
- 插件版本从 1.1.0 更新至 **1.2.0**

## 新增功能

### 1. CVVC 模式支持
为 UTAU oto.ini 导出插件添加了 CVVC（Consonant-Vowel-Vowel-Consonant）音源导出功能，可额外生成 **VC 部（元音到辅音过渡）** 条目。

### 2. 新增配置选项

在 [`get_options()`](src/export_plugins/utau_oto_export.py:254) 方法中添加了 4 个新选项：

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cvvc_mode` | 开关 | False | 启用/禁用 CVVC 模式 |
| `vc_alias_separator` | 下拉 | " " (空格) | VC 别名分隔符（空格/下划线/连字符） |
| `vc_offset_ratio` | 数字 | 0.5 | VC 偏移比例（0.3-0.8） |
| `vc_overlap_ratio` | 数字 | 0.5 | VC Overlap 比例（0.3-0.8） |

### 3. 新增方法

#### [`_calculate_vc_params()`](src/export_plugins/utau_oto_export.py:688)
计算 VC 部的 oto.ini 参数，包括：
- **offset**: 元音后半段位置
- **consonant**: 固定区域（较短）
- **cutoff**: 负值，到辅音结束
- **preutterance**: 从 offset 到辅音开始的距离
- **overlap**: 较大，平滑过渡

#### [`_extract_vc_pairs()`](src/export_plugins/utau_oto_export.py:649)
从 TextGrid 的 phones 层提取元音+辅音对（VC 部），遍历音素序列，当检测到元音后跟辅音时生成 VC 条目。

### 4. 修改的方法

#### [`_parse_textgrids()`](src/export_plugins/utau_oto_export.py:539)
- 添加了 4 个新参数支持 CVVC 模式
- 在提取 CV 对后，如果启用 CVVC 模式，额外调用 [`_extract_vc_pairs()`](src/export_plugins/utau_oto_export.py:649) 提取 VC 对

#### [`export()`](src/export_plugins/utau_oto_export.py:397)
- 读取 CVVC 相关配置选项
- 根据 CVVC 模式显示不同的日志信息
- 将 CVVC 参数传递给 [`_parse_textgrids()`](src/export_plugins/utau_oto_export.py:539)

## 工作原理

### VC 部提取流程

```
TextGrid phones 层:
[元音 V] → [辅音 C] → [元音 V] → [辅音 C]
    ↓           ↓           ↓           ↓
生成 VC 条目:  [V C]              [V C]
```

### VC 参数计算示例

假设：
- 元音时长：100ms (0-100ms)
- 辅音时长：60ms (100-160ms)
- `vc_offset_ratio` = 0.5
- `vc_overlap_ratio` = 0.5

计算结果：
- **offset** = 100 - 100×0.5 = 50ms
- **segment_duration** = 160 - 50 = 110ms
- **preutterance** = 100 - 50 = 50ms
- **consonant** = min(30, 110×0.3) = 30ms
- **overlap** = 50×0.5 = 25ms
- **cutoff** = -110ms

## 输出示例

启用 CVVC 模式后，oto.ini 将包含：

```ini
# CV 部（现有功能）
test_0000.wav=ba,30,50,-110,50,15
test_0000.wav=ka,140,60,-140,60,18

# VC 部（新增功能）
test_0000.wav=a k,70,20,-90,40,20
test_0000.wav=a n,180,25,-100,45,22
```

## 代码验证

✅ Python 语法检查通过
```bash
py -m py_compile src\export_plugins\utau_oto_export.py
# Exit code: 0 (成功)
```

## 使用方法

1. 在导出插件界面中找到 "UTAU oto.ini 导出" 插件
2. 启用 **"CVVC 模式"** 开关
3. 根据需要调整以下参数：
   - **VC 别名分隔符**：选择空格、下划线或连字符
   - **VC 偏移比例**：控制 VC 开始位置（推荐 0.5）
   - **VC Overlap 比例**：控制过渡平滑度（推荐 0.5）
4. 执行导出

## 技术特点

1. **无损兼容**：CVVC 模式为可选功能，不影响现有 CV 导出
2. **参数可调**：提供多个参数供用户微调 VC 部效果
3. **自动提取**：从 TextGrid 自动识别元音-辅音序列
4. **质量筛选**：VC 部条目同样参与质量评分和筛选
5. **编码兼容**：VC 别名支持多种分隔符，兼容不同编码

## 注意事项

1. VC 部的提取依赖于 TextGrid 中音素的正确标注
2. VC 别名使用分隔符（默认空格）连接元音和辅音
3. VC 参数的计算基于元音和辅音的时间边界
4. 建议先用小数据集测试参数效果，再批量导出

## 后续优化建议

1. 支持 VV 部（元音到元音过渡）
2. 支持跨字边界的 VC 提取控制
3. 添加 VC 部专用的质量评估指标
4. 支持自定义 VC 别名格式模板
