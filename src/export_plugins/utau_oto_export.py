# -*- coding: utf-8 -*-
"""
UTAU oto.ini 导出插件

从 TextGrid 提取音素时间边界，生成 UTAU 音源配置文件
一个 wav 文件可包含多条 oto 配置，无需裁剪音频
"""

import os
import json
import glob
import shutil
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .base import ExportPlugin, PluginOption, OptionType

logger = logging.getLogger(__name__)


# ==================== IPA 音素分类 ====================

# 中文辅音（MFA 输出的 IPA 符号）
CHINESE_CONSONANTS = {
    'p', 'pʰ', 'pʲ', 'b', 'm', 'f',
    't', 'tʰ', 'd', 'n', 'l',
    'k', 'kʰ', 'ɡ', 'g', 'ŋ', 'x', 'h',
    'tɕ', 'tɕʰ', 'dʑ', 'ɕ', 'ʑ',
    'ts', 'tsʰ', 'dz', 's', 'z',
    'ʈʂ', 'ʈʂʰ', 'ɖʐ', 'ʂ', 'ʐ',
    'ɲ', 'j', 'w', 'ɥ',
    'ʔ',  # 喉塞音
}

# 中文元音（可能带声调标记）
CHINESE_VOWELS = {
    'a', 'o', 'e', 'i', 'u', 'y', 'ü',
    'ə', 'ɛ', 'ɔ', 'ɤ', 'ɨ', 'ʅ', 'ʉ',
    'ai', 'ei', 'ao', 'ou',
    'ia', 'ie', 'iu', 'iao', 'iou',
    'ua', 'uo', 'ui', 'uai', 'uei',
    'üe', 'üan', 'ün',
    'an', 'en', 'in', 'un', 'ün',
    'ang', 'eng', 'ing', 'ong',
    'aw', 'ej', 'ow',  # MFA 输出格式
    'z̩',  # 舌尖元音
}

# 日语辅音
JAPANESE_CONSONANTS = {
    'p', 'b', 'm', 'ɸ',
    't', 'd', 'n', 's', 'z', 'ɾ', 'r',
    'k', 'ɡ', 'g', 'ŋ', 'h',
    'tɕ', 'dʑ', 'ɕ', 'ʑ',
    'ts', 'dz',
    'ɲ', 'j', 'w',
    # 长辅音
    'nː', 'sː', 'tː', 'kː', 'pː',
}

# 日语元音
JAPANESE_VOWELS = {
    'a', 'i', 'ɯ', 'u', 'e', 'o',
    'aː', 'iː', 'ɯː', 'uː', 'eː', 'oː',
}

# 跳过的标记
SKIP_MARKS = {'', 'SP', 'AP', '<unk>', 'spn', 'sil'}


def is_consonant(phone: str, language: str) -> bool:
    """判断音素是否为辅音"""
    base_phone = _strip_tone(phone)
    
    if language in ('chinese', 'zh', 'mandarin'):
        return base_phone in CHINESE_CONSONANTS
    elif language in ('japanese', 'ja', 'jp'):
        return base_phone in JAPANESE_CONSONANTS
    return False


def is_vowel(phone: str, language: str) -> bool:
    """判断音素是否为元音"""
    base_phone = _strip_tone(phone)
    
    if language in ('chinese', 'zh', 'mandarin'):
        if base_phone in CHINESE_VOWELS:
            return True
        for v in ['a', 'o', 'e', 'i', 'u', 'y', 'ə', 'ɛ', 'ɔ', 'ɤ', 'ɨ', 'ʅ', 'ʉ']:
            if base_phone.startswith(v):
                return True
        return False
    elif language in ('japanese', 'ja', 'jp'):
        return base_phone in JAPANESE_VOWELS or base_phone.rstrip('ː') in {'a', 'i', 'ɯ', 'u', 'e', 'o'}
    return False


def _strip_tone(phone: str) -> str:
    """移除声调标记"""
    tone_marks = '˥˦˧˨˩ˇˊˋ¯'
    result = phone
    for mark in tone_marks:
        result = result.replace(mark, '')
    return result


# ==================== IPA 到别名转换 ====================

# 中文 IPA 到拼音映射
CHINESE_IPA_TO_PINYIN = {
    # 辅音
    'p': 'b', 'pʰ': 'p', 'pʲ': 'p',
    'm': 'm', 'f': 'f',
    't': 'd', 'tʰ': 't',
    'n': 'n', 'l': 'l',
    'k': 'g', 'kʰ': 'k',
    'x': 'h', 'h': 'h',
    'tɕ': 'j', 'tɕʰ': 'q', 'ɕ': 'x',
    'ts': 'z', 'tsʰ': 'c', 's': 's',
    'ʈʂ': 'zh', 'ʈʂʰ': 'ch', 'ʂ': 'sh', 'ʐ': 'r',
    'ɲ': 'n', 'ŋ': 'ng',
    'j': 'y', 'w': 'w', 'ɥ': 'yu',
    'ʔ': '',
    # 元音
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'y': 'v', 'ü': 'v',
    'ə': 'e', 'ɛ': 'e', 'ɔ': 'o', 'ɤ': 'e',
    'ai': 'ai', 'ei': 'ei', 'ao': 'ao', 'ou': 'ou',
    'aw': 'ao', 'ej': 'ei', 'ow': 'ou',
    'z̩': 'i',
}

# 日语 IPA 到罗马音映射
JAPANESE_IPA_TO_ROMAJI = {
    # 辅音
    'p': 'p', 'b': 'b', 'm': 'm', 'ɸ': 'f',
    't': 't', 'd': 'd', 'n': 'n', 's': 's', 'z': 'z', 'ɾ': 'r', 'r': 'r',
    'k': 'k', 'ɡ': 'g', 'g': 'g', 'h': 'h',
    'tɕ': 'ch', 'dʑ': 'j', 'ɕ': 'sh', 'ʑ': 'j',
    'ts': 'ts', 'dz': 'z',
    'ɲ': 'ny', 'ŋ': 'ng', 'j': 'y', 'w': 'w',
    # 长辅音（促音后）
    'nː': 'n', 'sː': 's', 'tː': 't', 'kː': 'k', 'pː': 'p',
    # 元音
    'a': 'a', 'i': 'i', 'ɯ': 'u', 'u': 'u', 'e': 'e', 'o': 'o',
    'aː': 'a', 'iː': 'i', 'ɯː': 'u', 'uː': 'u', 'eː': 'e', 'oː': 'o',
}

# 罗马音到平假名映射
ROMAJI_TO_HIRAGANA = {
    # 基本元音
    'a': 'あ', 'i': 'い', 'u': 'う', 'e': 'え', 'o': 'お',
    # か行
    'ka': 'か', 'ki': 'き', 'ku': 'く', 'ke': 'け', 'ko': 'こ',
    # さ行
    'sa': 'さ', 'shi': 'し', 'si': 'し', 'su': 'す', 'se': 'せ', 'so': 'そ',
    # た行
    'ta': 'た', 'chi': 'ち', 'ti': 'ち', 'tsu': 'つ', 'tu': 'つ', 'te': 'て', 'to': 'と',
    # な行
    'na': 'な', 'ni': 'に', 'nu': 'ぬ', 'ne': 'ね', 'no': 'の',
    # は行
    'ha': 'は', 'hi': 'ひ', 'fu': 'ふ', 'hu': 'ふ', 'he': 'へ', 'ho': 'ほ',
    # ま行
    'ma': 'ま', 'mi': 'み', 'mu': 'む', 'me': 'め', 'mo': 'も',
    # や行
    'ya': 'や', 'yu': 'ゆ', 'yo': 'よ',
    # ら行
    'ra': 'ら', 'ri': 'り', 'ru': 'る', 're': 'れ', 'ro': 'ろ',
    # わ行
    'wa': 'わ', 'wo': 'を', 'n': 'ん',
    # が行
    'ga': 'が', 'gi': 'ぎ', 'gu': 'ぐ', 'ge': 'げ', 'go': 'ご',
    # ざ行
    'za': 'ざ', 'ji': 'じ', 'zi': 'じ', 'zu': 'ず', 'ze': 'ぜ', 'zo': 'ぞ',
    # だ行
    'da': 'だ', 'di': 'ぢ', 'du': 'づ', 'de': 'で', 'do': 'ど',
    # ば行
    'ba': 'ば', 'bi': 'び', 'bu': 'ぶ', 'be': 'べ', 'bo': 'ぼ',
    # ぱ行
    'pa': 'ぱ', 'pi': 'ぴ', 'pu': 'ぷ', 'pe': 'ぺ', 'po': 'ぽ',
    # 拗音
    'kya': 'きゃ', 'kyu': 'きゅ', 'kyo': 'きょ',
    'sha': 'しゃ', 'shu': 'しゅ', 'sho': 'しょ',
    'cha': 'ちゃ', 'chu': 'ちゅ', 'cho': 'ちょ',
    'nya': 'にゃ', 'nyu': 'にゅ', 'nyo': 'にょ',
    'hya': 'ひゃ', 'hyu': 'ひゅ', 'hyo': 'ひょ',
    'mya': 'みゃ', 'myu': 'みゅ', 'myo': 'みょ',
    'rya': 'りゃ', 'ryu': 'りゅ', 'ryo': 'りょ',
    'gya': 'ぎゃ', 'gyu': 'ぎゅ', 'gyo': 'ぎょ',
    'ja': 'じゃ', 'ju': 'じゅ', 'jo': 'じょ',
    'bya': 'びゃ', 'byu': 'びゅ', 'byo': 'びょ',
    'pya': 'ぴゃ', 'pyu': 'ぴゅ', 'pyo': 'ぴょ',
}


def ipa_to_alias(consonant: Optional[str], vowel: Optional[str], language: str, use_hiragana: bool = False) -> Optional[str]:
    """将 IPA 音素转换为别名"""
    c_base = _strip_tone(consonant) if consonant else ''
    v_base = _strip_tone(vowel) if vowel else ''
    
    if language in ('chinese', 'zh', 'mandarin'):
        c_alias = CHINESE_IPA_TO_PINYIN.get(c_base, c_base)
        v_alias = CHINESE_IPA_TO_PINYIN.get(v_base, v_base)
        alias = (c_alias or '') + (v_alias or '')
        # 清理非 ASCII 字符
        alias = ''.join(c for c in alias if c.isascii() and (c.isalnum() or c == '_'))
        return alias.lower() if alias else None
    else:
        # 日语
        c_alias = JAPANESE_IPA_TO_ROMAJI.get(c_base, c_base)
        v_alias = JAPANESE_IPA_TO_ROMAJI.get(v_base, v_base)
        romaji = (c_alias or '') + (v_alias or '')
        # 清理非 ASCII
        romaji = ''.join(c for c in romaji if c.isascii() and (c.isalnum() or c == '_'))
        romaji = romaji.lower()
        
        if not romaji:
            return None
        
        if use_hiragana:
            # 尝试转换为平假名
            return ROMAJI_TO_HIRAGANA.get(romaji, romaji)
        return romaji


class UTAUOtoExportPlugin(ExportPlugin):
    """UTAU oto.ini 导出插件"""
    
    name = "UTAU oto.ini 导出"
    description = "从 TextGrid 生成 UTAU 音源配置文件，一个 wav 可包含多条配置"
    version = "1.1.0"
    author = "内置"
    
    def get_options(self) -> List[PluginOption]:
        return [
            PluginOption(
                key="info",
                label="从 TextGrid phones 层提取音素，生成 oto.ini（音频不裁剪）",
                option_type=OptionType.LABEL
            ),
            PluginOption(
                key="max_samples",
                label="每个别名最大样本数",
                option_type=OptionType.NUMBER,
                default=5,
                min_value=1,
                max_value=100,
                description="同一别名保留的最大条目数"
            ),
            PluginOption(
                key="quality_metrics",
                label="质量评估维度",
                option_type=OptionType.COMBO,
                default="duration+rms",
                choices=["duration", "duration+rms", "duration+f0", "all"],
                description="duration=仅时长, +rms=音量稳定性, +f0=音高稳定性。选择 all 可能耗时较长"
            ),
            PluginOption(
                key="naming_rule",
                label="别名命名规则",
                option_type=OptionType.TEXT,
                default="%p%%n%",
                description="变量: %p%=拼音/罗马音, %n%=序号。示例: %p%_%n% → ba_1"
            ),
            PluginOption(
                key="first_naming_rule",
                label="首个样本命名规则",
                option_type=OptionType.TEXT,
                default="%p%",
                description="第0个样本的特殊规则，留空则使用通用规则。示例: %p% → ba"
            ),
            PluginOption(
                key="alias_style",
                label="别名风格（日语）",
                option_type=OptionType.COMBO,
                default="hiragana",
                choices=["romaji", "hiragana"],
                description="日语音源的别名格式：罗马音或平假名"
            ),
            PluginOption(
                key="overlap_ratio",
                label="Overlap 比例",
                option_type=OptionType.NUMBER,
                default=0.3,
                min_value=0.1,
                max_value=0.5,
                description="Overlap = Preutterance × 此比例"
            ),
            PluginOption(
                key="encoding",
                label="文件编码",
                option_type=OptionType.COMBO,
                default="shift_jis",
                choices=["shift_jis", "utf-8", "gbk"],
                description="oto.ini 和 character.txt 编码（UTAU 标准为 Shift_JIS）"
            ),
            PluginOption(
                key="sanitize_filename",
                label="文件名转拼音",
                option_type=OptionType.SWITCH,
                default=False,
                description="将中文文件名转为拼音，清理特殊字符，防止 UTAU 识别故障"
            ),
        ]
    
    def export(
        self,
        source_name: str,
        bank_dir: str,
        options: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """执行 UTAU oto.ini 导出"""
        try:
            # 加载语言设置
            language = self._load_language_from_meta(bank_dir, source_name)
            
            # 获取选项
            max_samples = int(options.get("max_samples", 5))
            quality_metrics = options.get("quality_metrics", "duration")
            naming_rule = options.get("naming_rule", "%p%%n%")
            first_naming_rule = options.get("first_naming_rule", "%p%")
            alias_style = options.get("alias_style", "romaji")
            overlap_ratio = float(options.get("overlap_ratio", 0.3))
            encoding = options.get("encoding", "utf-8")
            sanitize_filename = options.get("sanitize_filename", False)
            use_hiragana = (alias_style == "hiragana") and language in ('japanese', 'ja', 'jp')
            
            # 解析质量评估维度
            enabled_metrics = self._parse_quality_metrics(quality_metrics)
            
            paths = self.get_source_paths(bank_dir, source_name)
            export_dir = self.get_export_dir(bank_dir, source_name, "utau_oto")
            
            os.makedirs(export_dir, exist_ok=True)
            
            # 步骤1: 解析 TextGrid 并生成 oto 条目
            self._log("【解析 TextGrid 文件】")
            oto_entries, wav_files = self._parse_textgrids(
                paths["slices_dir"],
                paths["textgrid_dir"],
                language,
                use_hiragana,
                overlap_ratio
            )
            
            if not oto_entries:
                return False, "未能从 TextGrid 提取有效音素"
            
            self._log(f"提取到 {len(oto_entries)} 条原始 oto 配置")
            
            # 步骤2: 按别名分组并限制数量，添加编号
            self._log(f"\n【筛选最佳样本】评估维度: {enabled_metrics}")
            filtered_entries, used_wavs = self._filter_by_alias(
                oto_entries, max_samples, naming_rule, first_naming_rule,
                paths["slices_dir"], enabled_metrics
            )
            self._log(f"筛选后保留 {len(filtered_entries)} 条配置，涉及 {len(used_wavs)} 个音频文件")
            
            # 步骤3: 复制音频文件（可选文件名转拼音）
            self._log("\n【复制音频文件】")
            if sanitize_filename:
                self._log("已启用文件名转拼音")
            copied, filename_map = self._copy_wav_files(
                used_wavs, paths["slices_dir"], export_dir, sanitize_filename
            )
            self._log(f"复制了 {copied} 个音频文件")
            
            # 步骤4: 写入 oto.ini
            self._log("\n【生成 oto.ini】")
            oto_path = os.path.join(export_dir, "oto.ini")
            self._write_oto_ini(filtered_entries, oto_path, encoding, filename_map)
            self._log(f"写入: {oto_path}")
            
            # 步骤5: 写入 character.txt
            self._log("\n【生成 character.txt】")
            char_path = os.path.join(export_dir, "character.txt")
            self._write_character_txt(source_name, char_path, encoding)
            self._log(f"写入: {char_path}")
            
            # 统计别名数量
            unique_aliases = set(e["alias"] for e in filtered_entries)
            return True, f"导出完成: {export_dir}\n{len(unique_aliases)} 个别名，{len(filtered_entries)} 条配置，{copied} 个音频"
            
        except Exception as e:
            logger.error(f"UTAU oto.ini 导出失败: {e}", exc_info=True)
            return False, str(e)
    
    def _parse_quality_metrics(self, metrics_str: str) -> List[str]:
        """解析质量评估维度选项"""
        if metrics_str == "all":
            return ["duration", "rms", "f0"]
        elif metrics_str == "duration+rms":
            return ["duration", "rms"]
        elif metrics_str == "duration+f0":
            return ["duration", "f0"]
        else:
            return ["duration"]
    
    def _load_language_from_meta(self, bank_dir: str, source_name: str) -> str:
        """从 meta.json 加载语言设置"""
        meta_path = os.path.join(bank_dir, source_name, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    language = meta.get("language", "chinese")
                    self._log(f"语言设置: {language}")
                    return language
        except Exception as e:
            logger.warning(f"读取 meta.json 失败: {e}")
        return "chinese"
    
    def _parse_textgrids(
        self,
        slices_dir: str,
        textgrid_dir: str,
        language: str,
        use_hiragana: bool,
        overlap_ratio: float
    ) -> Tuple[List[Dict], set]:
        """解析 TextGrid 文件，提取音素边界"""
        import textgrid
        import soundfile as sf
        
        tg_files = glob.glob(os.path.join(textgrid_dir, '*.TextGrid'))
        if not tg_files:
            self._log("未找到 TextGrid 文件")
            return [], set()
        
        self._log(f"处理 {len(tg_files)} 个 TextGrid 文件")
        
        oto_entries = []
        wav_files = set()
        
        for tg_path in tg_files:
            basename = os.path.basename(tg_path).replace('.TextGrid', '')
            wav_name = basename + '.wav'
            wav_path = os.path.join(slices_dir, wav_name)
            
            if not os.path.exists(wav_path):
                continue
            
            try:
                info = sf.info(wav_path)
                wav_duration_ms = info.duration * 1000
            except Exception:
                continue
            
            wav_files.add(wav_name)
            
            try:
                tg = textgrid.TextGrid.fromFile(tg_path)
            except Exception:
                continue
            
            # 查找 words 层和 phones 层
            words_tier = None
            phones_tier = None
            for tier in tg:
                name_lower = tier.name.lower()
                if name_lower in ('words', 'word'):
                    words_tier = tier
                elif name_lower in ('phones', 'phone'):
                    phones_tier = tier
            
            # 如果没找到，按顺序取
            if words_tier is None and len(tg) >= 1:
                words_tier = tg[0]
            if phones_tier is None and len(tg) >= 2:
                phones_tier = tg[1]
            
            if phones_tier is None:
                continue
            
            # 提取音素对，使用 words 层限制配对范围
            entries = self._extract_cv_pairs(
                words_tier, phones_tier, wav_name, wav_duration_ms,
                language, use_hiragana, overlap_ratio
            )
            oto_entries.extend(entries)
        
        return oto_entries, wav_files
    
    def _extract_cv_pairs(
        self,
        words_tier,
        phones_tier,
        wav_name: str,
        wav_duration_ms: float,
        language: str,
        use_hiragana: bool,
        overlap_ratio: float
    ) -> List[Dict]:
        """
        从 phones 层提取辅音+元音对
        使用 words 层限制配对范围，确保辅音和元音属于同一个字
        """
        entries = []
        
        # 构建 word 时间范围列表
        word_ranges = []
        if words_tier:
            for interval in words_tier:
                text = interval.mark.strip()
                if text and text not in SKIP_MARKS:
                    word_ranges.append((interval.minTime, interval.maxTime))
        
        def get_word_range(time: float) -> Optional[Tuple[float, float]]:
            """获取某时间点所属的 word 范围"""
            for start, end in word_ranges:
                if start <= time < end:
                    return (start, end)
            return None
        
        def same_word(time1: float, time2: float) -> bool:
            """判断两个时间点是否在同一个 word 内"""
            if not word_ranges:
                return True  # 没有 words 层时不限制
            range1 = get_word_range(time1)
            range2 = get_word_range(time2)
            return range1 is not None and range1 == range2
        
        intervals = list(phones_tier)
        i = 0
        
        while i < len(intervals):
            interval = intervals[i]
            phone = interval.mark.strip()
            
            if phone in SKIP_MARKS:
                i += 1
                continue
            
            start_ms = interval.minTime * 1000
            end_ms = interval.maxTime * 1000
            
            if is_consonant(phone, language):
                consonant = phone
                consonant_start = start_ms
                consonant_end = end_ms
                consonant_time = interval.minTime  # 用于判断所属 word
                
                vowel = None
                vowel_end = end_ms
                
                # 检查下一个音素是否是元音，且在同一个 word 内
                if i + 1 < len(intervals):
                    next_interval = intervals[i + 1]
                    next_phone = next_interval.mark.strip()
                    next_time = next_interval.minTime
                    
                    if (next_phone not in SKIP_MARKS and 
                        is_vowel(next_phone, language) and
                        same_word(consonant_time, next_time)):
                        vowel = next_phone
                        vowel_end = next_interval.maxTime * 1000
                        i += 1
                
                alias = ipa_to_alias(consonant, vowel, language, use_hiragana)
                if not alias:
                    i += 1
                    continue
                
                consonant_duration = consonant_end - consonant_start
                
                entry = self._calculate_oto_params(
                    wav_name=wav_name,
                    alias=alias,
                    offset=consonant_start,
                    consonant_duration=consonant_duration,
                    segment_end=vowel_end,
                    wav_duration_ms=wav_duration_ms,
                    overlap_ratio=overlap_ratio
                )
                entries.append(entry)
                
            elif is_vowel(phone, language):
                alias = ipa_to_alias(None, phone, language, use_hiragana)
                if not alias:
                    i += 1
                    continue
                
                entry = self._calculate_oto_params(
                    wav_name=wav_name,
                    alias=alias,
                    offset=start_ms,
                    consonant_duration=min(30, (end_ms - start_ms) * 0.2),
                    segment_end=end_ms,
                    wav_duration_ms=wav_duration_ms,
                    overlap_ratio=overlap_ratio
                )
                entries.append(entry)
            
            i += 1
        
        return entries
    
    def _calculate_oto_params(
        self,
        wav_name: str,
        alias: str,
        offset: float,
        consonant_duration: float,
        segment_end: float,
        wav_duration_ms: float,
        overlap_ratio: float
    ) -> Dict:
        """
        计算 oto.ini 参数
        
        oto.ini 格式: wav=alias,offset,consonant,cutoff,preutterance,overlap
        
        - offset: 从音频开头跳过的毫秒数
        - consonant: 不被拉伸的区域长度
        - cutoff: 负值，表示这个音素的总时长（从 offset 开始）
        - preutterance: 先行发声
        - overlap: 与前一音符的交叉淡化区域
        """
        segment_duration = segment_end - offset
        preutterance = consonant_duration
        overlap = preutterance * overlap_ratio
        
        # cutoff 为负值，表示音素的总时长
        cutoff = -segment_duration
        
        return {
            "wav_name": wav_name,
            "alias": alias,
            "offset": round(offset, 1),
            "consonant": round(consonant_duration, 1),
            "cutoff": round(cutoff, 1),
            "preutterance": round(preutterance, 1),
            "overlap": round(overlap, 1),
            "segment_duration": segment_duration,  # 用于排序
        }
    
    def _filter_by_alias(
        self,
        entries: List[Dict],
        max_samples: int,
        naming_rule: str,
        first_naming_rule: str,
        slices_dir: str,
        enabled_metrics: List[str]
    ) -> Tuple[List[Dict], set]:
        """按别名分组，使用质量评分筛选最佳样本，并添加编号"""
        # 过滤空别名
        valid_entries = [e for e in entries if e.get("alias") and e["alias"].strip()]
        
        # 按基础别名分组
        alias_groups: Dict[str, List[Dict]] = defaultdict(list)
        for entry in valid_entries:
            alias_groups[entry["alias"]].append(entry)
        
        # 判断是否需要加载音频计算质量分数
        need_audio_scoring = any(m in enabled_metrics for m in ["rms", "f0"])
        
        filtered = []
        used_wavs = set()
        
        for base_alias, group in alias_groups.items():
            # 计算质量分数
            if need_audio_scoring:
                scored_group = self._score_entries(group, slices_dir, enabled_metrics)
            else:
                # 仅使用时长评分
                from ..quality_scorer import duration_score
                for entry in group:
                    duration = entry["segment_duration"] / 1000  # 转换为秒
                    entry["quality_score"] = duration_score(duration)
                scored_group = group
            
            # 按质量分数排序（降序）
            sorted_group = sorted(scored_group, key=lambda x: -x.get("quality_score", 0))
            
            # 保留前 N 个，并应用命名规则
            for idx, entry in enumerate(sorted_group[:max_samples]):
                # 生成带编号的别名
                if idx == 0 and first_naming_rule:
                    final_alias = self._apply_naming_rule(first_naming_rule, base_alias, idx)
                else:
                    final_alias = self._apply_naming_rule(naming_rule, base_alias, idx)
                
                entry["alias"] = final_alias
                filtered.append(entry)
                used_wavs.add(entry["wav_name"])
        
        return filtered, used_wavs
    
    def _score_entries(
        self,
        entries: List[Dict],
        slices_dir: str,
        enabled_metrics: List[str]
    ) -> List[Dict]:
        """为条目计算质量分数"""
        import soundfile as sf
        from ..quality_scorer import QualityScorer
        
        scorer = QualityScorer(enabled_metrics=enabled_metrics)
        
        # 缓存已加载的音频
        audio_cache: Dict[str, Tuple] = {}
        
        for entry in entries:
            wav_name = entry["wav_name"]
            wav_path = os.path.join(slices_dir, wav_name)
            
            try:
                # 加载或使用缓存的音频
                if wav_name not in audio_cache:
                    audio, sr = sf.read(wav_path)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    audio_cache[wav_name] = (audio, sr)
                else:
                    audio, sr = audio_cache[wav_name]
                
                # 提取片段（根据 offset 和 segment_duration）
                offset_samples = int(entry["offset"] / 1000 * sr)
                duration_samples = int(entry["segment_duration"] / 1000 * sr)
                segment = audio[offset_samples:offset_samples + duration_samples]
                
                if len(segment) > 0:
                    scores = scorer.score(segment, sr)
                    entry["quality_score"] = scores.get("combined", 0.5)
                else:
                    entry["quality_score"] = 0.5
                    
            except Exception as e:
                logger.warning(f"评分失败 {wav_name}: {e}")
                entry["quality_score"] = 0.5
        
        return entries
    
    def _apply_naming_rule(self, rule: str, base_alias: str, index: int) -> str:
        """应用命名规则生成别名"""
        return rule.replace("%p%", base_alias).replace("%n%", str(index))
    
    def _copy_wav_files(
        self,
        wav_files: set,
        slices_dir: str,
        export_dir: str,
        sanitize: bool = False
    ) -> Tuple[int, Dict[str, str]]:
        """
        复制音频文件到导出目录
        
        参数:
            wav_files: 需要复制的文件名集合
            slices_dir: 源目录
            export_dir: 目标目录
            sanitize: 是否对文件名进行转拼音和清理
        
        返回:
            (复制数量, 文件名映射表 {原文件名: 新文件名})
        """
        copied = 0
        filename_map: Dict[str, str] = {}
        used_names: set = set()
        
        for wav_name in wav_files:
            src = os.path.join(slices_dir, wav_name)
            if not os.path.exists(src):
                continue
            
            if sanitize:
                new_name = self._sanitize_filename(wav_name, used_names)
                used_names.add(new_name)
            else:
                new_name = wav_name
            
            filename_map[wav_name] = new_name
            dst = os.path.join(export_dir, new_name)
            shutil.copyfile(src, dst)
            copied += 1
        
        return copied, filename_map
    
    def _sanitize_filename(self, filename: str, used_names: set) -> str:
        """
        清理文件名：中文转拼音 + 特殊字符清理 + 防冲突
        
        参数:
            filename: 原文件名
            used_names: 已使用的文件名集合（用于防冲突）
        
        返回:
            清理后的文件名
        """
        from pypinyin import lazy_pinyin
        import re
        
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        
        # 中文转拼音
        pinyin_parts = lazy_pinyin(name)
        sanitized = ''.join(pinyin_parts)
        
        # 清理特殊字符，只保留字母、数字、下划线、连字符
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', sanitized)
        
        # 合并连续下划线
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # 去除首尾下划线
        sanitized = sanitized.strip('_')
        
        # 如果为空，使用默认名
        if not sanitized:
            sanitized = 'audio'
        
        # 防冲突：添加数字后缀
        base_name = sanitized
        counter = 1
        while f"{sanitized}{ext}" in used_names:
            sanitized = f"{base_name}_{counter}"
            counter += 1
        
        return f"{sanitized}{ext}"
    
    def _write_oto_ini(
        self,
        entries: List[Dict],
        output_path: str,
        encoding: str,
        filename_map: Optional[Dict[str, str]] = None
    ):
        """
        写入 oto.ini 文件
        
        参数:
            entries: oto 条目列表
            output_path: 输出路径
            encoding: 文件编码
            filename_map: 文件名映射表（原文件名 -> 新文件名）
        """
        lines = []
        for entry in entries:
            # 跳过空别名
            alias = entry.get("alias", "")
            if not alias or not alias.strip():
                logger.warning(f"跳过空别名: {entry.get('wav_name', 'unknown')}")
                continue
            
            # 应用文件名映射
            wav_name = entry["wav_name"]
            if filename_map and wav_name in filename_map:
                wav_name = filename_map[wav_name]
            
            line = "{wav}={alias},{offset},{consonant},{cutoff},{preutterance},{overlap}".format(
                wav=wav_name,
                alias=alias,
                offset=entry["offset"],
                consonant=entry["consonant"],
                cutoff=entry["cutoff"],
                preutterance=entry["preutterance"],
                overlap=entry["overlap"]
            )
            lines.append(line)
        
        # 按 wav 文件名 + 别名排序
        lines.sort(key=lambda x: (x.split('=')[0], x.split('=')[1].split(',')[0]))
        
        with open(output_path, 'w', encoding=encoding) as f:
            f.write('\n'.join(lines))
    
    def _write_character_txt(
        self,
        source_name: str,
        output_path: str,
        encoding: str
    ):
        """写入 character.txt 文件，用于 UTAU 识别音源名称"""
        with open(output_path, 'w', encoding=encoding) as f:
            f.write(f"name={source_name}")
