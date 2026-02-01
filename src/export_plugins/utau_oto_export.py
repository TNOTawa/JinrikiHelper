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

# ==================== 模糊拼字近似音素对照表 ====================

# 声母近似组（同组内音素互为替代，按优先级排序）
FUZZY_CONSONANT_GROUPS = [
    ('sh', 's'),       # 翘舌/平舌
    ('zh', 'z'),       # 翘舌/平舌
    ('ch', 'c'),       # 翘舌/平舌
    ('l', 'n', 'r'),   # 边音/鼻音/卷舌
    ('f', 'h'),        # 唇齿/喉音
]

# 韵母近似组（同组内音素互为替代，按优先级排序）
FUZZY_VOWEL_GROUPS = [
    ('an', 'ang'),       # 前鼻/后鼻
    ('en', 'eng', 'ong'), # 前鼻/后鼻/后鼻圆唇
    ('in', 'ing'),       # 前鼻/后鼻
    ('ian', 'iang'),     # 前鼻/后鼻
    ('uan', 'uang'),     # 前鼻/后鼻
]


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
                key="cross_language",
                label="跨语种导出",
                option_type=OptionType.SWITCH,
                default=False,
                description="【TODO】启用中跨日或日跨中的音素映射导出"
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
                key="auto_phoneme_combine",
                label="自动拼字",
                option_type=OptionType.SWITCH,
                default=False,
                description="用已有的高质量音素拼接生成缺失的音素组合"
            ),
            PluginOption(
                key="crossfade_ms",
                label="拼接淡入淡出时长(ms)",
                option_type=OptionType.NUMBER,
                default=10,
                min_value=5,
                max_value=50,
                description="自动拼字时辅音与元音之间的交叉淡化时长",
                visible_when={"auto_phoneme_combine": True}
            ),
            PluginOption(
                key="fuzzy_phoneme",
                label="模糊拼字",
                option_type=OptionType.SWITCH,
                default=False,
                description="用近似声母/韵母替代缺失音素（如 sh↔s, an↔ang），仅中文有效",
                visible_when={"auto_phoneme_combine": True}
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
                key="character_name",
                label="角色名称",
                option_type=OptionType.TEXT,
                default="",
                description="character.txt 中的角色名，留空则使用音源名称"
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
            # 使用基类方法加载语言设置
            language = self.load_language_from_meta(bank_dir, source_name)
            
            # 获取选项
            max_samples = int(options.get("max_samples", 5))
            quality_metrics = options.get("quality_metrics", "duration")
            naming_rule = options.get("naming_rule", "%p%%n%")
            first_naming_rule = options.get("first_naming_rule", "%p%")
            alias_style = options.get("alias_style", "romaji")
            overlap_ratio = float(options.get("overlap_ratio", 0.3))
            encoding = options.get("encoding", "utf-8")
            character_name = options.get("character_name", "").strip()
            auto_phoneme_combine = options.get("auto_phoneme_combine", False)
            crossfade_ms = int(options.get("crossfade_ms", 10))
            fuzzy_phoneme = options.get("fuzzy_phoneme", False)
            use_hiragana = (alias_style == "hiragana") and language in ('japanese', 'ja', 'jp')
            
            # 使用基类方法解析质量评估维度
            enabled_metrics = self.parse_quality_metrics(quality_metrics)
            
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
            
            # 步骤2.5: 自动拼字（如果启用）
            combined_count = 0
            if auto_phoneme_combine:
                self._log("\n【自动拼字】")
                combined_entries, combined_wavs = self._auto_combine_phonemes(
                    oto_entries,
                    filtered_entries,
                    paths["slices_dir"],
                    export_dir,
                    language,
                    use_hiragana,
                    overlap_ratio,
                    crossfade_ms,
                    first_naming_rule,
                    fuzzy_phoneme
                )
                if combined_entries:
                    filtered_entries.extend(combined_entries)
                    used_wavs.update(combined_wavs)
                    combined_count = len(combined_entries)
                    self._log(f"拼接生成 {combined_count} 条新配置")
            
            # 步骤3: 复制音频文件（自动检测文件名是否需要转拼音）
            self._log("\n【复制音频文件】")
            copied, filename_map = self._copy_wav_files(
                used_wavs, paths["slices_dir"], export_dir, encoding
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
            # 使用自定义角色名，留空则使用音源名称
            final_character_name = character_name if character_name else source_name
            self._write_character_txt(final_character_name, char_path, encoding)
            self._log(f"写入: {char_path}")
            
            # 统计别名数量
            unique_aliases = set(e["alias"] for e in filtered_entries)
            result_msg = f"导出完成: {export_dir}\n{len(unique_aliases)} 个别名，{len(filtered_entries)} 条配置，{copied} 个音频"
            if combined_count > 0:
                result_msg += f"\n（其中 {combined_count} 条为自动拼接生成）"
            return True, result_msg
            
        except Exception as e:
            logger.error(f"UTAU oto.ini 导出失败: {e}", exc_info=True)
            return False, str(e)
    
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
                # 使用基类方法应用命名规则
                if idx == 0 and first_naming_rule:
                    final_alias = self.apply_naming_rule(first_naming_rule, base_alias, idx)
                else:
                    final_alias = self.apply_naming_rule(naming_rule, base_alias, idx)
                
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
    
    def _copy_wav_files(
        self,
        wav_files: set,
        slices_dir: str,
        export_dir: str,
        encoding: str = "shift_jis"
    ) -> Tuple[int, Dict[str, str]]:
        """
        复制音频文件到导出目录
        
        参数:
            wav_files: 需要复制的文件名集合
            slices_dir: 源目录
            export_dir: 目标目录
            encoding: 目标编码，用于检测文件名是否合法
        
        返回:
            (复制数量, 文件名映射表 {原文件名: 新文件名})
        """
        copied = 0
        filename_map: Dict[str, str] = {}
        used_names: set = set()
        sanitized_count = 0
        
        for wav_name in wav_files:
            src = os.path.join(slices_dir, wav_name)
            if not os.path.exists(src):
                continue
            
            # 检测文件名是否能用指定编码表示
            if self._is_filename_valid(wav_name, encoding):
                new_name = wav_name
            else:
                new_name = self._sanitize_filename(wav_name, used_names)
                sanitized_count += 1
            
            used_names.add(new_name)
            filename_map[wav_name] = new_name
            dst = os.path.join(export_dir, new_name)
            shutil.copyfile(src, dst)
            copied += 1
        
        if sanitized_count > 0:
            self._log(f"已将 {sanitized_count} 个文件名转换为拼音（原文件名无法用 {encoding} 编码）")
        
        return copied, filename_map
    
    def _is_filename_valid(self, filename: str, encoding: str) -> bool:
        """
        检测文件名是否合法（能否用指定编码表示）
        
        参数:
            filename: 文件名
            encoding: 目标编码
        
        返回:
            True 表示文件名合法，False 表示需要转换
        """
        try:
            filename.encode(encoding)
            return True
        except UnicodeEncodeError:
            return False
    
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
        character_name: str,
        output_path: str,
        encoding: str
    ):
        """写入 character.txt 文件，用于 UTAU 识别音源名称
        
        参数:
            character_name: 角色名称（可以是用户自定义的名称或音源名称）
            output_path: 输出路径
            encoding: 文件编码
        
        注意：当角色名称包含无法用指定编码表示的字符时，
        自动将名称转换为拼音/罗马音。
        """
        name_to_write = character_name
        
        # 检测是否能用指定编码
        try:
            character_name.encode(encoding)
        except UnicodeEncodeError:
            # 无法编码，转换为拼音
            from pypinyin import lazy_pinyin
            pinyin_name = ''.join(lazy_pinyin(character_name))
            logger.warning(f"角色名称 '{character_name}' 无法用 {encoding} 编码，已转换为拼音: {pinyin_name}")
            self._log(f"角色名称 '{character_name}' 无法用 {encoding} 编码，已转换为拼音: {pinyin_name}")
            name_to_write = pinyin_name
        
        with open(output_path, 'w', encoding=encoding) as f:
            f.write(f"name={name_to_write}")

    # ==================== 自动拼字功能 ====================
    
    def _auto_combine_phonemes(
        self,
        all_entries: List[Dict],
        filtered_entries: List[Dict],
        slices_dir: str,
        export_dir: str,
        language: str,
        use_hiragana: bool,
        overlap_ratio: float,
        crossfade_ms: int,
        first_naming_rule: str,
        fuzzy_phoneme: bool = False
    ) -> Tuple[List[Dict], set]:
        """
        自动拼字：用已有音素拼接生成缺失的音素组合
        
        参数:
            all_entries: 所有原始 oto 条目（用于提取音素片段）
            filtered_entries: 已筛选的条目（用于确定已有别名）
            slices_dir: 切片目录
            export_dir: 导出目录
            language: 语言
            use_hiragana: 是否使用平假名
            overlap_ratio: overlap 比例
            crossfade_ms: 交叉淡化时长
            first_naming_rule: 首个样本命名规则
            fuzzy_phoneme: 是否启用模糊拼字（仅中文有效）
        
        返回:
            (新生成的条目列表, 新生成的 wav 文件名集合)
        """
        import numpy as np
        import soundfile as sf
        
        # 步骤1: 收集已有别名
        existing_aliases = set()
        for entry in filtered_entries:
            # 提取基础别名（去除序号后缀）
            alias = entry.get("alias", "")
            if alias:
                existing_aliases.add(alias)
        
        self._log(f"已有 {len(existing_aliases)} 个别名")
        
        # 步骤2: 从原始条目中提取最佳辅音和元音片段
        consonant_segments, vowel_segments = self._collect_phoneme_segments(
            all_entries, slices_dir, language
        )
        
        self._log(f"收集到 {len(consonant_segments)} 个辅音, {len(vowel_segments)} 个元音")
        
        if not consonant_segments or not vowel_segments:
            self._log("音素不足，跳过自动拼字")
            return [], set()
        
        # 步骤3: 生成候选组合并过滤
        # 模糊拼字仅对中文生效
        enable_fuzzy = fuzzy_phoneme and language in ('chinese', 'zh', 'mandarin')
        candidates = self._generate_candidates(
            consonant_segments, vowel_segments,
            existing_aliases, language, use_hiragana,
            enable_fuzzy
        )
        
        if not candidates:
            self._log("无缺失的有效组合")
            return [], set()
        
        self._log(f"发现 {len(candidates)} 个缺失组合，开始拼接...")
        
        # 步骤4: 执行音频拼接
        new_entries = []
        new_wavs = set()
        success_count = 0
        fail_count = 0
        
        for candidate in candidates:
            try:
                entry, wav_name = self._combine_and_save(
                    candidate,
                    slices_dir,
                    export_dir,
                    overlap_ratio,
                    crossfade_ms,
                    first_naming_rule
                )
                if entry:
                    new_entries.append(entry)
                    new_wavs.add(wav_name)
                    success_count += 1
            except Exception as e:
                logger.warning(f"拼接失败 {candidate['alias']}: {e}")
                fail_count += 1
        
        if fail_count > 0:
            self._log(f"拼接完成: 成功 {success_count}, 失败 {fail_count}")
        
        return new_entries, new_wavs
    
    def _collect_phoneme_segments(
        self,
        entries: List[Dict],
        slices_dir: str,
        language: str
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        从条目中收集辅音和元音片段信息
        
        返回:
            (辅音字典, 元音字典)
            每个字典: {IPA音素: {wav_path, offset_ms, duration_ms, quality_score}}
        """
        import soundfile as sf
        
        consonant_segments: Dict[str, List[Dict]] = defaultdict(list)
        vowel_segments: Dict[str, List[Dict]] = defaultdict(list)
        
        for entry in entries:
            wav_name = entry.get("wav_name", "")
            wav_path = os.path.join(slices_dir, wav_name)
            
            if not os.path.exists(wav_path):
                continue
            
            # 从条目中提取原始音素信息（如果有）
            # 这里需要重新解析，因为原始条目可能没有保存 IPA 信息
            # 我们使用 alias 反推（简化处理）
            alias = entry.get("alias", "")
            offset = entry.get("offset", 0)
            consonant_dur = entry.get("consonant", 0)
            segment_dur = entry.get("segment_duration", 0)
            quality = entry.get("quality_score", 0.5)
            
            # 尝试分离辅音和元音部分
            c_part, v_part = self._split_alias_to_cv(alias, language)
            
            if c_part:
                consonant_segments[c_part].append({
                    "wav_path": wav_path,
                    "offset_ms": offset,
                    "duration_ms": consonant_dur,
                    "quality_score": quality,
                    "ipa": c_part
                })
            
            if v_part:
                # 元音从辅音结束位置开始
                v_offset = offset + consonant_dur
                v_duration = segment_dur - consonant_dur
                if v_duration > 0:
                    vowel_segments[v_part].append({
                        "wav_path": wav_path,
                        "offset_ms": v_offset,
                        "duration_ms": v_duration,
                        "quality_score": quality,
                        "ipa": v_part
                    })
        
        # 选择最佳音素
        # 辅音：从质量前5中选择时长最接近中位数的（避免过长或过短）
        # 元音：从质量前5中选择时长最长的（避免UTAU过度拉伸）
        best_consonants = {}
        for ipa, segments in consonant_segments.items():
            if segments:
                best_consonants[ipa] = self._select_best_consonant(segments)
        
        best_vowels = {}
        for ipa, segments in vowel_segments.items():
            if segments:
                best_vowels[ipa] = self._select_best_vowel(segments)
        
        return best_consonants, best_vowels
    
    def _select_best_consonant(self, segments: List[Dict]) -> Dict:
        """
        选择最佳辅音片段
        
        策略：从质量排名前5中选择时长最接近中位数的
        （辅音不宜过长也不宜过短）
        """
        # 按质量排序，取前5
        sorted_by_quality = sorted(segments, key=lambda x: -x["quality_score"])
        top_candidates = sorted_by_quality[:5]
        
        if len(top_candidates) == 1:
            return top_candidates[0]
        
        # 计算这些候选的时长中位数
        durations = [s["duration_ms"] for s in top_candidates]
        durations.sort()
        median_duration = durations[len(durations) // 2]
        
        # 选择最接近中位数的
        best = min(top_candidates, key=lambda x: abs(x["duration_ms"] - median_duration))
        return best
    
    def _select_best_vowel(self, segments: List[Dict]) -> Dict:
        """
        选择最佳元音片段
        
        策略：从质量排名前5中选择时长最长的
        （元音过短会导致UTAU过度拉伸）
        """
        # 按质量排序，取前5
        sorted_by_quality = sorted(segments, key=lambda x: -x["quality_score"])
        top_candidates = sorted_by_quality[:5]
        
        # 从中选择时长最长的
        best = max(top_candidates, key=lambda x: x["duration_ms"])
        return best
    
    def _split_alias_to_cv(
        self,
        alias: str,
        language: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        将别名拆分为辅音和元音部分
        
        参数:
            alias: 别名（拼音、罗马音或平假名）
            language: 语言
        
        返回:
            (辅音部分, 元音部分) - 始终返回罗马音格式
        """
        if not alias:
            return None, None
        
        # 如果是平假名，先转换为罗马音
        alias_to_split = self._hiragana_to_romaji(alias)
        if alias_to_split is None:
            alias_to_split = alias.lower()
        
        if language in ('chinese', 'zh', 'mandarin'):
            # 中文拼音辅音列表（按长度降序排列以优先匹配长的）
            consonants = [
                'zh', 'ch', 'sh', 'ng',
                'b', 'p', 'm', 'f',
                'd', 't', 'n', 'l',
                'g', 'k', 'h',
                'j', 'q', 'x',
                'z', 'c', 's', 'r',
                'y', 'w'
            ]
        else:
            # 日语罗马音辅音
            consonants = [
                'ch', 'sh', 'ts', 'ny',
                'ky', 'gy', 'py', 'by', 'my', 'ry', 'hy',
                'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'r', 'w', 'y', 'f', 'j'
            ]
        
        # 尝试匹配辅音
        for c in consonants:
            if alias_to_split.startswith(c):
                vowel = alias_to_split[len(c):]
                if vowel:  # 确保有元音部分
                    return c, vowel
                else:
                    return c, None
        
        # 没有辅音，整个是元音
        return None, alias_to_split
    
    def _hiragana_to_romaji(self, text: str) -> Optional[str]:
        """
        将平假名转换为罗马音
        
        参数:
            text: 平假名文本
        
        返回:
            罗马音，如果无法转换则返回 None
        """
        # 平假名到罗马音映射（ROMAJI_TO_HIRAGANA 的反向映射）
        hiragana_to_romaji_map = {
            # 基本元音
            'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
            # か行
            'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
            # さ行
            'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
            # た行
            'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
            # な行
            'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
            # は行
            'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
            # ま行
            'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
            # や行
            'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
            # ら行
            'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
            # わ行
            'わ': 'wa', 'を': 'wo', 'ん': 'n',
            # が行
            'が': 'ga', 'ぎ': 'gi', 'ぐ': 'gu', 'げ': 'ge', 'ご': 'go',
            # ざ行
            'ざ': 'za', 'じ': 'ji', 'ず': 'zu', 'ぜ': 'ze', 'ぞ': 'zo',
            # だ行
            'だ': 'da', 'ぢ': 'di', 'づ': 'du', 'で': 'de', 'ど': 'do',
            # ば行
            'ば': 'ba', 'び': 'bi', 'ぶ': 'bu', 'べ': 'be', 'ぼ': 'bo',
            # ぱ行
            'ぱ': 'pa', 'ぴ': 'pi', 'ぷ': 'pu', 'ぺ': 'pe', 'ぽ': 'po',
            # 拗音
            'きゃ': 'kya', 'きゅ': 'kyu', 'きょ': 'kyo',
            'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho',
            'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho',
            'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo',
            'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo',
            'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo',
            'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo',
            'ぎゃ': 'gya', 'ぎゅ': 'gyu', 'ぎょ': 'gyo',
            'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo',
            'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo',
            'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo',
        }
        
        # 去除数字后缀
        base_text = text.rstrip('0123456789')
        
        # 直接查找
        if base_text in hiragana_to_romaji_map:
            return hiragana_to_romaji_map[base_text]
        
        # 如果是纯 ASCII，直接返回小写
        if base_text.isascii():
            return base_text.lower()
        
        return None
    
    def _generate_candidates(
        self,
        consonants: Dict[str, Dict],
        vowels: Dict[str, Dict],
        existing_aliases: set,
        language: str,
        use_hiragana: bool,
        fuzzy_phoneme: bool = False
    ) -> List[Dict]:
        """
        生成缺失的候选组合
        
        参数:
            consonants: 可用辅音字典
            vowels: 可用元音字典
            existing_aliases: 已存在的别名集合
            language: 语言
            use_hiragana: 是否使用平假名
            fuzzy_phoneme: 是否启用模糊拼字
        
        返回:
            候选列表，每个候选包含 {alias, consonant_info, vowel_info}
        """
        candidates = []
        
        # 获取有效的元音列表（用于验证组合）
        if language in ('chinese', 'zh', 'mandarin'):
            valid_vowels = {'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng', 'ong', 'er'}
        else:
            valid_vowels = {'a', 'i', 'u', 'e', 'o'}
        
        # 构建可用音素集合（用于模糊匹配）
        available_consonants = set(consonants.keys())
        available_vowels = set(vowels.keys())
        
        # 辅音 + 元音组合
        for c_alias, c_info in consonants.items():
            for v_alias, v_info in vowels.items():
                # 确保辅音和元音都是罗马音格式（小写ASCII）
                c_romaji = c_alias.lower() if c_alias.isascii() else None
                v_romaji = v_alias.lower() if v_alias.isascii() else None
                
                # 跳过非罗马音的音素（如已经是平假名的）
                if c_romaji is None or v_romaji is None:
                    continue
                
                combined_romaji = c_romaji + v_romaji
                
                # 检查组合是否合理（简单验证）
                if v_romaji not in valid_vowels and len(v_romaji) > 2:
                    continue
                
                # 转换为最终别名格式
                if use_hiragana:
                    final_alias = ROMAJI_TO_HIRAGANA.get(combined_romaji)
                    # 如果无法转换为平假名，跳过此组合
                    if final_alias is None:
                        continue
                else:
                    final_alias = combined_romaji
                
                # 检查是否已存在（检查最终别名）
                if final_alias in existing_aliases:
                    continue
                
                # 同时检查罗马音形式是否已存在
                if combined_romaji in existing_aliases:
                    continue
                
                candidates.append({
                    "alias": final_alias,
                    "base_alias": combined_romaji,  # 始终使用罗马音作为基础
                    "consonant_info": c_info,
                    "vowel_info": v_info
                })
        
        # 模糊拼字：生成使用近似音素的额外候选
        if fuzzy_phoneme and language in ('chinese', 'zh', 'mandarin'):
            fuzzy_candidates = self._generate_fuzzy_candidates(
                consonants, vowels,
                available_consonants, available_vowels,
                existing_aliases, candidates
            )
            candidates.extend(fuzzy_candidates)
        
        return candidates
    
    def _find_fuzzy_substitute(
        self,
        phoneme: str,
        available_phonemes: set,
        groups: List[Tuple[str, ...]]
    ) -> Optional[str]:
        """
        查找模糊替代音素
        
        参数:
            phoneme: 目标音素
            available_phonemes: 可用音素集合
            groups: 近似音素组列表（同组内音素互为替代）
        
        返回:
            替代音素，如果无法替代则返回 None
        """
        # 如果目标音素已存在，直接返回
        if phoneme in available_phonemes:
            return phoneme
        
        # 查找目标音素所在的近似组
        for group in groups:
            if phoneme in group:
                # 按组内顺序查找可用的替代音素
                for candidate in group:
                    if candidate != phoneme and candidate in available_phonemes:
                        return candidate
                # 该组内没有可用替代
                break
        
        return None
    
    def _generate_fuzzy_candidates(
        self,
        consonants: Dict[str, Dict],
        vowels: Dict[str, Dict],
        available_consonants: set,
        available_vowels: set,
        existing_aliases: set,
        normal_candidates: List[Dict]
    ) -> List[Dict]:
        """
        生成模糊拼字候选
        
        使用近似音素替代缺失的声母/韵母，生成额外的候选组合
        """
        fuzzy_candidates = []
        
        # 已生成的别名（包括普通候选）
        generated_aliases = set(c["base_alias"] for c in normal_candidates)
        generated_aliases.update(existing_aliases)
        
        # 中文所有可能的声母
        all_consonants = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
                          'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']
        
        # 中文所有可能的韵母
        all_vowels = ['a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ao', 'ou',
                      'an', 'en', 'ang', 'eng', 'ong', 'in', 'ing', 'ian', 'iang',
                      'uan', 'uang', 'un', 'ia', 'ie', 'iu', 'iao', 'ua', 'uo', 'ui', 'uai']
        
        fuzzy_count = 0
        
        for target_c in all_consonants:
            for target_v in all_vowels:
                target_alias = target_c + target_v
                
                # 跳过已存在或已生成的
                if target_alias in generated_aliases:
                    continue
                
                # 确定实际使用的辅音
                if target_c in available_consonants:
                    actual_c = target_c
                else:
                    actual_c = self._find_fuzzy_substitute(
                        target_c, available_consonants, FUZZY_CONSONANT_GROUPS
                    )
                
                # 确定实际使用的元音
                if target_v in available_vowels:
                    actual_v = target_v
                else:
                    actual_v = self._find_fuzzy_substitute(
                        target_v, available_vowels, FUZZY_VOWEL_GROUPS
                    )
                
                # 如果辅音或元音无法获取，跳过
                if actual_c is None or actual_v is None:
                    continue
                
                # 如果实际音素与目标相同，说明不需要模糊替换（普通候选已处理）
                if actual_c == target_c and actual_v == target_v:
                    continue
                
                # 获取音素信息
                c_info = consonants.get(actual_c)
                v_info = vowels.get(actual_v)
                
                if c_info is None or v_info is None:
                    continue
                
                fuzzy_candidates.append({
                    "alias": target_alias,
                    "base_alias": target_alias,
                    "consonant_info": c_info,
                    "vowel_info": v_info,
                    "is_fuzzy": True,
                    "fuzzy_from": f"{actual_c}+{actual_v}"
                })
                generated_aliases.add(target_alias)
                fuzzy_count += 1
        
        if fuzzy_count > 0:
            self._log(f"模糊拼字生成 {fuzzy_count} 个额外候选")
        
        return fuzzy_candidates
    
    def _combine_and_save(
        self,
        candidate: Dict,
        slices_dir: str,
        export_dir: str,
        overlap_ratio: float,
        crossfade_ms: int,
        first_naming_rule: str
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        执行音频拼接并保存
        
        参数:
            candidate: 候选信息
            slices_dir: 切片目录
            export_dir: 导出目录
            overlap_ratio: overlap 比例
            crossfade_ms: 交叉淡化时长
            first_naming_rule: 命名规则
        
        返回:
            (oto条目, wav文件名) 或 (None, None)
        """
        import numpy as np
        import soundfile as sf
        
        c_info = candidate["consonant_info"]
        v_info = candidate["vowel_info"]
        alias = candidate["alias"]
        
        # 加载辅音片段
        c_audio, c_sr = sf.read(c_info["wav_path"])
        if len(c_audio.shape) > 1:
            c_audio = c_audio.mean(axis=1)
        
        c_start = int(c_info["offset_ms"] / 1000 * c_sr)
        c_duration = int(c_info["duration_ms"] / 1000 * c_sr)
        c_segment = c_audio[c_start:c_start + c_duration]
        
        # 加载元音片段
        v_audio, v_sr = sf.read(v_info["wav_path"])
        if len(v_audio.shape) > 1:
            v_audio = v_audio.mean(axis=1)
        
        v_start = int(v_info["offset_ms"] / 1000 * v_sr)
        v_duration = int(v_info["duration_ms"] / 1000 * v_sr)
        v_segment = v_audio[v_start:v_start + v_duration]
        
        # 确保采样率一致
        if c_sr != v_sr:
            logger.warning(f"采样率不一致: {c_sr} vs {v_sr}，跳过")
            return None, None
        
        sr = c_sr
        
        # 检查片段有效性
        if len(c_segment) == 0 or len(v_segment) == 0:
            return None, None
        
        # 执行交叉淡化拼接
        crossfade_samples = int(crossfade_ms / 1000 * sr)
        crossfade_samples = min(crossfade_samples, len(c_segment) // 2, len(v_segment) // 2)
        
        if crossfade_samples < 1:
            crossfade_samples = 1
        
        combined = self._crossfade_concat(c_segment, v_segment, crossfade_samples)
        
        # 生成文件名（使用 C 前缀表示 Combined）
        wav_name = f"C{candidate['alias']}.wav"
        wav_path = os.path.join(export_dir, wav_name)
        
        # 保存音频
        sf.write(wav_path, combined, sr)
        
        # 计算 oto 参数
        c_duration_ms = c_info["duration_ms"]
        total_duration_ms = len(combined) / sr * 1000
        
        # 应用命名规则（作为首个样本）
        final_alias = self.apply_naming_rule(first_naming_rule, alias, 0) if first_naming_rule else alias
        
        entry = {
            "wav_name": wav_name,
            "alias": final_alias,
            "offset": 0,
            "consonant": round(c_duration_ms, 1),
            "cutoff": round(-total_duration_ms, 1),
            "preutterance": round(c_duration_ms, 1),
            "overlap": round(c_duration_ms * overlap_ratio, 1),
            "segment_duration": total_duration_ms,
            "is_combined": True  # 标记为拼接生成
        }
        
        return entry, wav_name
    
    def _crossfade_concat(
        self,
        audio1: 'np.ndarray',
        audio2: 'np.ndarray',
        crossfade_samples: int
    ) -> 'np.ndarray':
        """
        交叉淡化拼接两段音频
        
        参数:
            audio1: 第一段音频
            audio2: 第二段音频
            crossfade_samples: 交叉淡化采样数
        
        返回:
            拼接后的音频
        """
        import numpy as np
        
        if crossfade_samples <= 0:
            return np.concatenate([audio1, audio2])
        
        # 确保交叉淡化长度不超过音频长度
        crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))
        
        # 创建淡入淡出曲线
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        
        # 分离各部分
        part1 = audio1[:-crossfade_samples]
        overlap1 = audio1[-crossfade_samples:]
        overlap2 = audio2[:crossfade_samples]
        part2 = audio2[crossfade_samples:]
        
        # 交叉混合
        crossfaded = overlap1 * fade_out + overlap2 * fade_in
        
        # 拼接
        return np.concatenate([part1, crossfaded, part2])
