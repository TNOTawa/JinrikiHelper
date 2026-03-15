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
    # 双唇音
    'p', 'pʰ', 'pʲ', 'pʷ', 'b', 'm', 'f',
    # 齿龈音
    't', 'tʰ', 'tʲ', 'd', 'n', 'l',
    # 软腭音
    'k', 'kʰ', 'kʷ', 'ɡ', 'g', 'ŋ', 'x', 'h',
    # 齿龈-硬腭音（j, q, x）
    'tɕ', 'tɕʰ', 'dʑ', 'ɕ', 'ʑ',
    # 齿龈塞擦音（z, c, s）
    'ts', 'tsʰ', 'dz', 's', 'z',
    # 卷舌音（zh, ch, sh, r）
    'ʈʂ', 'ʈʂʰ', 'ɖʐ', 'ʂ', 'ʐ',
    # 鼻音和近音
    'ɲ', 'j', 'w', 'ɥ',
    # 喉塞音
    'ʔ',
}

# 中文元音（可能带声调标记）
# 注意：MFA 输出的元音通常是单个音素，复合韵母会被拆分成多个音素
CHINESE_VOWELS = {
    # 基本单元音
    'a', 'o', 'e', 'i', 'u', 'y', 'ü',
    'ə', 'ɛ', 'ɔ', 'ɤ', 'ɨ', 'ʅ', 'ʉ',
    # MFA 输出的特殊格式
    'aw', 'ej', 'ow',  # 双元音的 MFA 表示（ai, ei, ou）
    # 舌尖元音（zi, ci, si, zhi, chi, shi, ri）
    'z̩', 'ʐ̩',
    # 卷舌近音（er）
    'ɻ',
    # 儿化音
    'ɚ',
}

# 中文介音（声母和韵母之间的过渡音）
CHINESE_MEDIALS = {
    'j', 'w', 'ɥ',  # i, u, ü 介音
}

# 中文韵尾（鼻音和元音韵尾）
CHINESE_CODAS = {
    'n', 'ŋ',  # 鼻音韵尾
    'i', 'u',  # 元音韵尾（在复韵母中）
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
    # i 行韵母近似组（带鼻音韵尾的可以用不带鼻音韵尾的替代）
    ('ia', 'ian'),       # ia ←→ ian（如 xia ←→ xian）
    ('ie', 'ian'),       # ie ←→ ian（如 jie ←→ jian）
    ('iao', 'ian'),      # iao ←→ ian（如 qiao ←→ qian）
    ('iu', 'in'),        # iu ←→ in（如 liu ←→ lin）
    # u 行韵母近似组
    ('ua', 'uan'),       # ua ←→ uan（如 kua ←→ kuan）
    ('uo', 'un'),        # uo ←→ un（如 duo ←→ dun）
    ('ui', 'un'),        # ui ←→ un（如 dui ←→ dun）
    ('uai', 'uan'),      # uai ←→ uan（如 kuai ←→ kuan）
    # 单元音与复韵母近似组
    ('a', 'ai', 'ao', 'an'),  # a 系列
    ('o', 'ou', 'ong'),       # o 系列
    ('e', 'ei', 'en'),        # e 系列
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
        # 直接匹配
        if base_phone in CHINESE_VOWELS:
            return True
        
        # 检查是否以元音字符开头（处理复合元音）
        vowel_starts = ['a', 'o', 'e', 'i', 'u', 'y', 'ə', 'ɛ', 'ɔ', 'ɤ', 'ɨ', 'ʅ', 'ʉ', 'ɚ']
        for v in vowel_starts:
            if base_phone.startswith(v):
                return True
        
        # 检查特殊的舌尖元音（带组合字符）
        if 'z̩' in base_phone or 'ʐ̩' in base_phone:
            return True
        
        # 检查卷舌近音
        if 'ɻ' in base_phone:
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

# 中文 IPA 辅音到拼音声母映射
CHINESE_CONSONANT_TO_PINYIN = {
    'p': 'b', 'pʰ': 'p', 'pʲ': 'p', 'pʷ': 'b',
    'm': 'm', 'f': 'f',
    't': 'd', 'tʰ': 't', 'tʲ': 'd',
    'n': 'n', 'l': 'l',
    'k': 'g', 'kʰ': 'k', 'kʷ': 'g',
    'ɡ': 'g', 'g': 'g',
    'x': 'h', 'h': 'h',
    'tɕ': 'j', 'tɕʰ': 'q', 'ɕ': 'x',
    'ts': 'z', 'tsʰ': 'c', 's': 's',
    'ʈʂ': 'zh', 'ʈʂʰ': 'ch', 'ʂ': 'sh', 'ʐ': 'r',
    'ɲ': 'n', 'ŋ': '',  # ng 不作为声母
    'j': '', 'w': '', 'ɥ': '',  # 介音不作为声母
    'ʔ': '',
}

# 中文 IPA 元音到拼音韵母映射
CHINESE_VOWEL_TO_PINYIN = {
    # 单元音韵母
    'a': 'a', 'o': 'o', 'e': 'e', 'i': 'i', 'u': 'u', 'y': 'v', 'ü': 'v',
    'ə': 'e', 'ɛ': 'e', 'ɔ': 'o', 'ɤ': 'e', 'ɨ': 'i',
    # 复韵母（MFA 可能的 IPA 格式）
    'aj': 'ai', 'aw': 'ao', 'ej': 'ei', 'ow': 'ou',
    'ai': 'ai', 'ao': 'ao', 'ei': 'ei', 'ou': 'ou',  # 直接形式
    # i 行韵母（MFA 可能的组合形式）
    'ja': 'ia', 'je': 'ie', 'jɛ': 'ie', 'jao': 'iao', 'jow': 'iu', 'ju': 'iu',
    'ia': 'ia', 'ie': 'ie', 'iao': 'iao', 'iu': 'iu',  # 直接形式
    # u 行韵母（MFA 可能的组合形式）
    'wa': 'ua', 'wo': 'uo', 'wɔ': 'uo', 'wej': 'ui', 'waj': 'uai',
    'ua': 'ua', 'uo': 'uo', 'ui': 'ui', 'uai': 'uai',  # 直接形式
    # ü 行韵母（MFA 可能的组合形式）
    'ɥe': 've', 'ɥɛ': 've',
    've': 've', 'yue': 've',  # 直接形式
    # 鼻音韵母（MFA 可能的组合形式）
    'an': 'an', 'en': 'en', 'ang': 'ang', 'eng': 'eng', 'ong': 'ong',
    'in': 'in', 'ing': 'ing', 'ian': 'ian', 'iang': 'iang', 'iong': 'iong',
    'uan': 'uan', 'un': 'un', 'uang': 'uang', 'ueng': 'ueng',
    'van': 'van', 'vn': 'vn',
    # 舌尖元音
    'z̩': 'i', 'ʐ̩': 'i', 'ʅ': 'i',
    # 卷舌音
    'ɻ': 'er', 'ɚ': 'er',
}

# 介音+元音组合到韵母的映射
MEDIAL_VOWEL_TO_FINAL = {
    # j 介音（i 行韵母）
    ('j', 'a'): 'ia', ('j', 'e'): 'ie', ('j', 'ɛ'): 'ie',
    ('j', 'aw'): 'iao', ('j', 'o'): 'io',
    ('j', 'u'): 'iu', ('j', 'ow'): 'iou',
    # w 介音（u 行韵母）
    ('w', 'a'): 'ua', ('w', 'o'): 'uo', ('w', 'ɔ'): 'uo',
    ('w', 'ej'): 'uei', ('w', 'e'): 'ue',
    ('w', 'aj'): 'uai', ('w', 'ai'): 'uai',
    # ɥ 介音（ü 行韵母）
    ('ɥ', 'e'): 've', ('ɥ', 'ɛ'): 've',
}

# 介音+元音+韵尾组合到韵母的映射
MEDIAL_VOWEL_CODA_TO_FINAL = {
    # j 介音 + 元音 + 韵尾
    ('j', 'a', 'n'): 'ian', ('j', 'e', 'n'): 'in',
    ('j', 'a', 'ŋ'): 'iang', ('j', 'o', 'ŋ'): 'iong',
    # w 介音 + 元音 + 韵尾
    ('w', 'a', 'n'): 'uan', ('w', 'ə', 'n'): 'uen', ('w', 'e', 'n'): 'uen',
    ('w', 'a', 'ŋ'): 'uang', ('w', 'ə', 'ŋ'): 'ueng', ('w', 'e', 'ŋ'): 'ueng',
    # ɥ 介音 + 元音 + 韵尾
    ('ɥ', 'a', 'n'): 'van', ('ɥ', 'e', 'n'): 'vn',
}

# 元音+韵尾组合到拼音韵母的映射
VOWEL_CODA_TO_PINYIN = {
    # 前鼻音韵母
    ('a', 'n'): 'an', ('ə', 'n'): 'en', ('e', 'n'): 'en',
    ('i', 'n'): 'in', ('y', 'n'): 'un', ('u', 'n'): 'un',
    # 后鼻音韵母
    ('a', 'ŋ'): 'ang', ('ə', 'ŋ'): 'eng', ('e', 'ŋ'): 'eng',
    ('i', 'ŋ'): 'ing', ('o', 'ŋ'): 'ong', ('u', 'ŋ'): 'ong',
    # 复韵母（元音+元音）
    ('a', 'i'): 'ai', ('e', 'i'): 'ei', ('ej', 'i'): 'ei',
    ('a', 'u'): 'ao', ('aw', 'u'): 'ao', ('o', 'u'): 'ou', ('ow', 'u'): 'ou',
    # i 行韵母
    ('i', 'a'): 'ia', ('i', 'e'): 'ie', ('i', 'ɛ'): 'ie',
    ('i', 'u'): 'iu',
    # u 行韵母
    ('u', 'a'): 'ua', ('u', 'o'): 'uo', ('u', 'ɔ'): 'uo',
    ('u', 'i'): 'ui', ('u', 'e'): 'ue',
    # ü 行韵母
    ('y', 'e'): 've', ('y', 'ɛ'): 've',
}

# IPA 音节组合到标准拼音的映射表（处理特殊组合规则）
IPA_SYLLABLE_TO_PINYIN = {
    # j/q/x + ü 系列（ü 简写为 u）
    ('tɕ', 'y'): 'ju', ('tɕʰ', 'y'): 'qu', ('ɕ', 'y'): 'xu',
    ('tɕ', 'ɥ'): 'ju', ('tɕʰ', 'ɥ'): 'qu', ('ɕ', 'ɥ'): 'xu',
    ('tɕ', 'yɛ'): 'jue', ('tɕʰ', 'yɛ'): 'que', ('ɕ', 'yɛ'): 'xue',
    ('tɕ', 'yan'): 'juan', ('tɕʰ', 'yan'): 'quan', ('ɕ', 'yan'): 'xuan',
    ('tɕ', 'yn'): 'jun', ('tɕʰ', 'yn'): 'qun', ('ɕ', 'yn'): 'xun',
    
    # 零声母 + i/u/ü 开头的韵母（需要加 y/w）
    ('', 'i'): 'yi', ('', 'in'): 'yin', ('', 'ing'): 'ying',
    ('', 'u'): 'wu', ('', 'un'): 'wen', ('', 'ong'): 'weng',
    ('', 'y'): 'yu', ('', 'yn'): 'yun',
    
    # i 行韵母（ia, ie, iao, ian, iang, iong, iu）
    ('', 'ia'): 'ya', ('', 'iɛ'): 'ye', ('', 'ie'): 'ye',
    ('', 'iao'): 'yao', ('', 'ian'): 'yan', ('', 'iang'): 'yang',
    ('', 'iou'): 'you', ('', 'iu'): 'you',
    ('', 'iong'): 'yong',
    
    # u 行韵母（ua, uo, uai, uei, uan, uen, uang, ueng）
    ('', 'ua'): 'wa', ('', 'uɔ'): 'wo', ('', 'uo'): 'wo',
    ('', 'uai'): 'wai', ('', 'uei'): 'wei', ('', 'ui'): 'wei',
    ('', 'uan'): 'wan', ('', 'uen'): 'wen',
    ('', 'uang'): 'wang', ('', 'ueng'): 'weng',
    
    # ü 行韵母（üe, üan, ün）
    ('', 'yɛ'): 'yue', ('', 'üe'): 'yue',
    ('', 'yan'): 'yuan', ('', 'üan'): 'yuan',
    ('', 'yn'): 'yun', ('', 'ün'): 'yun',
    
    # zh/ch/sh/r + i 实际是舌尖元音
    ('ʈʂ', 'ʐ̩'): 'zhi', ('ʈʂʰ', 'ʐ̩'): 'chi', ('ʂ', 'ʐ̩'): 'shi', ('ʐ', 'ʐ̩'): 'ri',
    ('ʈʂ', 'z̩'): 'zhi', ('ʈʂʰ', 'z̩'): 'chi', ('ʂ', 'z̩'): 'shi', ('ʐ', 'z̩'): 'ri',
    ('ʈʂ', 'ʅ'): 'zhi', ('ʈʂʰ', 'ʅ'): 'chi', ('ʂ', 'ʅ'): 'shi', ('ʐ', 'ʅ'): 'ri',
    
    # z/c/s + i 实际是舌尖元音
    ('ts', 'z̩'): 'zi', ('tsʰ', 'z̩'): 'ci', ('s', 'z̩'): 'si',
    ('ts', 'ʅ'): 'zi', ('tsʰ', 'ʅ'): 'ci', ('s', 'ʅ'): 'si',
    
    # n/l + ü 系列（保持 ü）
    ('n', 'y'): 'nv', ('l', 'y'): 'lv',
    ('n', 'yɛ'): 'nve', ('l', 'yɛ'): 'lve',
    
    # 其他特殊组合
    ('ʔ', 'a'): 'a', ('ʔ', 'o'): 'o', ('ʔ', 'e'): 'e',
    ('ʔ', 'ai'): 'ai', ('ʔ', 'ei'): 'ei', ('ʔ', 'ao'): 'ao', ('ʔ', 'ou'): 'ou',
    ('ʔ', 'an'): 'an', ('ʔ', 'en'): 'en', ('ʔ', 'ang'): 'ang', ('ʔ', 'eng'): 'eng',
    ('ʔ', 'ej'): 'ei', ('ʔ', 'aw'): 'ao', ('ʔ', 'ow'): 'ou',
    
    # 儿化音
    ('', 'ɻ'): 'er', ('', 'ɚ'): 'er',
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
    """将 IPA 音素转换为别名（标准拼音或罗马音）"""
    c_base = _strip_tone(consonant) if consonant else ''
    v_base = _strip_tone(vowel) if vowel else ''
    
    if language in ('chinese', 'zh', 'mandarin'):
        # 中文：使用完整的音节转换规则
        return _ipa_to_pinyin(c_base, v_base)
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


def _ipa_to_pinyin(consonant: str, vowel: str) -> Optional[str]:
    """
    将 IPA 辅音+韵母转换为标准汉语拼音
    
    参数:
        consonant: IPA 辅音（已去除声调），可以是空字符串表示零声母
        vowel: IPA 韵母（已去除声调），可能是单个元音或元音+韵尾的组合
    
    返回:
        标准拼音，如果无法转换则返回 None
    """
    # 1. 先查找特殊组合映射
    syllable_key = (consonant, vowel)
    if syllable_key in IPA_SYLLABLE_TO_PINYIN:
        return IPA_SYLLABLE_TO_PINYIN[syllable_key]
    
    # 2. 获取声母的拼音
    c_pinyin = ''
    if consonant and consonant != 'ʔ':
        if consonant in CHINESE_CONSONANT_TO_PINYIN:
            c_pinyin = CHINESE_CONSONANT_TO_PINYIN[consonant]
        else:
            # 未知辅音，无法转换
            return None
    
    # 3. 获取韵母的拼音
    # 韵母可能是单个元音，也可能是元音+韵尾的组合字符串
    v_pinyin = ''
    if vowel:
        # 直接查找完整韵母
        if vowel in CHINESE_VOWEL_TO_PINYIN:
            v_pinyin = CHINESE_VOWEL_TO_PINYIN[vowel]
        else:
            # 韵母可能是组合形式，无法直接映射
            # 这种情况应该在 _syllable_to_pinyin 中处理
            return None
    
    if not v_pinyin:
        return None
    
    # 4. 处理零声母（无声母或喉塞音）
    if not c_pinyin:
        # 零声母需要根据韵母添加 y/w/yu
        if v_pinyin == 'i':
            return 'yi'
        elif v_pinyin in ('in', 'ing'):
            return 'y' + v_pinyin
        elif v_pinyin.startswith('i') and len(v_pinyin) > 1:
            # ia->ya, ie->ye, iao->yao, ian->yan, iang->yang, iu->you, iong->yong
            return 'y' + v_pinyin[1:]
        elif v_pinyin == 'u':
            return 'wu'
        elif v_pinyin == 'un':
            return 'wen'
        elif v_pinyin == 'ong':
            return 'weng'
        elif v_pinyin.startswith('u') and len(v_pinyin) > 1:
            # ua->wa, uo->wo, uai->wai, ui->wei, uan->wan, uang->wang
            return 'w' + v_pinyin[1:]
        elif v_pinyin == 'v':
            # ü 单独出现写作 yu
            return 'yu'
        elif v_pinyin.startswith('v') and len(v_pinyin) > 1:
            # ve->yue, van->yuan, vn->yun
            return 'yu' + v_pinyin[1:]
        else:
            # a, o, e, ai, ei, ao, ou, an, en, ang, eng, er 等
            return v_pinyin
    
    # 5. 有声母的情况
    # 5.1 j/q/x + ü 系列：ü 写作 u
    if c_pinyin in ('j', 'q', 'x'):
        if v_pinyin == 'v':
            return c_pinyin + 'u'
        elif v_pinyin.startswith('v'):
            # jve->jue, jvan->juan, jvn->jun
            return c_pinyin + 'u' + v_pinyin[1:]
        else:
            return c_pinyin + v_pinyin
    
    # 5.2 n/l + ü 系列：保持 v（表示 ü）
    elif c_pinyin in ('n', 'l'):
        # 只有 n/l 才需要区分 u 和 ü
        return c_pinyin + v_pinyin
    
    # 5.3 其他声母 + v：v 改写为 u（因为不会产生歧义）
    elif v_pinyin == 'v':
        return c_pinyin + 'u'
    elif v_pinyin.startswith('v'):
        return c_pinyin + 'u' + v_pinyin[1:]
    
    # 5.4 普通组合
    else:
        return c_pinyin + v_pinyin


class UTAUOtoExportPlugin(ExportPlugin):
    """UTAU oto.ini 导出插件"""
    
    name = "UTAU oto.ini 导出"
    description = "从 TextGrid 生成 UTAU 音源配置文件，一个 wav 可包含多条配置"
    version = "1.2.0"
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
            PluginOption(
                key="cvvc_mode",
                label="CVVC 模式",
                option_type=OptionType.SWITCH,
                default=False,
                description="启用 CVVC 模式，额外生成 VC 部（元音到辅音过渡）条目"
            ),
            PluginOption(
                key="vc_alias_separator",
                label="VC 别名分隔符",
                option_type=OptionType.COMBO,
                default=" ",
                choices=[" ", "_", "-"],
                description="VC 部别名中元音和辅音之间的分隔符",
                visible_when={"cvvc_mode": True}
            ),
            PluginOption(
                key="vc_offset_ratio",
                label="VC 偏移比例",
                option_type=OptionType.NUMBER,
                default=0.5,
                min_value=0.3,
                max_value=0.8,
                description="VC 部开始位置 = 元音结束位置 - 元音时长 × 此比例",
                visible_when={"cvvc_mode": True}
            ),
            PluginOption(
                key="vc_overlap_ratio",
                label="VC Overlap 比例",
                option_type=OptionType.NUMBER,
                default=0.5,
                min_value=0.3,
                max_value=0.8,
                description="VC 部的 Overlap = Preutterance × 此比例",
                visible_when={"cvvc_mode": True}
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
            
            # CVVC 模式选项
            cvvc_mode = options.get("cvvc_mode", False)
            vc_separator = options.get("vc_alias_separator", " ")
            vc_offset_ratio = float(options.get("vc_offset_ratio", 0.5))
            vc_overlap_ratio = float(options.get("vc_overlap_ratio", 0.5))
            
            # 使用基类方法解析质量评估维度
            enabled_metrics = self.parse_quality_metrics(quality_metrics)
            
            paths = self.get_source_paths(bank_dir, source_name)
            export_dir = self.get_export_dir(bank_dir, source_name, "utau_oto")
            
            os.makedirs(export_dir, exist_ok=True)
            
            # 步骤1: 解析 TextGrid 并生成 oto 条目
            if cvvc_mode:
                self._log("【解析 TextGrid 文件】（CVVC 模式）")
            else:
                self._log("【解析 TextGrid 文件】")
            oto_entries, wav_files = self._parse_textgrids(
                paths["slices_dir"],
                paths["textgrid_dir"],
                language,
                use_hiragana,
                overlap_ratio,
                cvvc_mode=cvvc_mode,
                vc_offset_ratio=vc_offset_ratio,
                vc_overlap_ratio=vc_overlap_ratio,
                vc_separator=vc_separator
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
        overlap_ratio: float,
        cvvc_mode: bool = False,
        vc_offset_ratio: float = 0.5,
        vc_overlap_ratio: float = 0.5,
        vc_separator: str = " "
    ) -> Tuple[List[Dict], set]:
        """解析 TextGrid 文件，提取音素边界
        
        参数:
            slices_dir: 切片目录
            textgrid_dir: TextGrid 目录
            language: 语言
            use_hiragana: 是否使用平假名
            overlap_ratio: CV 部 overlap 比例
            cvvc_mode: 是否启用 CVVC 模式
            vc_offset_ratio: VC 偏移比例
            vc_overlap_ratio: VC overlap 比例
            vc_separator: VC 别名分隔符
        """
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
            
            # 提取 CV 对，使用 words 层限制配对范围
            entries = self._extract_cv_pairs(
                words_tier, phones_tier, wav_name, wav_duration_ms,
                language, use_hiragana, overlap_ratio
            )
            oto_entries.extend(entries)
            
            # 如果启用 CVVC 模式，额外提取 VC 对
            if cvvc_mode:
                vc_entries = self._extract_vc_pairs(
                    words_tier, phones_tier, wav_name, wav_duration_ms,
                    language, use_hiragana,
                    vc_offset_ratio, vc_overlap_ratio, vc_separator
                )
                oto_entries.extend(vc_entries)
        
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
        从 phones 层提取音节（可能包含辅音+元音+韵尾）
        使用 words 层限制配对范围，确保音素属于同一个字
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
            
            # 中文音节结构：(辅音) + (介音) + 元音 + (韵尾)
            if language in ('chinese', 'zh', 'mandarin'):
                syllable_phones = []
                syllable_start = start_ms
                syllable_end = end_ms
                consonant_duration = 0
                
                # 1. 检查是否有声母（辅音）
                if is_consonant(phone, language):
                    syllable_phones.append(phone)
                    consonant_duration = end_ms - start_ms
                    i += 1
                    
                    # 检查下一个音素
                    if i < len(intervals):
                        next_interval = intervals[i]
                        next_phone = next_interval.mark.strip()
                        
                        if next_phone not in SKIP_MARKS and same_word(interval.minTime, next_interval.minTime):
                            phone = next_phone
                            end_ms = next_interval.maxTime * 1000
                            syllable_end = end_ms
                        else:
                            # 只有辅音，没有元音，跳过
                            continue
                    else:
                        # 只有辅音，没有元音，跳过
                        continue
                
                # 2. 检查是否有介音（j, w, ɥ）
                phone_base = _strip_tone(phone)
                if phone_base in CHINESE_MEDIALS:
                    syllable_phones.append(phone)
                    i += 1
                    
                    # 检查下一个音素（必须是元音）
                    if i < len(intervals):
                        next_interval = intervals[i]
                        next_phone = next_interval.mark.strip()
                        
                        if next_phone not in SKIP_MARKS and same_word(interval.minTime, next_interval.minTime):
                            phone = next_phone
                            end_ms = next_interval.maxTime * 1000
                            syllable_end = end_ms
                        else:
                            # 只有介音，没有元音，跳过
                            continue
                    else:
                        # 只有介音，没有元音，跳过
                        continue
                
                # 3. 必须有韵母（元音）
                if is_vowel(phone, language):
                    syllable_phones.append(phone)
                    if not consonant_duration:
                        # 零声母，辅音时长设为元音前30ms
                        consonant_duration = min(30, (end_ms - start_ms) * 0.2)
                    syllable_end = end_ms
                    i += 1
                    
                    # 4. 检查是否有韵尾（n, ng, i, u）
                    if i < len(intervals):
                        next_interval = intervals[i]
                        next_phone = next_interval.mark.strip()
                        
                        if (next_phone not in SKIP_MARKS and
                            same_word(interval.minTime, next_interval.minTime)):
                            # 检查是否是韵尾
                            next_phone_base = _strip_tone(next_phone)
                            if next_phone_base in CHINESE_CODAS:
                                syllable_phones.append(next_phone)
                                syllable_end = next_interval.maxTime * 1000
                                i += 1
                    
                    # 5. 将音节转换为拼音
                    alias = self._syllable_to_pinyin(syllable_phones, language, use_hiragana)
                    if alias:
                        entry = self._calculate_oto_params(
                            wav_name=wav_name,
                            alias=alias,
                            offset=syllable_start,
                            consonant_duration=consonant_duration,
                            segment_end=syllable_end,
                            wav_duration_ms=wav_duration_ms,
                            overlap_ratio=overlap_ratio
                        )
                        entries.append(entry)
                else:
                    # 不是元音，跳过
                    i += 1
            
            else:
                # 日语：简单的 CV 结构
                if is_consonant(phone, language):
                    consonant = phone
                    consonant_start = start_ms
                    consonant_end = end_ms
                    consonant_time = interval.minTime
                    
                    vowel = None
                    vowel_end = end_ms
                    
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
                    if alias:
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
                    if alias:
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
    
    def _syllable_to_pinyin(
        self,
        phones: List[str],
        language: str,
        use_hiragana: bool
    ) -> Optional[str]:
        """
        将音素列表转换为标准汉语拼音（通用方法）
        
        采用新的通用转换算法，支持所有标准汉语拼音音节
        
        参数:
            phones: 音素列表（带声调的 IPA 符号）
            language: 语言
            use_hiragana: 是否使用平假名（中文忽略此参数）
        
        返回:
            拼音字符串
        """
        if not phones:
            return None
        
        # 去除声调
        phones_base = [_strip_tone(p) for p in phones]
        
        # 解析音节结构：(辅音) + (介音) + 元音 + (韵尾)
        idx = 0
        c = ''  # 声母
        m = ''  # 介音
        v = ''  # 元音
        cd = ''  # 韵尾
        
        # 1. 声母
        if idx < len(phones_base) and is_consonant(phones_base[idx], language):
            c = phones_base[idx]
            idx += 1
        
        # 2. 介音
        if idx < len(phones_base) and phones_base[idx] in CHINESE_MEDIALS:
            m = phones_base[idx]
            idx += 1
        
        # 3. 元音（必须）
        if idx < len(phones_base) and is_vowel(phones_base[idx], language):
            v = phones_base[idx]
            idx += 1
        else:
            # 没有元音，无法形成音节
            return None
        
        # 4. 韵尾
        if idx < len(phones_base) and phones_base[idx] in CHINESE_CODAS:
            cd = phones_base[idx]
            idx += 1
        
        # 转换为拼音
        c_py = CHINESE_CONSONANT_TO_PINYIN.get(c, '')
        v_py = CHINESE_VOWEL_TO_PINYIN.get(v, v)
        
        # 组合韵母
        final = ''
        
        if m == 'j':
            # i 行韵母
            if cd == 'n':
                if v_py == 'a':
                    final = 'ian'
                elif v_py == 'e':
                    final = 'in'  # j + e + n = in (如 xin, yin)
                else:
                    final = 'i' + v_py + 'n'
            elif cd == 'ŋ':
                if v_py == 'a':
                    final = 'iang'
                elif v_py == 'o':
                    final = 'iong'
                else:
                    final = 'i' + v_py + 'ng'
            elif cd:
                final = 'i' + v_py + cd
            else:
                if v_py == 'a':
                    final = 'ia'
                elif v_py == 'e':
                    final = 'ie'
                elif v_py == 'ao':
                    final = 'iao'
                elif v_py == 'ou':
                    final = 'iu'
                else:
                    final = 'i' + v_py
        
        elif m == 'w':
            # u 行韵母
            if cd == 'n':
                if v_py == 'a':
                    final = 'uan'
                elif v_py == 'e':
                    final = 'un'  # w + ə + n = un (如 shun)
                else:
                    final = 'u' + v_py + 'n'
            elif cd == 'ŋ':
                if v_py == 'a':
                    final = 'uang'
                elif v_py == 'e':
                    final = 'ueng'
                else:
                    final = 'u' + v_py + 'ng'
            elif cd:
                final = 'u' + v_py + cd
            else:
                if v_py == 'a':
                    final = 'ua'
                elif v_py == 'o':
                    final = 'uo'
                elif v_py == 'ei':
                    final = 'ui'  # w + ej = ui (如 shui)
                elif v_py == 'ai':
                    final = 'uai'
                else:
                    final = 'u' + v_py
        
        elif m == 'ɥ':
            # ü 行韵母
            if cd == 'n':
                if v_py == 'a':
                    final = 'van'
                elif v_py == 'e':
                    final = 'vn'
                else:
                    final = 'v' + v_py + 'n'
            elif cd:
                final = 'v' + v_py + cd
            else:
                if v_py == 'e':
                    final = 've'
                else:
                    final = 'v' + v_py
        
        else:
            # 无介音
            if cd == 'n':
                final = v_py + 'n'
            elif cd == 'ŋ':
                final = v_py + 'ng'
            elif cd:
                final = v_py + cd
            else:
                final = v_py
        
        # 组合声母和韵母
        if not c_py:
            # 零声母，需要添加 y/w/yu
            if final.startswith('i'):
                if final == 'i':
                    return 'yi'
                elif final in ('in', 'ing'):
                    return 'y' + final
                else:
                    return 'y' + final[1:]
            elif final.startswith('u'):
                if final == 'u':
                    return 'wu'
                elif final == 'un':
                    return 'wen'
                elif final in ('ueng', 'ong'):
                    return 'weng'
                else:
                    return 'w' + final[1:]
            elif final.startswith('v'):
                if final == 'v':
                    return 'yu'
                else:
                    return 'yu' + final[1:]
            else:
                return final
        
        # 有声母
        if c_py in ('j', 'q', 'x'):
            # j/q/x + ü 系列，ü 写作 u
            if final.startswith('v'):
                return c_py + 'u' + final[1:]
            else:
                return c_py + final
        elif c_py in ('n', 'l'):
            # n/l + ü 系列，保持 v
            return c_py + final
        else:
            # 其他声母 + ü，ü 写作 u
            if final.startswith('v'):
                return c_py + 'u' + final[1:]
            else:
                return c_py + final
    
    def _extract_vc_pairs(
        self,
        words_tier,
        phones_tier,
        wav_name: str,
        wav_duration_ms: float,
        language: str,
        use_hiragana: bool,
        vc_offset_ratio: float,
        vc_overlap_ratio: float,
        vc_separator: str
    ) -> List[Dict]:
        """
        从 phones 层提取元音+辅音对（VC 部）
        
        VC 部是当前音节的韵母(V) + 下一个音节的声母(C)
        用于连接两个相邻音节的过渡部分
        
        使用 presamp.ini 中的映射规则来确定韵母和声母的对应关系
        
        注意：VC 部的别名始终使用拼音格式，不受 use_hiragana 参数影响
        
        参数:
            words_tier: words 层
            phones_tier: phones 层
            wav_name: 音频文件名
            wav_duration_ms: 音频总时长
            language: 语言
            use_hiragana: 是否使用平假名（VC 部忽略此参数，始终用拼音）
            vc_offset_ratio: VC 偏移比例
            vc_overlap_ratio: VC overlap 比例
            vc_separator: VC 别名分隔符
        
        返回:
            VC 条目列表
        """
        entries = []
        
        if language not in ('chinese', 'zh', 'mandarin'):
            # 非中文暂不支持 CVVC
            return entries
        
        # 加载 presamp.ini 映射
        vowel_map, consonant_map = self._load_presamp_mapping()
        if not vowel_map or not consonant_map:
            self._log("警告: 无法加载 presamp.ini 映射，跳过 VC 部生成")
            return entries
        
        intervals = list(phones_tier)
        
        # 解析所有音节，提取韵母和声母信息
        syllables = []
        i = 0
        
        while i < len(intervals):
            interval = intervals[i]
            phone = interval.mark.strip()
            
            if phone in SKIP_MARKS:
                i += 1
                continue
            
            # 解析一个完整音节：(辅音) + (介音) + 元音 + (韵尾)
            syllable_phones = []
            syllable_start = interval.minTime * 1000
            syllable_end = interval.maxTime * 1000
            consonant_duration = 0
            vowel_start = syllable_start
            vowel_end = syllable_end
            has_consonant = False
            
            # 1. 检查是否有声母（辅音）
            if is_consonant(phone, language):
                syllable_phones.append(phone)
                consonant_duration = interval.maxTime * 1000 - syllable_start
                has_consonant = True
                i += 1
                
                # 检查下一个音素
                if i < len(intervals):
                    next_interval = intervals[i]
                    next_phone = next_interval.mark.strip()
                    
                    if next_phone not in SKIP_MARKS:
                        phone = next_phone
                        syllable_end = next_interval.maxTime * 1000
                        vowel_start = next_interval.minTime * 1000
                    else:
                        # 只有辅音，没有元音，跳过
                        continue
                else:
                    # 只有辅音，没有元音，跳过
                    continue
            
            # 2. 检查是否有介音（j, w, ɥ）
            phone_base = _strip_tone(phone)
            if phone_base in CHINESE_MEDIALS:
                syllable_phones.append(phone)
                i += 1
                
                # 检查下一个音素（必须是元音）
                if i < len(intervals):
                    next_interval = intervals[i]
                    next_phone = next_interval.mark.strip()
                    
                    if next_phone not in SKIP_MARKS:
                        phone = next_phone
                        syllable_end = next_interval.maxTime * 1000
                    else:
                        # 只有介音，没有元音，跳过
                        continue
                else:
                    # 只有介音，没有元音，跳过
                    continue
            
            # 3. 必须有韵母（元音）
            if is_vowel(phone, language):
                syllable_phones.append(phone)
                vowel_end = interval.maxTime * 1000
                if not consonant_duration:
                    # 零声母，辅音时长设为元音前30ms
                    consonant_duration = min(30, (vowel_end - vowel_start) * 0.2)
                syllable_end = vowel_end
                i += 1
                
                # 4. 检查是否有韵尾（n, ng, i, u）
                if i < len(intervals):
                    next_interval = intervals[i]
                    next_phone = next_interval.mark.strip()
                    
                    if next_phone not in SKIP_MARKS:
                        # 检查是否是韵尾
                        next_phone_base = _strip_tone(next_phone)
                        if next_phone_base in CHINESE_CODAS:
                            syllable_phones.append(next_phone)
                            syllable_end = next_interval.maxTime * 1000
                            vowel_end = next_interval.maxTime * 1000
                            i += 1
                
                # 5. 将音节转换为拼音并保存
                pinyin = self._syllable_to_pinyin(syllable_phones, language, False)
                if pinyin:
                    # 使用 presamp.ini 映射查找韵母和声母
                    vowel_part = self._find_vowel_in_mapping(pinyin, vowel_map)
                    consonant_part = self._find_consonant_in_mapping(pinyin, consonant_map) if has_consonant else None
                    
                    if vowel_part:
                        syllables.append({
                            'pinyin': pinyin,
                            'vowel_part': vowel_part,
                            'consonant_part': consonant_part,
                            'vowel_start': vowel_start,
                            'vowel_end': vowel_end,
                            'syllable_end': syllable_end
                        })
            else:
                # 不是元音，跳过
                i += 1
        
        # 生成 VC 对：当前音节的韵母 + 下一个音节的声母
        for idx in range(len(syllables) - 1):
            current = syllables[idx]
            next_syl = syllables[idx + 1]
            
            # 获取下一个音节的声母
            next_consonant = next_syl.get('consonant_part')
            
            # 如果下一个音节没有声母（零声母），跳过
            if not next_consonant:
                continue
            
            # 生成 VC 别名
            vc_alias = f"{current['vowel_part']}{vc_separator}{next_consonant}"
            
            # 计算 VC 参数
            entry = self._calculate_vc_params(
                wav_name=wav_name,
                alias=vc_alias,
                vowel_start_ms=current['vowel_start'],
                vowel_end_ms=current['vowel_end'],
                consonant_end_ms=next_syl['syllable_end'],
                wav_duration_ms=wav_duration_ms,
                vc_offset_ratio=vc_offset_ratio,
                vc_overlap_ratio=vc_overlap_ratio
            )
            entries.append(entry)
        
        return entries
    
    def _load_presamp_mapping(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        加载中文 CVVC 韵母和声母映射（内置数据）
        
        返回:
            (韵母映射字典, 声母映射字典)
            韵母映射: {完整拼音: 韵母标识}
            声母映射: {完整拼音: 声母标识}
        """
        vowel_map = {}  # {拼音: 韵母标识}
        consonant_map = {}  # {拼音: 声母标识}
        
        # 内置韵母映射数据（来自 presamp.ini [VOWEL] 部分）
        vowel_data = {
            'a': ['a', 'ba', 'pa', 'ma', 'fa', 'da', 'ta', 'na', 'la', 'ga', 'ka', 'ha', 'zha', 'cha', 'sha', 'za', 'ca', 'sa', 'ya', 'lia', 'jia', 'qia', 'xia', 'wa', 'gua', 'kua', 'hua', 'zhua', 'shua', 'dia'],
            'ai': ['ai', 'bai', 'pai', 'mai', 'dai', 'tai', 'nai', 'lai', 'gai', 'kai', 'hai', 'zhai', 'chai', 'shai', 'zai', 'cai', 'sai', 'wai', 'guai', 'kuai', 'huai', 'zhuai', 'chuai', 'shuai'],
            'an': ['an', 'ban', 'pan', 'man', 'fan', 'dan', 'tan', 'nan', 'lan', 'gan', 'kan', 'han', 'zhan', 'chan', 'shan', 'ran', 'zan', 'can', 'san', 'wan', 'duan', 'tuan', 'nuan', 'luan', 'guan', 'kuan', 'huan', 'zhuan', 'chuan', 'shuan', 'ruan', 'zuan', 'cuan', 'suan'],
            'ang': ['ang', 'bang', 'pang', 'mang', 'fang', 'dang', 'tang', 'nang', 'lang', 'gang', 'kang', 'hang', 'zhang', 'chang', 'shang', 'rang', 'zang', 'cang', 'sang', 'yang', 'liang', 'jiang', 'qiang', 'xiang', 'wang', 'guang', 'kuang', 'huang', 'zhuang', 'chuang', 'shuang', 'niang'],
            'ao': ['ao', 'bao', 'pao', 'mao', 'dao', 'tao', 'nao', 'lao', 'gao', 'kao', 'hao', 'zhao', 'chao', 'shao', 'rao', 'zao', 'cao', 'sao', 'yao', 'biao', 'piao', 'miao', 'diao', 'tiao', 'niao', 'liao', 'jiao', 'qiao', 'xiao'],
            'e': ['e', 'me', 'de', 'te', 'ne', 'le', 'ge', 'ke', 'he', 'zhe', 'che', 'she', 're', 'ze', 'ce', 'se'],
            'e0': ['ye', 'bie', 'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'jie', 'qie', 'xie', 'yue', 'nue', 'lue', 'jue', 'que', 'xue'],
            'ei': ['ei', 'bei', 'pei', 'mei', 'fei', 'dei', 'tei', 'nei', 'lei', 'gei', 'kei', 'hei', 'zhei', 'shei', 'zei', 'wei', 'dui', 'tui', 'gui', 'kui', 'hui', 'zhui', 'chui', 'shui', 'rui', 'zui', 'cui', 'sui'],
            'en': ['en', 'ben', 'pen', 'men', 'fen', 'nen', 'gen', 'ken', 'hen', 'zhen', 'chen', 'shen', 'ren', 'zen', 'cen', 'sen', 'wen', 'dun', 'tun', 'lun', 'gun', 'kun', 'hun', 'zhun', 'chun', 'shun', 'run', 'zun', 'cun', 'sun'],
            'en0': ['yan', 'bian', 'pian', 'mian', 'dian', 'tian', 'nian', 'lian', 'jian', 'qian', 'xian', 'yuan', 'juan', 'quan', 'xuan'],
            'eng': ['beng', 'peng', 'meng', 'feng', 'deng', 'teng', 'neng', 'leng', 'geng', 'keng', 'heng', 'weng', 'zheng', 'cheng', 'sheng', 'reng', 'zeng', 'ceng', 'seng'],
            'er': ['er'],
            'i': ['bi', 'pi', 'mi', 'di', 'ti', 'ni', 'li', 'ji', 'qi', 'xi', 'yi'],
            'in': ['yin', 'bin', 'pin', 'min', 'nin', 'lin', 'jin', 'qin', 'xin'],
            'ing': ['ying', 'bing', 'ping', 'ming', 'ding', 'ting', 'ning', 'ling', 'jing', 'qing', 'xing'],
            'i0': ['zi', 'ci', 'si'],
            'ir': ['zhi', 'chi', 'shi', 'ri'],
            'o': ['bo', 'po', 'mo', 'fo', 'wo', 'duo', 'tuo', 'nuo', 'luo', 'guo', 'kuo', 'huo', 'zhuo', 'chuo', 'shuo', 'ruo', 'zuo', 'cuo', 'suo'],
            'ong': ['dong', 'tong', 'nong', 'long', 'gong', 'kong', 'hong', 'zhong', 'chong', 'rong', 'zong', 'cong', 'song', 'yong', 'jiong', 'qiong', 'xiong'],
            'ou': ['ou', 'pou', 'mou', 'fou', 'dou', 'tou', 'lou', 'gou', 'kou', 'hou', 'zhou', 'chou', 'shou', 'rou', 'zou', 'cou', 'sou', 'you', 'miu', 'diu', 'niu', 'liu', 'jiu', 'qiu', 'xiu'],
            'u': ['bu', 'pu', 'mu', 'fu', 'du', 'tu', 'nu', 'lu', 'gu', 'ku', 'hu', 'zhu', 'chu', 'shu', 'ru', 'zu', 'cu', 'su', 'wu'],
            'v': ['yu', 'nv', 'lv', 'ju', 'qu', 'xu'],
            'vn': ['yun', 'jun', 'qun', 'xun'],
        }
        
        # 内置声母映射数据（来自 presamp.ini [CONSONANT] 部分）
        consonant_data = {
            'b': ['ba', 'bai', 'ban', 'bang', 'bao', 'biao', 'bie', 'bei', 'ben', 'bian', 'beng', 'bi', 'bin', 'bing', 'bo', 'bu'],
            'p': ['pa', 'pai', 'pan', 'pang', 'pao', 'piao', 'pie', 'pei', 'pen', 'pian', 'peng', 'pi', 'pin', 'ping', 'po', 'pou', 'pu'],
            'm': ['ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mo', 'mou', 'mu'],
            'f': ['fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fo', 'fou', 'fu'],
            'd': ['da', 'dia', 'dai', 'dan', 'duan', 'dang', 'dao', 'diao', 'de', 'die', 'dei', 'dui', 'dun', 'dian', 'deng', 'di', 'ding', 'duo', 'dong', 'dou', 'diu', 'du'],
            't': ['ta', 'tai', 'tan', 'tuan', 'tang', 'tao', 'tiao', 'te', 'tie', 'tei', 'tui', 'tun', 'tian', 'teng', 'ti', 'ting', 'tuo', 'tong', 'tou', 'tu'],
            'n': ['na', 'nai', 'nan', 'nuan', 'nang', 'nao', 'ne', 'nue', 'nei', 'nen', 'neng', 'nuo', 'nong', 'nu', 'nv'],
            'l': ['la', 'lai', 'lan', 'luan', 'lang', 'lao', 'le', 'lue', 'lei', 'lun', 'leng', 'luo', 'long', 'lou', 'lu', 'lv'],
            'g': ['ga', 'gua', 'gai', 'guai', 'gan', 'guan', 'gang', 'guang', 'gao', 'ge', 'gei', 'gui', 'gen', 'gun', 'geng', 'guo', 'gong', 'gou', 'gu'],
            'k': ['ka', 'kua', 'kai', 'kuai', 'kan', 'kuan', 'kang', 'kuang', 'kao', 'ke', 'kei', 'kui', 'ken', 'kun', 'keng', 'kuo', 'kong', 'kou', 'ku'],
            'h': ['ha', 'hai', 'han', 'hang', 'hao', 'he', 'hei', 'hen', 'heng', 'hong', 'hou'],
            'zh': ['zha', 'zhua', 'zhai', 'zhuai', 'zhan', 'zhuan', 'zhang', 'zhuang', 'zhao', 'zhe', 'zhei', 'zhui', 'zhen', 'zhun', 'zheng', 'zhi', 'zhuo', 'zhong', 'zhou', 'zhu'],
            'ch': ['cha', 'chai', 'chuai', 'chan', 'chuan', 'chang', 'chuang', 'chao', 'che', 'chui', 'chen', 'chun', 'cheng', 'chi', 'chuo', 'chong', 'chou', 'chu'],
            'sh': ['sha', 'shai', 'shan', 'shang', 'shao', 'she', 'shei', 'shen', 'sheng', 'shi', 'shou'],
            'z': ['za', 'zai', 'zan', 'zuan', 'zang', 'zao', 'ze', 'zei', 'zui', 'zen', 'zun', 'zeng', 'zi', 'zuo', 'zong', 'zou', 'zu'],
            'c': ['ca', 'cai', 'can', 'cuan', 'cang', 'cao', 'ce', 'cui', 'cen', 'cun', 'ceng', 'ci', 'cuo', 'cong', 'cou', 'cu'],
            's': ['sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'si', 'song', 'sou'],
            'y': ['ya', 'yang', 'yao', 'ye', 'yan', 'yi', 'yin', 'ying', 'yong', 'you'],
            'ly': ['lia', 'liang', 'liao', 'lie', 'lian', 'li', 'lin', 'ling', 'liu'],
            'j': ['jia', 'jiang', 'jiao', 'jie', 'jue', 'jian', 'juan', 'ji', 'jin', 'jing', 'jiong', 'jiu', 'ju', 'jun'],
            'q': ['qia', 'qiang', 'qiao', 'qie', 'que', 'qian', 'quan', 'qi', 'qin', 'qing', 'qiong', 'qiu', 'qu', 'qun'],
            'xy': ['xia', 'xiang', 'xiao', 'xie', 'xian', 'xi', 'xin', 'xing', 'xiong', 'xiu'],
            'w': ['wa', 'wai', 'wan', 'wang', 'wei', 'wen', 'weng', 'wo', 'wu'],
            'hw': ['hua', 'huai', 'huan', 'huang', 'hui', 'hun', 'huo', 'hu'],
            'shw': ['shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'shu'],
            'r': ['ran', 'ruan', 'rang', 'rao', 're', 'rui', 'ren', 'run', 'reng', 'ri', 'ruo', 'rong', 'rou', 'ru'],
            'sw': ['suan', 'sui', 'sun', 'suo', 'su'],
            'ny': ['niang', 'niao', 'nie', 'nian', 'ni', 'nin', 'ning', 'niu'],
            'my': ['miao', 'mie', 'mian', 'mi', 'min', 'ming', 'miu'],
            'v': ['yu', 'yue', 'yuan', 'yun'],
            'xw': ['xue', 'xuan', 'xu', 'xun'],
        }
        
        # 构建韵母映射
        for vowel_id, pinyins in vowel_data.items():
            for pinyin in pinyins:
                vowel_map[pinyin] = vowel_id
        
        # 构建声母映射
        for consonant_id, pinyins in consonant_data.items():
            for pinyin in pinyins:
                consonant_map[pinyin] = consonant_id
        
        self._log(f"加载内置 CVVC 映射: {len(vowel_map)} 个韵母映射, {len(consonant_map)} 个声母映射")
        return vowel_map, consonant_map
    
    def _find_vowel_in_mapping(self, pinyin: str, vowel_map: Dict[str, str]) -> Optional[str]:
        """
        在韵母映射中查找拼音对应的韵母标识
        
        参数:
            pinyin: 完整拼音
            vowel_map: 韵母映射字典
        
        返回:
            韵母标识，如果未找到则返回 None
        """
        return vowel_map.get(pinyin)
    
    def _find_consonant_in_mapping(self, pinyin: str, consonant_map: Dict[str, str]) -> Optional[str]:
        """
        在声母映射中查找拼音对应的声母标识
        
        参数:
            pinyin: 完整拼音
            consonant_map: 声母映射字典
        
        返回:
            声母标识，如果未找到则返回 None
        """
        return consonant_map.get(pinyin)
    
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
    
    def _calculate_vc_params(
        self,
        wav_name: str,
        alias: str,
        vowel_start_ms: float,
        vowel_end_ms: float,
        consonant_end_ms: float,
        wav_duration_ms: float,
        vc_offset_ratio: float,
        vc_overlap_ratio: float
    ) -> Dict:
        """
        计算 VC 部的 oto.ini 参数
        
        VC 部从元音后半段开始，到辅音结束
        
        参数:
            wav_name: 音频文件名
            alias: VC 别名
            vowel_start_ms: 元音开始时间
            vowel_end_ms: 元音结束时间（即辅音开始时间）
            consonant_end_ms: 辅音结束时间
            wav_duration_ms: 音频总时长
            vc_offset_ratio: VC 偏移比例
            vc_overlap_ratio: VC overlap 比例
        
        返回:
            oto 参数字典
        """
        vowel_duration = vowel_end_ms - vowel_start_ms
        
        # offset: 元音后半段位置
        offset = vowel_end_ms - vowel_duration * vc_offset_ratio
        
        # 总时长（从 offset 到辅音结束）
        segment_duration = consonant_end_ms - offset
        
        # preutterance: 从 offset 到辅音开始（即元音结束）的距离
        preutterance = vowel_end_ms - offset
        
        # consonant: 固定区域，较短
        consonant = min(30, segment_duration * 0.3)
        
        # overlap: 较大，平滑过渡
        overlap = preutterance * vc_overlap_ratio
        
        # cutoff: 负值，表示总时长
        cutoff = -segment_duration
        
        return {
            "wav_name": wav_name,
            "alias": alias,
            "offset": round(offset, 1),
            "consonant": round(consonant, 1),
            "cutoff": round(cutoff, 1),
            "preutterance": round(preutterance, 1),
            "overlap": round(overlap, 1),
            "segment_duration": segment_duration,
            "is_vc": True  # 标记为 VC 部
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
            valid_vowels = {'a', 'o', 'e', 'i', 'u', 'v',
                          'ai', 'ei', 'ao', 'ou',
                          'an', 'en', 'ang', 'eng', 'ong',
                          'ia', 'ie', 'iao', 'iu', 'ian', 'in', 'iang', 'ing', 'iong',
                          'ua', 'uo', 'uai', 'ui', 'uan', 'un', 'uang', 'ueng',
                          've', 'van', 'vn', 'er'}
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
        
        # 中文所有可能的韵母（包含所有标准韵母）
        all_vowels = ['a', 'o', 'e', 'i', 'u', 'v',
                      'ai', 'ei', 'ao', 'ou',
                      'an', 'en', 'ang', 'eng', 'ong',
                      'ia', 'ie', 'iao', 'iu', 'ian', 'in', 'iang', 'ing', 'iong',
                      'ua', 'uo', 'uai', 'ui', 'uan', 'un', 'uang', 'ueng',
                      've', 'van', 'vn', 'er']
        
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
