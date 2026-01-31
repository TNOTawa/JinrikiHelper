# -*- coding: utf-8 -*-
"""
文本处理模块
将中文文本转换为拼音，供 MFA 对齐使用
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)


# ==================== 单字转拼音/罗马音 ====================

def char_to_pinyin(char: str, language: str = "chinese") -> Optional[str]:
    """
    将单个字符转换为拼音/罗马音
    
    参数:
        char: 单个字符
        language: 语言 (chinese/japanese)
    
    返回:
        拼音/罗马音字符串，无法转换返回 None
    """
    if not char or len(char) != 1:
        return None
    
    if language in ("chinese", "zh", "mandarin"):
        return _chinese_char_to_pinyin(char)
    elif language in ("japanese", "ja", "jp"):
        return _japanese_char_to_romaji(char)
    else:
        # 英文字母直接返回小写
        if char.isalpha():
            return char.lower()
        return None


def _chinese_char_to_pinyin(char: str) -> Optional[str]:
    """中文单字转拼音"""
    try:
        from pypinyin import pinyin, Style
        
        # 数字转中文读法
        digit_map = {
            '0': 'ling', '1': 'yi', '2': 'er', '3': 'san', '4': 'si',
            '5': 'wu', '6': 'liu', '7': 'qi', '8': 'ba', '9': 'jiu',
            '０': 'ling', '１': 'yi', '２': 'er', '３': 'san', '４': 'si',
            '５': 'wu', '６': 'liu', '７': 'qi', '８': 'ba', '９': 'jiu',
        }
        if char in digit_map:
            return digit_map[char]
        
        # 英文字母按中文读法
        letter_map = {
            'a': 'ei', 'b': 'bi', 'c': 'xi', 'd': 'di', 'e': 'yi',
            'f': 'ai fu', 'g': 'ji', 'h': 'ai qi', 'i': 'ai', 'j': 'jie',
            'k': 'kai', 'l': 'ai lu', 'm': 'ai mu', 'n': 'en', 'o': 'ou',
            'p': 'pi', 'q': 'kiu', 'r': 'a', 's': 'ai si', 't': 'ti',
            'u': 'you', 'v': 'wei', 'w': 'da bu liu', 'x': 'ai ke si',
            'y': 'wai', 'z': 'zei',
        }
        lower_char = char.lower()
        if lower_char in letter_map:
            # 返回第一个音节
            return letter_map[lower_char].split()[0]
        
        # 汉字转拼音
        result = pinyin(char, style=Style.NORMAL, heteronym=False)
        if result and result[0] and result[0][0]:
            return result[0][0].strip()
        
        return None
    except ImportError:
        logger.error("pypinyin 未安装")
        return None


def _japanese_char_to_romaji(char: str) -> Optional[str]:
    """日文单字转罗马音"""
    try:
        import pykakasi
        
        # 数字转日文读法
        digit_map = {
            '0': 'zero', '1': 'ichi', '2': 'ni', '3': 'san', '4': 'yon',
            '5': 'go', '6': 'roku', '7': 'nana', '8': 'hachi', '9': 'kyuu',
        }
        if char in digit_map:
            return digit_map[char]
        
        kks = pykakasi.kakasi()
        result = kks.convert(char)
        if result and result[0]:
            romaji = result[0].get('hepburn', result[0].get('orig', ''))
            return romaji if romaji else None
        return None
    except ImportError:
        logger.error("pykakasi 未安装")
        return None


def is_valid_char(char: str, language: str = "chinese") -> bool:
    """
    判断字符是否为有效的可转换字符
    
    参数:
        char: 单个字符
        language: 语言
    
    返回:
        是否有效
    """
    if not char or len(char) != 1:
        return False
    
    # 数字有效
    if char.isdigit():
        return True
    
    # 英文字母有效
    if char.isalpha() and char.isascii():
        return True
    
    if language in ("chinese", "zh", "mandarin"):
        # 中文字符范围
        return '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf'
    elif language in ("japanese", "ja", "jp"):
        # 日文假名和汉字
        return (
            '\u3040' <= char <= '\u309f' or  # 平假名
            '\u30a0' <= char <= '\u30ff' or  # 片假名
            '\u4e00' <= char <= '\u9fff'     # 汉字
        )
    
    return False


def chinese_to_pinyin(text: str) -> str:
    """
    将中文文本转换为拼音（空格分隔）
    
    参数:
        text: 中文文本
    
    返回:
        拼音字符串，空格分隔
    """
    try:
        from pypinyin import pinyin, Style
        
        # 获取拼音，不带声调
        result = pinyin(text, style=Style.NORMAL, heteronym=False)
        
        # 展平并过滤空值
        pinyins = []
        for item in result:
            if item and item[0]:
                py = item[0].strip()
                if py:
                    pinyins.append(py)
        
        return ' '.join(pinyins)
    except ImportError:
        logger.error("pypinyin 未安装，请运行: pip install pypinyin")
        raise


def japanese_to_romaji(text: str) -> str:
    """
    将日文文本转换为罗马字
    
    参数:
        text: 日文文本
    
    返回:
        罗马字字符串，空格分隔
    """
    try:
        import pykakasi
        
        kks = pykakasi.kakasi()
        result = kks.convert(text)
        
        romajis = []
        for item in result:
            romaji = item.get('hepburn', item.get('orig', ''))
            if romaji:
                romajis.append(romaji)
        
        return ' '.join(romajis)
    except ImportError:
        logger.error("pykakasi 未安装，请运行: pip install pykakasi")
        raise


def process_lab_file(
    lab_path: str,
    language: str = "chinese",
    output_path: Optional[str] = None
) -> Tuple[bool, str]:
    """
    处理单个 .lab 文件，将文本转换为拼音/罗马字
    
    参数:
        lab_path: .lab 文件路径
        language: 语言 (chinese/japanese)
        output_path: 输出路径，默认覆盖原文件
    
    返回:
        (成功标志, 转换后的文本或错误信息)
    """
    try:
        with open(lab_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            return False, "空文件"
        
        # 根据语言选择转换函数
        if language in ("chinese", "zh", "mandarin"):
            converted = chinese_to_pinyin(text)
        elif language in ("japanese", "ja", "jp"):
            converted = japanese_to_romaji(text)
        else:
            # 英文或其他语言，保持原样但分词
            converted = ' '.join(text.split())
        
        # 写入文件
        output = output_path or lab_path
        with open(output, 'w', encoding='utf-8') as f:
            f.write(converted)
        
        return True, converted
        
    except Exception as e:
        logger.error(f"处理 {lab_path} 失败: {e}")
        return False, str(e)


def process_lab_directory(
    input_dir: str,
    language: str = "chinese",
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str, int]:
    """
    批量处理目录下的所有 .lab 文件
    
    参数:
        input_dir: 输入目录
        language: 语言
        output_dir: 输出目录，默认覆盖原文件
        progress_callback: 进度回调
    
    返回:
        (成功标志, 消息, 处理文件数)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    try:
        lab_files = list(Path(input_dir).glob('*.lab'))
        
        if not lab_files:
            return False, "未找到 .lab 文件", 0
        
        log(f"找到 {len(lab_files)} 个 .lab 文件")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for i, lab_path in enumerate(lab_files):
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, lab_path.name)
            
            success, result = process_lab_file(
                str(lab_path), 
                language, 
                output_path
            )
            
            if success:
                success_count += 1
                log(f"[{i+1}/{len(lab_files)}] {lab_path.name} -> {result[:30]}...")
            else:
                log(f"[{i+1}/{len(lab_files)}] {lab_path.name} 失败: {result}")
        
        return True, f"处理完成: {success_count}/{len(lab_files)}", success_count
        
    except Exception as e:
        logger.error(f"批量处理失败: {e}", exc_info=True)
        return False, str(e), 0
