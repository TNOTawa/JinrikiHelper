# -*- coding: utf-8 -*-
"""
简单单字导出插件

从TextGrid提取分词片段，按拼音排序导出
"""

import os
import json
import glob
import shutil
import logging
from typing import Any, Dict, List, Tuple

from .base import ExportPlugin, PluginOption, OptionType

logger = logging.getLogger(__name__)


class SimpleExportPlugin(ExportPlugin):
    """简单单字导出插件"""
    
    name = "简单单字导出"
    description = "从TextGrid提取分词片段，按拼音排序导出"
    version = "1.1.0"
    author = "内置"
    
    def get_options(self) -> List[PluginOption]:
        return [
            PluginOption(
                key="info",
                label="将每个汉字按拼音分类，选取最佳样本导出",
                option_type=OptionType.LABEL
            ),
            PluginOption(
                key="max_samples",
                label="每个拼音最大样本数",
                option_type=OptionType.NUMBER,
                default=10,
                min_value=1,
                max_value=1000,
                description="按时长排序，保留最长的N个"
            ),
            PluginOption(
                key="naming_rule",
                label="命名规则",
                option_type=OptionType.TEXT,
                default="%p%%n%",
                description="变量: %p%=拼音, %n%=序号。示例: %p%_%n% → ba_1.wav"
            ),
            PluginOption(
                key="first_naming_rule",
                label="首个样本命名规则",
                option_type=OptionType.TEXT,
                default="%p%",
                description="第0个样本的特殊规则，留空则使用通用规则。示例: %p% → ba.wav"
            ),
            PluginOption(
                key="clean_temp",
                label="导出后清理临时文件",
                option_type=OptionType.SWITCH,
                default=True,
                description="删除临时的segments目录"
            )
        ]
    
    def _load_language_from_meta(self, bank_dir: str, source_name: str) -> str:
        """从meta.json加载语言设置"""
        meta_path = os.path.join(bank_dir, source_name, "meta.json")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    language = meta.get("language", "chinese")
                    self._log(f"从meta.json读取语言设置: {language}")
                    return language
        except Exception as e:
            logger.warning(f"读取meta.json失败: {e}")
        return "chinese"
    
    def _apply_naming_rule(self, rule: str, pinyin: str, index: int) -> str:
        """应用命名规则生成文件名"""
        name = rule.replace("%p%", pinyin).replace("%n%", str(index))
        return name
    
    def export(
        self,
        source_name: str,
        bank_dir: str,
        options: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """执行简单单字导出"""
        try:
            # 自动从meta.json获取语言设置
            language = self._load_language_from_meta(bank_dir, source_name)
            max_samples = int(options.get("max_samples", 10))
            naming_rule = options.get("naming_rule", "%p%_%n%")
            first_naming_rule = options.get("first_naming_rule", "")
            clean_temp = options.get("clean_temp", True)
            
            paths = self.get_source_paths(bank_dir, source_name)
            export_dir = self.get_export_dir(bank_dir, source_name, "simple_export")
            
            # 临时segments目录
            temp_base = os.path.join(bank_dir, ".temp_segments")
            segments_dir = os.path.join(temp_base, source_name)
            
            # 步骤1: 提取分词片段
            self._log("【提取分词片段】")
            success, msg, pinyin_counts = self._extract_segments(
                paths["slices_dir"],
                paths["textgrid_dir"],
                segments_dir,
                language
            )
            if not success:
                return False, msg
            
            # 步骤2: 排序导出
            self._log("\n【排序导出】")
            success, msg = self._sort_and_export(
                segments_dir,
                export_dir,
                max_samples,
                naming_rule,
                first_naming_rule
            )
            if not success:
                return False, msg
            
            # 清理临时目录
            if clean_temp and os.path.exists(segments_dir):
                self._log(f"\n清理临时目录: {segments_dir}")
                shutil.rmtree(segments_dir)
                if os.path.exists(temp_base) and not os.listdir(temp_base):
                    shutil.rmtree(temp_base)
            
            return True, f"导出完成: {export_dir}"
            
        except Exception as e:
            logger.error(f"简单单字导出失败: {e}", exc_info=True)
            return False, str(e)
    
    def _extract_segments(
        self,
        slices_dir: str,
        textgrid_dir: str,
        segments_dir: str,
        language: str
    ) -> Tuple[bool, str, Dict[str, int]]:
        """
        提取分词片段
        
        中文：使用words层按字切分，用char_to_pinyin获取拼音名称
        日语：使用phones层按音素切分，合并辅音+元音为音节
        """
        try:
            import textgrid
            import soundfile as sf
            
            os.makedirs(segments_dir, exist_ok=True)
            
            tg_files = glob.glob(os.path.join(textgrid_dir, '*.TextGrid'))
            if not tg_files:
                return False, "未找到TextGrid文件", {}
            
            self._log(f"处理 {len(tg_files)} 个TextGrid文件")
            
            # 根据语言选择提取方法
            if language in ("japanese", "ja", "jp"):
                return self._extract_japanese_segments(
                    tg_files, slices_dir, segments_dir
                )
            else:
                return self._extract_chinese_segments(
                    tg_files, slices_dir, segments_dir, language
                )
            
        except Exception as e:
            logger.error(f"提取分词失败: {e}", exc_info=True)
            return False, str(e), {}
    
    def _extract_chinese_segments(
        self,
        tg_files: List[str],
        slices_dir: str,
        segments_dir: str,
        language: str
    ) -> Tuple[bool, str, Dict[str, int]]:
        """
        中文音频提取
        
        使用words层的时间边界，按字符切分，用char_to_pinyin获取拼音
        """
        import textgrid
        import soundfile as sf
        from src.text_processor import char_to_pinyin, is_valid_char
        
        pinyin_counts: Dict[str, int] = {}
        
        for tg_path in tg_files:
            basename = os.path.basename(tg_path).replace('.TextGrid', '.wav')
            wav_path = os.path.join(slices_dir, basename)
            
            if not os.path.exists(wav_path):
                self._log(f"警告: 找不到 {basename}")
                continue
            
            tg = textgrid.TextGrid.fromFile(tg_path)
            audio, sr = sf.read(wav_path, dtype='float32')
            
            # 使用words层（第一层）
            words_tier = tg[0]
            
            for interval in words_tier:
                word_text = interval.mark.strip()
                
                if not word_text or word_text in ['', 'SP', 'AP', '<unk>', 'spn', 'sil']:
                    continue
                
                start_time = interval.minTime
                end_time = interval.maxTime
                duration = end_time - start_time
                
                # 获取有效字符
                chars = list(word_text)
                valid_chars = [c for c in chars if is_valid_char(c, language)]
                
                if not valid_chars:
                    continue
                
                # 按字符均分时长
                char_duration = duration / len(valid_chars)
                
                for i, char in enumerate(valid_chars):
                    pinyin = char_to_pinyin(char, language)
                    if not pinyin:
                        continue
                    
                    char_start = start_time + i * char_duration
                    char_end = char_start + char_duration
                    
                    pinyin_dir = os.path.join(segments_dir, pinyin)
                    os.makedirs(pinyin_dir, exist_ok=True)
                    
                    current_count = pinyin_counts.get(pinyin, 0)
                    index = current_count + 1
                    pinyin_counts[pinyin] = index
                    
                    start_sample = int(round(char_start * sr))
                    end_sample = int(round(char_end * sr))
                    segment = audio[start_sample:end_sample]
                    
                    if len(segment) == 0:
                        pinyin_counts[pinyin] = current_count
                        continue
                    
                    output_path = os.path.join(pinyin_dir, f'{index}.wav')
                    sf.write(output_path, segment, sr, subtype='PCM_16')
        
        total = sum(pinyin_counts.values())
        self._log(f"提取完成: {len(pinyin_counts)} 个拼音，共 {total} 个片段")
        
        return True, f"提取完成: {len(pinyin_counts)} 个拼音", pinyin_counts
    
    def _extract_japanese_segments(
        self,
        tg_files: List[str],
        slices_dir: str,
        segments_dir: str
    ) -> Tuple[bool, str, Dict[str, int]]:
        """
        日语音频提取
        
        使用phones层，将辅音+元音合并为音节
        """
        import textgrid
        import soundfile as sf
        
        phone_counts: Dict[str, int] = {}
        
        for tg_path in tg_files:
            basename = os.path.basename(tg_path).replace('.TextGrid', '.wav')
            wav_path = os.path.join(slices_dir, basename)
            
            if not os.path.exists(wav_path):
                self._log(f"警告: 找不到 {basename}")
                continue
            
            tg = textgrid.TextGrid.fromFile(tg_path)
            audio, sr = sf.read(wav_path, dtype='float32')
            
            # 查找phones层
            phones_tier = None
            for tier in tg:
                if tier.name.lower() in ('phones', 'phone'):
                    phones_tier = tier
                    break
            
            if phones_tier is None and len(tg) >= 2:
                phones_tier = tg[1]
            
            if phones_tier is None:
                self._log(f"警告: {basename} 未找到phones层，跳过")
                continue
            
            # 合并音素为音节
            syllables = self._merge_japanese_phones(phones_tier)
            
            for syllable, start_time, end_time in syllables:
                if not syllable:
                    continue
                
                # 标准化为ASCII
                normalized = self._normalize_japanese_phone(syllable)
                if not normalized:
                    continue
                
                phone_dir = os.path.join(segments_dir, normalized)
                os.makedirs(phone_dir, exist_ok=True)
                
                current_count = phone_counts.get(normalized, 0)
                index = current_count + 1
                phone_counts[normalized] = index
                
                start_sample = int(round(start_time * sr))
                end_sample = int(round(end_time * sr))
                segment = audio[start_sample:end_sample]
                
                if len(segment) == 0:
                    phone_counts[normalized] = current_count
                    continue
                
                output_path = os.path.join(phone_dir, f'{index}.wav')
                sf.write(output_path, segment, sr, subtype='PCM_16')
        
        total = sum(phone_counts.values())
        self._log(f"提取完成: {len(phone_counts)} 个音节，共 {total} 个片段")
        
        return True, f"提取完成: {len(phone_counts)} 个音节", phone_counts
    
    def _merge_japanese_phones(self, phones_tier) -> List[Tuple[str, float, float]]:
        """
        日语音素合并
        
        规则：辅音 + 元音 合并为一个音节
        """
        # 元音集合
        vowels = {'a', 'e', 'i', 'o', 'u', 'ɯ'}
        skip_marks = {'', 'SP', 'AP', '<unk>', 'spn', 'sil'}
        
        syllables = []
        pending_consonant = None
        pending_start = None
        
        for interval in phones_tier:
            phone = interval.mark.strip()
            
            if phone in skip_marks:
                pending_consonant = None
                pending_start = None
                continue
            
            # 移除长音符号判断元音
            base_phone = phone.rstrip('ː')
            is_vowel = base_phone in vowels
            
            if is_vowel:
                if pending_consonant is not None:
                    syllable = pending_consonant + phone
                    syllables.append((syllable, pending_start, interval.maxTime))
                    pending_consonant = None
                    pending_start = None
                else:
                    syllables.append((phone, interval.minTime, interval.maxTime))
            else:
                if pending_consonant is not None:
                    syllables.append((pending_consonant, pending_start, interval.minTime))
                pending_consonant = phone
                pending_start = interval.minTime
        
        if pending_consonant is not None:
            syllables.append((pending_consonant, pending_start, phones_tier[-1].maxTime))
        
        return syllables
    
    def _normalize_japanese_phone(self, phone: str) -> str:
        """
        日语音素标准化为ASCII
        """
        # IPA到罗马音的映射
        ipa_map = {
            # 元音
            'ɯ': 'u',
            'ɯː': 'u',
            'aː': 'a',
            'eː': 'e',
            'iː': 'i',
            'oː': 'o',
            'uː': 'u',
            # 辅音
            'ɲ': 'n',
            'ŋ': 'n',
            'ɕ': 'sh',
            'ʑ': 'j',
            'dʑ': 'j',
            'tɕ': 'ch',
            'ɡ': 'g',
            'ː': '',
        }
        
        result = phone
        
        # 按长度降序处理映射
        for ipa in sorted(ipa_map.keys(), key=len, reverse=True):
            if ipa in result:
                result = result.replace(ipa, ipa_map[ipa])
        
        # 移除非ASCII字符
        result = ''.join(c for c in result if c.isascii() and c.isalnum())
        
        return result.lower() if result else None
    

    
    def _sort_and_export(
        self,
        segments_dir: str,
        export_dir: str,
        max_samples: int,
        naming_rule: str,
        first_naming_rule: str
    ) -> Tuple[bool, str]:
        """排序并导出"""
        try:
            import soundfile as sf
            
            os.makedirs(export_dir, exist_ok=True)
            
            # 清空已有导出
            for f in os.listdir(export_dir):
                fp = os.path.join(export_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
            
            wav_files = glob.glob(
                os.path.join(segments_dir, '**', '*.wav'),
                recursive=True
            )
            
            if not wav_files:
                return False, "未找到分字片段"
            
            self._log(f"扫描到 {len(wav_files)} 个片段")
            
            # 按拼音分组
            stats: Dict[str, List[Tuple[str, float]]] = {}
            for path in wav_files:
                rel_path = os.path.relpath(path, segments_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    pinyin = parts[0]
                    if pinyin not in stats:
                        stats[pinyin] = []
                    info = sf.info(path)
                    stats[pinyin].append((path, info.duration))
            
            self._log(f"统计到 {len(stats)} 个拼音")
            self._log(f"命名规则: {naming_rule}")
            if first_naming_rule:
                self._log(f"首个样本规则: {first_naming_rule}")
            
            # 按时长排序并导出
            exported = 0
            for pinyin, files in stats.items():
                sorted_files = sorted(files, key=lambda x: -x[1])
                for idx, (src_path, _) in enumerate(sorted_files[:max_samples]):
                    # 第0个样本使用特殊规则（如果设置了）
                    if idx == 0 and first_naming_rule:
                        filename = self._apply_naming_rule(first_naming_rule, pinyin, idx)
                    else:
                        filename = self._apply_naming_rule(naming_rule, pinyin, idx)
                    
                    dst_path = os.path.join(export_dir, f'{filename}.wav')
                    shutil.copyfile(src_path, dst_path)
                    exported += 1
            
            self._log(f"导出完成: {exported} 个文件")
            return True, f"导出完成: {len(stats)} 个拼音，{exported} 个文件"
            
        except Exception as e:
            logger.error(f"排序导出失败: {e}", exc_info=True)
            return False, str(e)
