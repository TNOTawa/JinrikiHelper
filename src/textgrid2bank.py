# -*- coding: utf-8 -*-
"""
TextGrid 转音频库模块
从 TextGrid 对齐结果中提取分词片段
"""

import os
import glob
import logging
from typing import Optional, Callable, Dict, Tuple

logger = logging.getLogger(__name__)


def textgrid_to_bank(
    wav_dir: str,
    textgrid_dir: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str, Dict[str, int]]:
    """
    将 TextGrid 对齐结果转换为音频库
    
    参数:
        wav_dir: WAV文件目录
        textgrid_dir: TextGrid文件目录
        output_dir: 输出目录
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 消息, 词条统计)
    """
    import textgrid
    import audiofile
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        tg_files = glob.glob(os.path.join(textgrid_dir, '*.TextGrid'))
        if not tg_files:
            return False, "未找到TextGrid文件", {}
        
        log(f"处理 {len(tg_files)} 个TextGrid文件")
        
        word_counts = {}
        
        for idx, tg_path in enumerate(tg_files):
            basename = os.path.basename(tg_path).replace('.TextGrid', '.wav')
            wav_path = os.path.join(wav_dir, basename)
            
            if not os.path.exists(wav_path):
                log(f"警告: 找不到 {basename}")
                continue
            
            tg = textgrid.TextGrid.fromFile(tg_path)
            audio, sr = audiofile.read(wav_path)
            
            for word in tg[0]:
                if word.mark in ['SP', 'AP', '']:
                    continue
                
                word_text = word.mark.split(':')[0]
                word_dir = os.path.join(output_dir, word_text)
                os.makedirs(word_dir, exist_ok=True)
                
                # 找到下一个可用编号
                index = 1
                while os.path.exists(os.path.join(word_dir, f'{index}.wav')):
                    index += 1
                
                # 切出片段并保存
                start_sample = int(word.minTime * sr)
                end_sample = int(word.maxTime * sr)
                segment = audio[start_sample:end_sample]
                
                output_path = os.path.join(word_dir, f'{index}.wav')
                audiofile.write(output_path, segment, sr)
                
                word_counts[word_text] = word_counts.get(word_text, 0) + 1
            
            log(f"进度: {idx+1}/{len(tg_files)} - {basename}")
        
        total = sum(word_counts.values())
        log(f"提取完成: {len(word_counts)} 个词条，共 {total} 个片段")
        
        return True, f"提取完成: {len(word_counts)} 个词条", word_counts
        
    except Exception as e:
        logger.error(f"TextGrid转换失败: {e}", exc_info=True)
        return False, str(e), {}


# 保留原有脚本入口以兼容
if __name__ == "__main__":
    import tqdm
    
    wavDir = r'E:\Workspace\umamusume-voice-text-extractor\extracted'
    tgDir = r'E:\SVS\DiffSinger\MakeDiffSinger\temp\revised'
    saveDir = 'bank'
    
    success, msg, stats = textgrid_to_bank(
        wav_dir=wavDir,
        textgrid_dir=tgDir,
        output_dir=saveDir,
        progress_callback=print
    )
    print(f"结果: {msg}")
