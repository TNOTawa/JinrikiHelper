# -*- coding: utf-8 -*-
"""
音频库排序模块
按时长排序并导出音频片段
"""

import os
import glob
import shutil
import logging
from typing import Optional, Callable, Tuple, Dict, List

logger = logging.getLogger(__name__)


def sort_and_export_bank(
    bank_dir: str,
    output_dir: str,
    max_per_word: int = 100,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str, Dict[str, int]]:
    """
    对音频库进行排序并导出
    
    参数:
        bank_dir: 音频库目录 (包含 [词]/[编号].wav 结构)
        output_dir: 导出目录
        max_per_word: 每个词最多保留的样本数
        progress_callback: 进度回调函数
    
    返回:
        (成功标志, 消息, 导出统计)
    """
    import audiofile
    
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计所有词条
        stats: Dict[str, List[Tuple[str, float]]] = {}
        wav_files = glob.glob(os.path.join(bank_dir, '**', '*.wav'), recursive=True)
        
        if not wav_files:
            return False, "未找到WAV文件", {}
        
        log(f"扫描到 {len(wav_files)} 个WAV文件")
        
        for path in wav_files:
            rel_path = os.path.relpath(path, bank_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                word = parts[0]
                if word not in stats:
                    stats[word] = []
                try:
                    duration = audiofile.duration(path)
                    stats[word].append((path, duration))
                except Exception as e:
                    log(f"警告: 无法读取 {path}: {e}")
        
        log(f"统计到 {len(stats)} 个词条")
        
        # 按时长排序并导出
        export_counts = {}
        for word, files in stats.items():
            # 按时长降序排序
            sorted_files = sorted(files, key=lambda x: -x[1])
            count = 0
            for idx, (src_path, _) in enumerate(sorted_files[:max_per_word]):
                dst_path = os.path.join(output_dir, f'{word}_{idx}.wav')
                shutil.copyfile(src_path, dst_path)
                count += 1
            export_counts[word] = count
            log(f"处理词条: {word} ({count} 个文件)")
        
        total = sum(export_counts.values())
        log(f"导出完成: {len(export_counts)} 个词条，{total} 个文件")
        
        return True, f"导出完成: {len(export_counts)} 个词条，{total} 个文件", export_counts
        
    except Exception as e:
        logger.error(f"排序导出失败: {e}", exc_info=True)
        return False, str(e), {}


# 保留原有脚本入口以兼容
if __name__ == "__main__":
    bank_dir = 'bank'
    output_dir = 'bank_export'
    
    success, msg, stats = sort_and_export_bank(
        bank_dir=bank_dir,
        output_dir=output_dir,
        max_per_word=100,
        progress_callback=print
    )
    print(f"结果: {msg}")
