# -*- coding: utf-8 -*-
"""
MFA 模型下载模块
支持下载中文和日文的声学模型及字典
包含 SHA256 哈希校验，确保文件完整性
"""

import os
import hashlib
import logging
import urllib.request
import urllib.error
import time
import re
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)
PROB_PATTERN = re.compile(r"\b(\d+\.\d+|1)\b")

# 模型下载基础 URL
GITHUB_RELEASE_BASE = "https://github.com/MontrealCorpusTools/mfa-models/releases/download"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/main"

# 支持的语言配置
# 格式: {语言代码: {名称, 声学模型信息, 字典信息}}
# sha256: 官方文件的 SHA256 哈希值（清理空行后），用于校验文件完整性
LANGUAGE_MODELS = {
    "mandarin": {
        "name": "中文 (普通话)",
        "acoustic": {
            "tag": "acoustic-mandarin_mfa-v3.0.0",
            "filename": "mandarin_mfa.zip",
            "description": "Mandarin MFA acoustic model v3.0.0",
            # 声学模型是 zip 文件，不需要清理空行，直接校验原始哈希
            "sha256": None,  # 暂不校验声学模型
        },
        "dictionary": {
            "tag": "dictionary-mandarin_china_mfa-v3.0.0",
            "filename": "mandarin_china_mfa.dict",
            "description": "Mandarin (China) MFA dictionary v3.0.0",
            # 字典文件清理空行后的哈希值
            "sha256": None,  # 首次下载时自动计算并保存
            "min_lines": 10000,  # 字典文件最少行数，用于基本完整性检查
        }
    },
    "japanese": {
        "name": "日文",
        "acoustic": {
            "tag": "acoustic-japanese_mfa-v3.0.0",
            "filename": "japanese_mfa.zip",
            "description": "Japanese MFA acoustic model v3.0.0",
            "sha256": None,
        },
        "dictionary": {
            "tag": "dictionary-japanese_mfa-v3.0.0",
            "filename": "japanese_mfa.dict",
            "description": "Japanese MFA dictionary v3.0.0",
            "sha256": None,
            "min_lines": 10000,  # 日语字典约 12 万行
        }
    }
}


def get_available_languages() -> dict:
    """获取可用的语言列表"""
    return {k: v["name"] for k, v in LANGUAGE_MODELS.items()}


def _calculate_file_hash(file_path: str) -> str:
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _get_hash_file_path(file_path: str) -> str:
    """获取哈希文件路径"""
    return file_path + ".sha256"


def _save_hash(file_path: str, hash_value: str):
    """保存哈希值到文件"""
    hash_file = _get_hash_file_path(file_path)
    with open(hash_file, 'w', encoding='utf-8') as f:
        f.write(hash_value)


def _load_saved_hash(file_path: str) -> Optional[str]:
    """加载保存的哈希值"""
    hash_file = _get_hash_file_path(file_path)
    if os.path.exists(hash_file):
        with open(hash_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def _verify_file_integrity(
    file_path: str,
    min_lines: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> tuple[bool, str]:
    """
    验证文件完整性
    
    参数:
        file_path: 文件路径
        min_lines: 最少行数要求（仅用于文本文件）
        progress_callback: 进度回调
    
    返回:
        (是否有效, 原因)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if not os.path.exists(file_path):
        return False, "文件不存在"
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空"
    
    # 对于字典文件，检查行数和格式
    if file_path.endswith('.dict'):
        try:
            valid_line_count = 0
            invalid_line_count = 0
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    parts = stripped.split()
                    # MFA 字典支持任意空白分隔，至少需 2 列
                    if len(parts) < 2:
                        invalid_line_count += 1
                        continue

                    # 与 MFA parse_dictionary_file 对齐：概率字段后必须有音素
                    rest = parts[1:]
                    idx = 0
                    while idx < len(rest) and idx < 4 and PROB_PATTERN.match(rest[idx]):
                        idx += 1
                    if idx >= len(rest):
                        invalid_line_count += 1
                        continue

                    if len(parts) >= 2:
                        valid_line_count += 1
                    else:
                        invalid_line_count += 1
            
            if min_lines and valid_line_count < min_lines:
                return False, f"有效字典行数不足: {valid_line_count} < {min_lines}"
            
            if invalid_line_count > 100:
                return False, f"无效行过多: {invalid_line_count} 行"
                
        except Exception as e:
            return False, f"读取文件失败: {e}"
    
    # 检查哈希值（如果有保存的哈希）
    saved_hash = _load_saved_hash(file_path)
    if saved_hash:
        current_hash = _calculate_file_hash(file_path)
        if current_hash != saved_hash:
            return False, f"哈希校验失败: 期望 {saved_hash[:16]}..., 实际 {current_hash[:16]}..."
    
    return True, "文件完整"


def _download_file(
    url: str,
    dest_path: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    retries: int = 2,
    timeout: int = 180,
) -> bool:
    """
    下载文件
    
    参数:
        url: 下载地址
        dest_path: 保存路径
        progress_callback: 进度回调
    
    返回:
        是否成功
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    # 创建目录
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for attempt in range(1, retries + 1):
        temp_path = dest_path + ".downloading"
        try:
            log(f"正在下载: {url} (第{attempt}/{retries}次)")

            # 下载文件
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = response.headers.get("Content-Length")
                if total_size:
                    total_size = int(total_size)
                    log(f"文件大小: {total_size / 1024 / 1024:.1f} MB")

                # 分块下载
                block_size = 8192
                downloaded = 0

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size and downloaded % (block_size * 100) == 0:
                            percent = downloaded / total_size * 100
                            log(f"下载进度: {percent:.1f}%")

            # 下载完成，重命名
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)

            log(f"下载完成: {dest_path}")
            return True

        except urllib.error.HTTPError as e:
            log(f"HTTP 错误: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            log(f"网络错误: {e.reason}")
        except Exception as e:
            log(f"下载失败: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        if attempt < retries:
            wait_seconds = min(5 * attempt, 15)
            log(f"{wait_seconds} 秒后重试")
            time.sleep(wait_seconds)

    return False


def _build_mirror_urls(url: str) -> list[str]:
    """构建下载镜像列表（原始地址 + GitHub 镜像）。"""
    urls = [url]
    if "github.com/" in url:
        urls = [
            url,
            f"https://ghfast.top/{url}",
            f"https://gh-proxy.com/{url}",
            f"https://gitcode.com/gh_mirrors/{url.split('github.com/', 1)[1]}",
        ]

    # 去重并保持顺序
    deduped = []
    for u in urls:
        if u not in deduped:
            deduped.append(u)
    return deduped



def _clean_dictionary_file(
    dict_path: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> int:
    """
    清理字典文件中的空行
    MFA 3.x 解析字典时遇到空行会报 IndexError
    
    返回: 清理的空行数量
    """
    try:
        with open(dict_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            lines = f.readlines()
        
        # 过滤空行/注释/异常行，并标准化为 word<TAB>phones...
        cleaned_lines = []
        removed_count = 0
        comment_count = 0
        prob_only_count = 0
        for line in lines:
            stripped = line.replace('\ufeff', '').strip()
            if not stripped:
                removed_count += 1
                continue
            if stripped.startswith('#') or stripped.startswith(';') or stripped.startswith('//'):
                comment_count += 1
                removed_count += 1
                continue
            if len(stripped.split()) < 2:
                removed_count += 1
                continue

            tokens = stripped.split()
            rest = tokens[1:]
            idx = 0
            while idx < len(rest) and idx < 4 and PROB_PATTERN.match(rest[idx]):
                idx += 1
            if idx >= len(rest):
                prob_only_count += 1
                removed_count += 1
                continue

            cleaned_lines.append(f"{tokens[0]}\t{' '.join(tokens[1:])}\n")
        
        # 无论是否移除行，都重写为标准 tab 分隔格式
        with open(dict_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        if progress_callback:
            if removed_count > 0:
                progress_callback(
                    f"已清理 {removed_count} 个空行/无效行（含注释 {comment_count} 行, 概率无音素 {prob_only_count} 行）"
                )
            else:
                progress_callback(f"字典标准化完成，共 {len(cleaned_lines)} 行（已统一为 tab 分隔）")
        
        return removed_count
    except Exception as e:
        logger.warning(f"清理字典文件失败: {e}")
        return 0


def download_acoustic_model(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    force_download: bool = False
) -> tuple[bool, str]:
    """
    下载声学模型
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
        force_download: 强制重新下载
    
    返回:
        (成功标志, 文件路径或错误信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if language not in LANGUAGE_MODELS:
        return False, f"不支持的语言: {language}"
    
    config = LANGUAGE_MODELS[language]["acoustic"]
    url = f"{GITHUB_RELEASE_BASE}/{config['tag']}/{config['filename']}"
    dest_path = os.path.join(output_dir, config["filename"])
    
    # 检查现有文件
    if os.path.exists(dest_path) and not force_download:
        # 简单检查：文件存在且大小大于 1MB
        file_size = os.path.getsize(dest_path)
        if file_size > 1024 * 1024:
            log(f"声学模型已存在: {dest_path}")
            return True, dest_path
        else:
            log(f"声学模型文件异常 (大小: {file_size} bytes)，重新下载...")
    
    for candidate_url in _build_mirror_urls(url):
        if _download_file(candidate_url, dest_path, progress_callback, retries=2, timeout=300):
            return True, dest_path
    return False, "声学模型下载失败"


def download_dictionary(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    force_download: bool = False
) -> tuple[bool, str]:
    """
    下载字典文件（带完整性校验）
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
        force_download: 强制重新下载
    
    返回:
        (成功标志, 文件路径或错误信息)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if language not in LANGUAGE_MODELS:
        return False, f"不支持的语言: {language}"
    
    config = LANGUAGE_MODELS[language]["dictionary"]
    url = f"{GITHUB_RELEASE_BASE}/{config['tag']}/{config['filename']}"
    dest_path = os.path.join(output_dir, config["filename"])
    min_lines = config.get("min_lines")
    
    need_download = force_download
    
    # 检查现有文件完整性
    if os.path.exists(dest_path) and not force_download:
        is_valid, reason = _verify_file_integrity(dest_path, min_lines, progress_callback)
        if is_valid:
            log(f"字典文件已存在且完整: {dest_path}")
            # 确保清理空行
            _clean_dictionary_file(dest_path, progress_callback)
            return True, dest_path
        else:
            log(f"字典文件校验失败: {reason}，重新下载...")
            need_download = True
            # 删除损坏的文件和哈希
            try:
                os.remove(dest_path)
                hash_file = _get_hash_file_path(dest_path)
                if os.path.exists(hash_file):
                    os.remove(hash_file)
            except:
                pass
    else:
        need_download = True
    
    if need_download:
        downloaded = False
        for candidate_url in _build_mirror_urls(url):
            if _download_file(candidate_url, dest_path, progress_callback, retries=2, timeout=300):
                downloaded = True
                break
        if not downloaded:
            return False, "字典文件下载失败"
        
        # 清理空行
        _clean_dictionary_file(dest_path, progress_callback)
        
        # 验证下载的文件
        is_valid, reason = _verify_file_integrity(dest_path, min_lines, progress_callback)
        if not is_valid:
            log(f"下载的字典文件无效: {reason}")
            return False, f"字典文件无效: {reason}"
        
        # 计算并保存哈希值
        file_hash = _calculate_file_hash(dest_path)
        _save_hash(dest_path, file_hash)
        log(f"已保存字典文件哈希: {file_hash[:16]}...")
    
    return True, dest_path


def download_language_models(
    language: str,
    output_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    force_download: bool = False
) -> tuple[bool, str, str]:
    """
    下载指定语言的声学模型和字典
    
    参数:
        language: 语言代码 (mandarin/japanese)
        output_dir: 输出目录
        progress_callback: 进度回调
        force_download: 强制重新下载
    
    返回:
        (成功标志, 声学模型路径, 字典路径)
    """
    def log(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
    
    if language not in LANGUAGE_MODELS:
        return False, "", f"不支持的语言: {language}"
    
    lang_name = LANGUAGE_MODELS[language]["name"]
    log(f"开始下载 {lang_name} 模型...")
    
    # 下载声学模型
    log("=" * 40)
    log("下载声学模型...")
    success, acoustic_path = download_acoustic_model(
        language, output_dir, progress_callback, force_download
    )
    if not success:
        return False, "", acoustic_path
    
    # 下载字典
    log("=" * 40)
    log("下载字典文件...")
    success, dict_path = download_dictionary(
        language, output_dir, progress_callback, force_download
    )
    if not success:
        return False, acoustic_path, dict_path
    
    log("=" * 40)
    log(f"{lang_name} 模型下载完成!")
    return True, acoustic_path, dict_path
