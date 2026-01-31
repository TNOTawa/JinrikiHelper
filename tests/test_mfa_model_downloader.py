# -*- coding: utf-8 -*-
"""
MFA 模型下载模块单元测试
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mfa_model_downloader import (
    get_available_languages,
    LANGUAGE_MODELS,
    GITHUB_RELEASE_BASE,
    download_acoustic_model,
    download_dictionary,
    download_language_models,
)


class TestGetAvailableLanguages(unittest.TestCase):
    """测试获取可用语言列表"""
    
    def test_returns_dict(self):
        """返回值应为字典"""
        result = get_available_languages()
        self.assertIsInstance(result, dict)
    
    def test_contains_mandarin(self):
        """应包含中文"""
        result = get_available_languages()
        self.assertIn("mandarin", result)
        self.assertEqual(result["mandarin"], "中文 (普通话)")
    
    def test_contains_japanese(self):
        """应包含日文"""
        result = get_available_languages()
        self.assertIn("japanese", result)
        self.assertEqual(result["japanese"], "日文")


class TestLanguageModelsConfig(unittest.TestCase):
    """测试语言模型配置"""
    
    def test_mandarin_config_complete(self):
        """中文配置应完整"""
        config = LANGUAGE_MODELS["mandarin"]
        self.assertIn("name", config)
        self.assertIn("acoustic", config)
        self.assertIn("dictionary", config)
        
        # 声学模型配置
        acoustic = config["acoustic"]
        self.assertIn("tag", acoustic)
        self.assertIn("filename", acoustic)
        self.assertTrue(acoustic["filename"].endswith(".zip"))
        
        # 字典配置
        dictionary = config["dictionary"]
        self.assertIn("tag", dictionary)
        self.assertIn("filename", dictionary)
        self.assertTrue(dictionary["filename"].endswith(".dict"))
    
    def test_japanese_config_complete(self):
        """日文配置应完整"""
        config = LANGUAGE_MODELS["japanese"]
        self.assertIn("name", config)
        self.assertIn("acoustic", config)
        self.assertIn("dictionary", config)
    
    def test_acoustic_url_format(self):
        """声学模型 URL 格式应正确"""
        for lang, config in LANGUAGE_MODELS.items():
            acoustic = config["acoustic"]
            url = f"{GITHUB_RELEASE_BASE}/{acoustic['tag']}/{acoustic['filename']}"
            self.assertTrue(url.startswith("https://github.com/"))
            self.assertIn("mfa-models", url)
    
    def test_dictionary_url_format(self):
        """字典 URL 格式应正确"""
        for lang, config in LANGUAGE_MODELS.items():
            dictionary = config["dictionary"]
            url = f"{GITHUB_RELEASE_BASE}/{dictionary['tag']}/{dictionary['filename']}"
            self.assertTrue(url.startswith("https://github.com/"))
            self.assertIn("dictionary-", url)


class TestDownloadAcousticModel(unittest.TestCase):
    """测试声学模型下载"""
    
    def test_invalid_language(self):
        """不支持的语言应返回失败"""
        success, result = download_acoustic_model("invalid_lang", "/tmp")
        self.assertFalse(success)
        self.assertIn("不支持的语言", result)
    
    @patch('src.mfa_model_downloader._download_file')
    def test_download_called_with_correct_url(self, mock_download):
        """应使用正确的 URL 下载"""
        mock_download.return_value = True
        
        with patch('os.path.exists', return_value=False):
            download_acoustic_model("mandarin", "/tmp/models")
        
        # 验证调用参数
        call_args = mock_download.call_args
        url = call_args[0][0]
        self.assertIn("mandarin_mfa.zip", url)
        self.assertIn("acoustic-mandarin_mfa", url)
    
    @patch('os.path.exists')
    def test_skip_if_exists(self, mock_exists):
        """文件已存在时应跳过下载"""
        mock_exists.return_value = True
        
        success, result = download_acoustic_model("mandarin", "/tmp/models")
        self.assertTrue(success)
        self.assertIn("mandarin_mfa.zip", result)


class TestDownloadDictionary(unittest.TestCase):
    """测试字典下载"""
    
    def test_invalid_language(self):
        """不支持的语言应返回失败"""
        success, result = download_dictionary("invalid_lang", "/tmp")
        self.assertFalse(success)
        self.assertIn("不支持的语言", result)
    
    @patch('src.mfa_model_downloader._download_file')
    def test_download_called_with_correct_url(self, mock_download):
        """应使用正确的 URL 下载"""
        mock_download.return_value = True
        
        with patch('os.path.exists', return_value=False):
            download_dictionary("japanese", "/tmp/models")
        
        call_args = mock_download.call_args
        url = call_args[0][0]
        self.assertIn("github.com", url)
        self.assertIn("dictionary-japanese", url)


class TestDownloadLanguageModels(unittest.TestCase):
    """测试完整语言模型下载"""
    
    def test_invalid_language(self):
        """不支持的语言应返回失败"""
        success, acoustic, dict_path = download_language_models("invalid", "/tmp")
        self.assertFalse(success)
    
    @patch('src.mfa_model_downloader.download_dictionary')
    @patch('src.mfa_model_downloader.download_acoustic_model')
    def test_downloads_both_models(self, mock_acoustic, mock_dict):
        """应同时下载声学模型和字典"""
        mock_acoustic.return_value = (True, "/tmp/acoustic.zip")
        mock_dict.return_value = (True, "/tmp/dict.dict")
        
        success, acoustic, dict_path = download_language_models("mandarin", "/tmp")
        
        self.assertTrue(success)
        mock_acoustic.assert_called_once()
        mock_dict.assert_called_once()
    
    @patch('src.mfa_model_downloader.download_dictionary')
    @patch('src.mfa_model_downloader.download_acoustic_model')
    def test_stops_on_acoustic_failure(self, mock_acoustic, mock_dict):
        """声学模型下载失败时应停止"""
        mock_acoustic.return_value = (False, "下载失败")
        
        success, _, _ = download_language_models("mandarin", "/tmp")
        
        self.assertFalse(success)
        mock_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
