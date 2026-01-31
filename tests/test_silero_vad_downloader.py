# -*- coding: utf-8 -*-
"""
Silero VAD 下载模块测试
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.silero_vad_downloader import (
    get_vad_model_path,
    is_vad_model_downloaded,
    download_silero_vad,
    ensure_vad_model,
    SILERO_VAD_CONFIG
)


class TestSileroVadDownloader(unittest.TestCase):
    """Silero VAD 下载器测试类"""
    
    def test_get_vad_model_path(self):
        """测试获取模型路径"""
        models_dir = "/test/models"
        expected = os.path.join(models_dir, "silero_vad", "silero_vad.onnx")
        self.assertEqual(get_vad_model_path(models_dir), expected)
    
    def test_is_vad_model_downloaded_false(self):
        """测试模型未下载时返回 False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(is_vad_model_downloaded(tmpdir))
    
    def test_is_vad_model_downloaded_true(self):
        """测试模型已下载时返回 True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vad_dir = os.path.join(tmpdir, "silero_vad")
            os.makedirs(vad_dir)
            model_path = os.path.join(vad_dir, "silero_vad.onnx")
            with open(model_path, "w") as f:
                f.write("dummy")
            self.assertTrue(is_vad_model_downloaded(tmpdir))
    
    def test_download_silero_vad_already_exists(self):
        """测试模型已存在时跳过下载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vad_dir = os.path.join(tmpdir, "silero_vad")
            os.makedirs(vad_dir)
            model_path = os.path.join(vad_dir, "silero_vad.onnx")
            with open(model_path, "w") as f:
                f.write("dummy")
            
            success, result = download_silero_vad(tmpdir)
            self.assertTrue(success)
            self.assertEqual(result, model_path)
    
    def test_config_values(self):
        """测试配置值正确性"""
        self.assertEqual(SILERO_VAD_CONFIG["onnx_filename"], "silero_vad.onnx")
        self.assertEqual(SILERO_VAD_CONFIG["jit_filename"], "silero_vad.jit")
        self.assertIn("snakers4/silero-vad", SILERO_VAD_CONFIG["repo"])


if __name__ == "__main__":
    unittest.main()
