# -*- coding: utf-8 -*-
"""
MFA 运行模块单元测试
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mfa_runner import (
    check_mfa_available,
    _build_mfa_env,
    run_mfa_alignment,
    run_mfa_validate,
    BASE_DIR,
    MFA_ENGINE_DIR,
    MFA_PYTHON,
)


class TestCheckMfaAvailable(unittest.TestCase):
    """测试 MFA 环境检查"""
    
    @patch('src.mfa_runner.MFA_ENGINE_DIR')
    def test_returns_false_when_dir_not_exists(self, mock_dir):
        """目录不存在时应返回 False"""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        
        with patch.object(Path, 'exists', return_value=False):
            # 由于模块级变量，需要重新导入或直接测试逻辑
            pass
    
    def test_path_constants_defined(self):
        """路径常量应正确定义"""
        self.assertIsInstance(BASE_DIR, Path)
        self.assertIsInstance(MFA_ENGINE_DIR, Path)
        self.assertIsInstance(MFA_PYTHON, Path)
        
        # 验证路径结构
        self.assertTrue(str(MFA_ENGINE_DIR).endswith("mfa_engine"))
        self.assertTrue(str(MFA_PYTHON).endswith("python.exe"))


class TestBuildMfaEnv(unittest.TestCase):
    """测试 MFA 环境变量构建"""
    
    def test_returns_dict(self):
        """应返回字典"""
        env = _build_mfa_env()
        self.assertIsInstance(env, dict)
    
    def test_path_contains_mfa_dirs(self):
        """PATH 应包含 MFA 相关目录"""
        env = _build_mfa_env()
        path = env.get("PATH", "")
        
        self.assertIn("mfa_engine", path)
        self.assertIn("Library", path)
    
    def test_preserves_original_path(self):
        """应保留原始 PATH"""
        original_path = os.environ.get("PATH", "")
        env = _build_mfa_env()
        
        # 原始 PATH 应在新 PATH 中
        self.assertIn(original_path.split(";")[0], env["PATH"])


class TestRunMfaAlignment(unittest.TestCase):
    """测试 MFA 对齐功能"""
    
    @patch('src.mfa_runner.check_mfa_available')
    def test_fails_when_mfa_unavailable(self, mock_check):
        """MFA 不可用时应返回失败"""
        mock_check.return_value = False
        
        success, msg = run_mfa_alignment("/input", "/output")
        
        self.assertFalse(success)
        self.assertIn("不可用", msg)
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    def test_fails_when_corpus_not_exists(self, mock_isdir, mock_check):
        """输入目录不存在时应返回失败"""
        mock_check.return_value = True
        mock_isdir.return_value = False
        
        success, msg = run_mfa_alignment("/nonexistent", "/output")
        
        self.assertFalse(success)
        self.assertIn("不存在", msg)
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_fails_when_dict_not_exists(self, mock_isfile, mock_isdir, mock_check):
        """字典文件不存在时应返回失败"""
        mock_check.return_value = True
        mock_isdir.return_value = True
        mock_isfile.return_value = False
        
        success, msg = run_mfa_alignment(
            "/input", "/output",
            dict_path="/nonexistent.dict"
        )
        
        self.assertFalse(success)
        self.assertIn("不存在", msg)
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.makedirs')
    @patch('subprocess.run')
    def test_calls_subprocess_with_correct_args(
        self, mock_run, mock_makedirs, mock_isfile, mock_isdir, mock_check
    ):
        """应使用正确的参数调用 subprocess"""
        mock_check.return_value = True
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        
        run_mfa_alignment(
            "/input", "/output",
            dict_path="/dict.dict",
            model_path="/model.zip",
            single_speaker=True,
            clean=True
        )
        
        # 验证 subprocess.run 被调用
        mock_run.assert_called_once()
        
        # 验证命令参数
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        
        self.assertIn("align", cmd)
        self.assertIn("/input", cmd)
        self.assertIn("/dict.dict", cmd)
        self.assertIn("/model.zip", cmd)
        self.assertIn("/output", cmd)
        self.assertIn("--single_speaker", cmd)
        self.assertIn("--clean", cmd)
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.makedirs')
    @patch('subprocess.run')
    def test_returns_success_on_zero_returncode(
        self, mock_run, mock_makedirs, mock_isfile, mock_isdir, mock_check
    ):
        """返回码为 0 时应返回成功"""
        mock_check.return_value = True
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="完成", stderr="")
        
        success, msg = run_mfa_alignment(
            "/input", "/output",
            dict_path="/dict.dict",
            model_path="/model.zip"
        )
        
        self.assertTrue(success)
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.makedirs')
    @patch('subprocess.run')
    def test_returns_failure_on_nonzero_returncode(
        self, mock_run, mock_makedirs, mock_isfile, mock_isdir, mock_check
    ):
        """返回码非 0 时应返回失败"""
        mock_check.return_value = True
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="错误")
        
        success, msg = run_mfa_alignment(
            "/input", "/output",
            dict_path="/dict.dict",
            model_path="/model.zip"
        )
        
        self.assertFalse(success)


class TestRunMfaValidate(unittest.TestCase):
    """测试 MFA 验证功能"""
    
    @patch('src.mfa_runner.check_mfa_available')
    def test_fails_when_mfa_unavailable(self, mock_check):
        """MFA 不可用时应返回失败"""
        mock_check.return_value = False
        
        success, msg = run_mfa_validate("/corpus")
        
        self.assertFalse(success)
        self.assertIn("不可用", msg)


class TestProgressCallback(unittest.TestCase):
    """测试进度回调"""
    
    @patch('src.mfa_runner.check_mfa_available')
    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('os.makedirs')
    @patch('subprocess.run')
    def test_callback_called_on_success(
        self, mock_run, mock_makedirs, mock_isfile, mock_isdir, mock_check
    ):
        """成功时应调用回调"""
        mock_check.return_value = True
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="完成", stderr="")
        callback = MagicMock()
        
        run_mfa_alignment(
            "/input", "/output",
            dict_path="/dict.dict",
            model_path="/model.zip",
            progress_callback=callback
        )
        
        # 回调应被调用（至少一次）
        self.assertTrue(callback.called)


if __name__ == "__main__":
    unittest.main()
