# -*- coding: utf-8 -*-
"""
导出插件加载器

负责扫描和加载内置及外部插件
"""

import os
import sys
import logging
import importlib.util
from typing import Dict, List, Type

from .base import ExportPlugin
from .simple_export import SimpleExportPlugin

logger = logging.getLogger(__name__)


def get_builtin_plugins() -> List[Type[ExportPlugin]]:
    """获取内置插件列表"""
    return [SimpleExportPlugin]


def load_plugins(plugins_dir: str = None) -> Dict[str, ExportPlugin]:
    """
    加载所有插件
    
    参数:
        plugins_dir: 外部插件目录路径，默认为 export_plugins 同级目录
    
    返回:
        {插件名称: 插件实例} 字典
    """
    plugins: Dict[str, ExportPlugin] = {}
    
    # 加载内置插件
    for plugin_cls in get_builtin_plugins():
        try:
            instance = plugin_cls()
            plugins[instance.name] = instance
            logger.info(f"加载内置插件: {instance.name}")
        except Exception as e:
            logger.error(f"加载内置插件失败: {plugin_cls.__name__}, {e}")
    
    # 加载外部插件
    if plugins_dir and os.path.exists(plugins_dir):
        for filename in os.listdir(plugins_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                plugin_path = os.path.join(plugins_dir, filename)
                try:
                    plugin = _load_plugin_from_file(plugin_path)
                    if plugin:
                        plugins[plugin.name] = plugin
                        logger.info(f"加载外部插件: {plugin.name} ({filename})")
                except Exception as e:
                    logger.error(f"加载外部插件失败: {filename}, {e}")
    
    return plugins


def _load_plugin_from_file(filepath: str) -> ExportPlugin:
    """
    从文件加载插件
    
    参数:
        filepath: 插件文件路径
    
    返回:
        插件实例，加载失败返回None
    """
    try:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # 查找 ExportPlugin 子类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, ExportPlugin) and 
                attr is not ExportPlugin):
                return attr()
        
        return None
        
    except Exception as e:
        logger.error(f"加载插件文件失败: {filepath}, {e}", exc_info=True)
        return None
