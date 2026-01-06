# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseGenerator(ABC):
    """
    @description 生成器基类，定义所有生成器必须实现的接口
    """
    @abstractmethod
    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        """生成答案的抽象方法"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass