# -*- coding: utf-8 -*-

from typing import Dict, Type
from .base import BaseGenerator
from .deepseek import DeepseekGenerator
from .gpt4o import GPT4Generator
from .chatglm import ChatGLMGenerator
from .local_gpt2 import LocalGPT2Generator
from .claude3 import Claude3Generator
from .qwen import QwenGenerator


class GeneratorFactory:
    """
    @description 生成器工厂类，负责创建和管理不同的生成器实例
    """
    _generators: Dict[str, Type[BaseGenerator]] = {
        "deepseek": DeepseekGenerator,
        "gpt4o": GPT4Generator,
        "chatGLM": ChatGLMGenerator,
        "local_gpt2": LocalGPT2Generator,
        "claude3.5": Claude3Generator,
        "qwen": QwenGenerator,
    }

    @classmethod
    def create(cls, model_name: str) -> BaseGenerator:
        """
        @description 创建指定模型的生成器实例
        @param model_name - 模型名称
        @return 生成器实例
        """
        if model_name not in cls._generators:
            raise ValueError(f"Unknown model: {model_name}")
        return cls._generators[model_name]()