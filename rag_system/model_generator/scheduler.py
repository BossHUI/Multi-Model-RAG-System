# -*- coding: utf-8 -*-

from typing import List, Dict, Any
from .base import BaseGenerator
from .factory import GeneratorFactory

class GeneratorScheduler:
    """
    @description 生成器调度器，负责模型的选择和切换
    """
    def __init__(self, default_model: str = "local_gpt2"):
        self.current_model = default_model
        self.generator = GeneratorFactory.create(default_model)
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self.fallback_order = ["local_gpt2", "chatGLM", "deepseek", "gpt4o", "claude3.5", "qwen"]

    def switch_model(self, model_name: str) -> None:
        if model_name != self.current_model:
            self.current_model = model_name
            self.generator = GeneratorFactory.create(model_name)

    def _try_fallback_models(self, context: str, question: str, **kwargs) -> str:
        for model in self.fallback_order:
            if model != self.current_model:
                try:
                    self.switch_model(model)
                    return self.generator.generate_answer(context, question, **kwargs)
                except Exception as e:
                    print(f"Fallback to {model} failed: {e}")
                    continue
        raise Exception("All models failed to generate answer")

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        try:
            return self.generator.generate_answer(context, question, **kwargs)
        except Exception as e:
            print(f"Error with {self.current_model}: {e}")
            return self._try_fallback_models(context, question, **kwargs)