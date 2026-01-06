#-*-- coding:utf-8 --*--

from transformers import pipeline
import torch
from .model_generator.scheduler import GeneratorScheduler
from typing import Optional
from typing import List

class MultiModelGenerator:
    def __init__(self):
        self.models = {
            "gpt4o": 0.3,
            "claude3.5":0.2,
            "chatGLM": 0.2,
            "deepseek": 0.2,
            "local_gpt2": 0.1
        }
        self.generators = {name: AnswerGenerator() for name in self.models}

    def weighted_average(self, answers: List[str]) -> str:
        from collections import Counter
        # 简单的加权投票机制
        weighted_counts = Counter()
        for answer, weight in zip(answers, self.models.values()):
            weighted_counts[answer] += weight
        
        return weighted_counts.most_common(1)[0][0]

    def generate_answer(self, context: str, question: str, domain: str = "arxiv" , **kwargs) -> str:
        answers = []
        for model_name, generator in self.generators.items():
            try:
                answer = generator.generate_answer(context, question, model_name, domain, **kwargs)
                answers.append(answer)
            except Exception as e:
                print(f"Error generating answer with {model_name}: {e}")
        
        if not answers:
            raise ValueError("All models failed to generate answers")
        
        return self.weighted_average(answers)
    

class AnswerGenerator:
    def __init__(self, model_name: str = "local_gpt2"):
        self.scheduler = GeneratorScheduler(default_model=model_name)
        self.prompt_templates = {
            "gpt4o": "You are an expert in {domain}. Given the following context:\n{context}\n\nAnswer the question: {question}",
            "chatglm": "[INST] <<SYS>>\nYou are a helpful assistant in {domain}.\n<</SYS>>\n\nContext: {context}\n\nQuestion: {question} [/INST]",
            "deepseek": "### System: You are an AI assistant specialized in {domain}.\n### Context: {context}\n### Question: {question}\n### Answer:",
            "local_gpt2": "Context: {context}\nQuestion: {question}\nAnswer:",
            "claude3.5": "\n\nHuman: I need help with a {domain} question. Here's the context: {context}\n\nQuestion: {question}\n\nAssistant:",
            "qwen": "<|im_start|>system\nYou are an AI assistant specialized in {domain}.<|im_end|>\n<|im_start|>user\nContext: {context}\nQuestion: {question}<|im_end|>\n<|im_start|>assistant"
        }
    
    def generate_answer(self, context: str, question: str, 
                       model: Optional[str] = None, domain: str = "arxiv", **kwargs) -> str:
        """
        @description 生成答案
        @param context - 上下文
        @param question - 问题
        @param model - 指定使用的模型（可选）
        @return 生成的答案
        """
        template = self.prompt_templates.get(model, self.prompt_templates["gpt4o"])
        prompt = template.format(domain=domain, context=context, question=question)
        if model and model != self.scheduler.current_model:
            self.scheduler.switch_model(model)
        return self.scheduler.generate_answer(context, question, **kwargs)
