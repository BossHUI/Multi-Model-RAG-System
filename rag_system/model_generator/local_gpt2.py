# -*- coding: utf-8 -*-

from typing import Dict, Any
from transformers import pipeline
import torch
from .base import BaseGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
class LocalGPT2Generator(BaseGenerator):
    """
    @description 使用本地GPT-2模型的生成器
    """
    def __init__(self, model_path="./models/local_gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id


    def generate_answer(self, context: str, question: str, max_length: int = 500) -> str:
        """
        @description 根据上下文和问题生成答案，同时修正 attention_mask 和最大长度问题
        @param context - 上下文
        @param question - 问题
        @param max_length - 希望额外生成的最大 token 数
        @return 生成的答案字符串
        """
        try:
            prompt = f"Question: {question}\nAnswer:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.n_ctx  # 模型的最大上下文长度（通常为1024）
            ).to(self.device)
            
             
            input_length = inputs.input_ids.shape[1]
            # 计算生成时允许的最大总长度，确保不超过模型的 n_ctx
            total_max_length = min(input_length + max_length, self.model.config.n_ctx)
            
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  
                max_length=total_max_length,             # 总长度限制（输入 + 生成）
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"LocalGPT2 Error: {str(e)}")
            return f"Error generating response: {str(e)}"
        
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "local_gpt2",
            "type": "local",
            "context_length": 4096
        }