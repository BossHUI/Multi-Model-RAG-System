# -*- coding: utf-8 -*-

import os
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from .base import BaseGenerator

class DeepseekGenerator(BaseGenerator):
    """
    @description 使用Deepseek API的生成器
    """
    def __init__(self):

        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        
    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL_ID"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Deepseek API Error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "deepseek",
            "type": "api",
            "context_length": 8192
        }