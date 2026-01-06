# -*- coding: utf-8 -*-

import os
import zhipuai
from typing import Dict, Any
from dotenv import load_dotenv
from .base import BaseGenerator
from zhipuai import ZhipuAI

class ChatGLMGenerator(BaseGenerator):
    """
    @description 使用ChatGLM API的生成器
    """
    def __init__(self):
        load_dotenv()
        self.client = ZhipuAI(api_key=os.getenv("CHATGLM_API_KEY"))
        
    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        try:

            response = self.client.chat.completions.create(
                model=os.getenv("GLM_MODEL_ID"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                ],
                temperature=0.7,
                top_p=0.7,
                max_tokens=8192,
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"ChatGLM API Error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "chatglm",
            "type": "api",
            "context_length": 8192
        }