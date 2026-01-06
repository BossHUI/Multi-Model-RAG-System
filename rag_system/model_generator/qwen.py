# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from .base import BaseGenerator

class QwenGenerator(BaseGenerator):
    """
    @description 使用Qwen API的生成器
    """
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        try:
            # 添加重试机制和延迟
            max_retries = 3
            retry_delay = 2  
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=os.getenv("QWEN_MODEL_ID"),
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    return response.choices[0].message.content
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed, waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    
        except Exception as e:
            print(f"Qwen API Error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "qwen",
            "type": "api",
            "context_length": 8192
        }