# -*- coding: utf-8 -*-

import os
from typing import Dict, Any
from dotenv import load_dotenv
from .base import BaseGenerator
from openai import OpenAI

class Claude3Generator(BaseGenerator):
    """
    @description 使用Claude 3.5 Sonnet API的生成器
    """
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("CLAUDE_API_KEY"),base_url=os.getenv("CLAUDE_BASE_URL"))

    def generate_answer(self, context: str, question: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("CLAUDE_MODEL_ID"),
                max_tokens=8192,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Claude 3.5 API Error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "claude-3-5-sonnet-20241022",
            "type": "api",
            "context_length": 8192
        }
