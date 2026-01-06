# -*- coding: utf-8 -*-

from transformers import AutoModel, AutoTokenizer
import os
from transformers import AutoModelForCausalLM


model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # 添加信任远程代码参数
    mirror="https://hf-mirror.com"  # 显式指定镜像源
)
model_name = "openai-community/gpt2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./local_gpt2")
tokenizer.save_pretrained("./local_gpt2")