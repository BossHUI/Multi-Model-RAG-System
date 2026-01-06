# -*- coding: utf-8 -*-

import faiss
import json
import os
from langchain.schema import Document

def save_rag_system(index, chunks, save_dir="processed_data", domain="arxiv"):
    """
    @description 保存处理后的RAG系统数据
    @param domain - 数据领域，如 'arxiv' 或 'ev'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存FAISS索引
    faiss.write_index(index, f"{save_dir}/{domain}/{domain}_index.faiss")
    
    # 保存文档chunks的元数据
    chunks_data = [{
        "content": chunk.page_content,
        "metadata": chunk.metadata
    } for chunk in chunks]
    
    with open(f"{save_dir}/{domain}/{domain}_chunks_metadata.json", "w") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

def load_rag_system(load_dir="processed_data", domain="arxiv"):
    """
    @description 加载保存的RAG系统数据
    @param domain - 数据领域，如 'arxiv' 或 'ev'
    """
    # 检查文件是否存在
    index_path = f"{load_dir}/{domain}/{domain}_index.faiss"
    metadata_path = f"{load_dir}/{domain}/{domain}_chunks_metadata.json"
    # print(index_path)
    # print(metadata_path)
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"did not find {domain} domain data files")
    
    # 加载FAISS索引
    index = faiss.read_index(index_path)
    
    # 加载文档chunks
    with open(metadata_path, "r") as f:
        chunks_data = json.load(f)
    
    chunks = [Document(
        page_content=item["content"],
        metadata=item["metadata"]
    ) for item in chunks_data]
    
    return index, chunks

def save_ev_data(index, chunks):
    """
    @description 保存处理后的电动车数据
    """
    save_rag_system(index, chunks, domain="ev")

def load_ev_data():
    """
    @description 加载保存的电动车数据
    """
    return load_rag_system(domain="ev")