# -*- coding: utf-8 -*-

import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from langchain.schema import Document


class BaseLoader:
    """
    @description 基础数据加载器
    """
    def load(self, limit: Optional[int] = None) -> List[Document]:
        raise NotImplementedError
    
class ArxivLoader:
    """
    @description Arxiv论文数据加载器
    """
    def __init__(self, file_path, max_samples=1000):
        self.file_path = file_path
        self.max_samples = max_samples  # 控制初始样本量

    def stream_documents(self):
        """流式读取大型JSON文件"""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Arxiv Data"):
                if count >= self.max_samples:
                    break
                yield json.loads(line)
                count += 1
    
    def load(self, limit=None): 
        """加载指定数量的论文数据"""
        papers = []
        for doc in self.stream_documents():
            papers.append(doc)
            if limit and len(papers) >= limit:
                break
        return papers
    
    def to_dataframe(self):
        """转换为DataFrame格式"""
        records = []
        for doc in self.stream_documents():
            record = {
                "paper_id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "authors": ", ".join(doc.get("authors", [])),
                "categories": doc.get("categories", []),
                "full_text": f"Title: {doc['title']}\nAbstract: {doc['abstract']}"
            }
            records.append(record)
        return pd.DataFrame(records)
    

class CSVLoader(BaseLoader):
    """
    @description CSV文件加载器
    """
    def __init__(self, file_path: str):
        """
        @description 初始化CSV加载器
        @param file_path - CSV文件路径
        """
        self.file_path = file_path

    def load(self, limit: int = None) -> List[Document]:
        """
        @description 加载电动车CSV数据并转换为文档格式
        @param limit - 限制加载的数据条数
        @return Document列表
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(self.file_path)
            if limit:
                df = df.head(limit)
            
            documents = []
            # 将每行数据转换为Document对象
            for _, row in df.iterrows():
                # 构建结构化的文本内容
                content = f"""
                    Vehicle Information:
                    - VIN : {row['VIN (1-10)']}
                    - Year: {row['Model Year']}
                    - Make: {row['Make']}
                    - Model: {row['Model']}
                    - EV Type: {row['Electric Vehicle Type']}
                    - Electric Range: {row['Electric Range']} miles
                    - Base MSRP: ${row['Base MSRP']}

                    Location Information:
                    - County: {row['County']}
                    - City: {row['City']}
                    - State: {row['State']}
                    - Postal Code: {row['Postal Code']}
                    - Legislative District: {row['Legislative District']}
                    - Census Tract: {row['2020 Census Tract']}

                    Additional Details:
                    - CAFV Eligibility: {row['Clean Alternative Fuel Vehicle (CAFV) Eligibility']}
                    - Electric Utility: {row['Electric Utility']}
                    - DOL Vehicle ID: {row['DOL Vehicle ID']}
                    - Vehicle Location: {row['Vehicle Location']}
                    """

                # 创建Document对象，包含完整的元数据
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        'vin': row['VIN (1-10)'],
                        'make': row['Make'],
                        'model': row['Model'],
                        'year': row['Model Year'],
                        'ev_type': row['Electric Vehicle Type'],
                        'city': row['City'],
                        'state': row['State'],
                        'postal_code': row['Postal Code'],
                        'electric_range': row['Electric Range'],
                        'cafv_eligible': row['Clean Alternative Fuel Vehicle (CAFV) Eligibility'],
                        'base_msrp': row['Base MSRP'],
                        'source': self.file_path
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error loading EV CSV file: {e}")
            return []