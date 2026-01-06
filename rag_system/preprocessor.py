# -*- coding: utf-8 -*-

import faiss, os, json
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from rag_system.data_loader import ArxivLoader
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel


class EmbeddingFactory:
    """支持多种向量化方法的工厂类"""
    @staticmethod
    def get_embedder(embedding_type: str = "sentence-transformers", model_name: str = None):
        if embedding_type == "sentence-transformers":
            model = SentenceTransformer(model_name or "all-MiniLM-L6-v2")
            return lambda text: model.encode(text)
        elif embedding_type == "openai":
            
            client = OpenAI()
            return lambda text: client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            ).data[0].embedding
        elif embedding_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_name or "sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained(model_name or "sentence-transformers/all-MiniLM-L6-v2")
            return lambda text: model(**tokenizer(text, return_tensors="pt")).last_hidden_state.mean(dim=1).detach().numpy()
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        

class SentenceTransformerEmbeddings(Embeddings):
    """
    @description 封装 SentenceTransformer 以适配 LangChain 接口
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embedding_type: str = "sentence-transformers"):
        # self.model = SentenceTransformer(model_name)
        self.embedder = EmbeddingFactory.get_embedder(embedding_type, model_name)
        self.embedding_type = embedding_type

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        @description 将文档转换为向量
        @param texts - 文档列表
        @return 向量列表
        """
        embeddings = []
        for text in tqdm(texts, desc="Encoding documents"):
            # embeddings.append(self.model.encode(text))
            embeddings.append(self.embedder(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        @description 将查询转换为向量
        @param text - 查询文本
        @return 向量
        """
        # embedding = self.model.encode(text)
        embedding = self.embedder([text])[0]
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        @description 将文档转换为向量（与原始代码兼容）
        @param texts - 文档列表
        @return 向量列表
        """
        return self.embed_documents(texts)

    def encode_query(self, text: str) -> List[float]:
        """
        @description 将查询转换为向量（与原始代码兼容）
        @param text - 查询文本
        @return 向量
        """
        return self.embed_query(text) 
    

class DataProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformerEmbeddings(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)
        
        # 生成嵌入向量
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, chunks
    
    def process_arxiv_papers(self, papers: List[Dict]) -> Tuple[faiss.Index, List[Document]]:
        """
        @description 处理Arxiv论文数据
        @param {List[ArxivPaper]} papers - 论文列表
        @return {Tuple[faiss.Index, List[Document]]} - FAISS索引和文档列表
        """
        documents = []
        for paper in papers:
            # 将标题和摘要组合成文档
            text = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"
            doc = Document(
                page_content=text,
                metadata={
                    "title": paper.get('title', ''),
                    "authors": paper.get('authors', []),
                    "categories": paper.get('categories', '')
                }
            )
            documents.append(doc)
        
        # 分块处理
        chunks = self.text_splitter.split_documents(documents)
        
        # 生成嵌入向量
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        embeddings = np.array(embeddings)  
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 保存元数据到 chunks_metadata.json
        chunks_metadata = [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]
        with open("processed_data/chunks_metadata.json", "w") as f:
            json.dump(chunks_metadata, f)
        
        return index, chunks
    
    def process_documents(self, documents: List[Document]) -> Tuple[faiss.Index, List[Document]]:
        doc_chunks = []
        for doc in documents:
            # 处理字典格式的输入
            if isinstance(doc, dict):
                page_content = doc.get('text', doc.get('abstract', ''))
                metadata = doc.get('metadata', {})
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
            
            # 确保是 Document 对象
            if not isinstance(doc, Document):
                raise ValueError(f"Expected Document object, got {type(doc)}")
                
            chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                doc_chunks.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))

        embeddings = self.embedder.encode([d.page_content for d in doc_chunks])
        embeddings = np.array(embeddings)
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, doc_chunks
    

class ArxivProcessor(DataProcessor):
    def process_arxiv(self, df):
        """
        输入：包含full_text列的DataFrame
        输出：(FAISS索引, 文档块列表)
        """
        # 专业领域文本分块策略
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 适应论文摘要长度
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for _, row in df.iterrows():
            chunks.extend(text_splitter.create_documents(
                [row['full_text']],
                metadatas=[{
                    "paper_id": row['paper_id'],
                    "categories": row['categories']
                }]
            ))
        
        # 生成嵌入时增加领域适配
        texts = [f"Category: {chunk.metadata['categories']}\n{chunk.page_content}" 
                for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        embeddings = np.array(embeddings)
        
        # 创建带元数据的FAISS索引
        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array([i for i in range(len(chunks))]))
        
        return index, chunks