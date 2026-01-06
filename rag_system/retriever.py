# -*- coding: utf-8 -*-

from langchain_community.vectorstores import FAISS
from typing import List
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

class Retriever:
    def __init__(self, embedder: Embeddings, documents: List[Document]):
        """
        @description 初始化检索器
        @param embedder - 实现了 LangChain Embeddings 接口的嵌入模型
        @param documents - 文档列表
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedder
        )
        
    def get_retriever(self, k=3):
        return self.faiss_retriever.as_retriever(search_kwargs={"k": k})
    
    def get_relevant_documents(self, question: str, k: int = 3) -> List[Document]:
        """
        @description 根据问题检索相关文档
        @param {str} question - 用户输入的问题
        @param {int} k - 返回的相关文档数量
        @return {List[Document]} - 相关文档列表
        """
        return self.vector_store.similarity_search(question, k=k)