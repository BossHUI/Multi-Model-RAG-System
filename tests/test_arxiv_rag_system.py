# -*- coding: utf-8 -*-

import logging
import numpy as np
from rag_system.data_loader import ArxivLoader
from rag_system.preprocessor import DataProcessor
from rag_system.retriever import Retriever
from rag_system.generator import AnswerGenerator
from rag_system.visualization import visualize_retrieval_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_components():
    """
    @description test the components of RAG system
    """
    try:
        print("1. Test data loading...")
        loader = ArxivLoader("data/Arxiv/arxiv-metadata-oai-snapshot.json")
        papers = loader.load(limit=100)  # 先用100篇论文测试
        print(f"Successfully loaded the {len(papers)} paper")

        print("\n2. Test data processing...")
        processor = DataProcessor()
        index, chunks = processor.process_arxiv_papers(papers)
        print(f"{len(chunks)} blocks of text are generated")

        print("\n3. Test retrieval...")
        test_query = "What is machine learning?"
        query_embedding = np.array(processor.embedder.embed_query(test_query))
        D, I = index.search(query_embedding.reshape(1, -1), k=3)
        print(f"The most relevant was successfully retrieved {len(I[0])} block")

        print("\n4. Visual search results...")
        retriever = Retriever(processor.embedder, chunks)
        relevant_docs = retriever.get_relevant_documents(test_query)
        visualize_retrieval_results(relevant_docs, test_query)

        print("\n5. Quiz Full Q&A...")
        question = "What are the recent developments in transformer architecture?"
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in relevant_docs])

        generator = AnswerGenerator()
        answer = generator.generate_answer(context, question)
        print(f"question: {question}")
        print(f"answer: {answer}")
        
        return True
    except Exception as e:
        logger.error(f"Testing RAG system failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_rag_components()