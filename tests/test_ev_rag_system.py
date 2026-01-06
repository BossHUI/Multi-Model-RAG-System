# -*- coding: utf-8 -*-

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from rag_system.data_loader import CSVLoader
from rag_system.preprocessor import DataProcessor
from rag_system.retriever import Retriever
from rag_system.generator import AnswerGenerator
from rag_system.utils import load_ev_data, save_ev_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_ev_distribution(docs, save_path):
    """
    @description visualize the distribution of electric vehicles data
    """
    # 提取数据
    data = []
    for doc in docs:
        data.append({
            'make': doc.metadata['make'],
            'model': doc.metadata['model'],
            'year': doc.metadata['year'],
            'ev_type': doc.metadata['ev_type'],
            'electric_range': float(doc.metadata['electric_range']),
            'base_msrp': float(doc.metadata['base_msrp'])
        })
    
    df = pd.DataFrame(data)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.countplot(data=df, y='make', ax=ax1, order=df['make'].value_counts().index[:10])
    ax1.set_title('Top 10 EV Manufacturers')
    
    sns.countplot(data=df, x='ev_type', ax=ax2)
    ax2.set_title('EV Type Distribution')
    
    sns.histplot(data=df, x='electric_range', ax=ax3)
    ax3.set_title('Electric Range Distribution')
    
    sns.boxplot(data=df, x='make', y='base_msrp', ax=ax4, order=df['make'].value_counts().index[:10])
    ax4.set_title('Price Distribution by Manufacturer')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_retrieval_quality(retriever, test_questions):
    """
    @description test the retrieval quality of the RAG system
    """
    results = []
    for question in test_questions:
        relevant_docs = retriever.get_relevant_documents(question)
        results.append({
            'question': question,
            'num_docs': len(relevant_docs),
            'top_doc_content': relevant_docs[0].page_content if relevant_docs else None
        })
    return pd.DataFrame(results)

def test_ev_rag_system():
    """
    @description test the RAG system for electric vehicles

    """
    results_dir = Path("results/ev_rag")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    processor = DataProcessor()
    try:
        # 尝试加载已处理的数据
        index, chunks = load_ev_data()
        logger.info("Successfully loaded the processed EV data.")
    except FileNotFoundError:
        logger.info("No processed data is found, start loading and processing the raw data...")
        # 1. 加载和处理数据
        loader = CSVLoader("data/Electric_Vehicle_Population_Data.csv")
        ev_docs = loader.load(limit=1000)
        
        # 2. 可视化数据分布
        visualize_ev_distribution(ev_docs, results_dir / f"ev_distribution_{timestamp}.png")
        
        # 3. 处理数据
        index, chunks = processor.process_documents(ev_docs)
        save_ev_data(index, chunks)
        logger.info("The processed EV data has been saved.")
    
    # 4. 测试检索
    retriever = Retriever(processor.embedder, chunks)
    test_questions = [
        "What are the most popular electric vehicle manufacturers?",
        "Which EVs have the longest range?",
        "What is the average price of Tesla models?",
        "How many different types of electric vehicles are there?",
        "Which cities have the most electric vehicles?"
    ]
    
    retrieval_results = test_retrieval_quality(retriever, test_questions)
    retrieval_results.to_csv(results_dir / f"retrieval_results_{timestamp}.csv")
    
    # 5. 测试不同模型的回答
    generator = AnswerGenerator()
    models = ["gpt4o", "chatGLM", "claude3.5"]
    qa_results = []
    
    for model in models:
        for question in test_questions:
            try:
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n".join([d.page_content for d in relevant_docs])
                answer = generator.generate_answer(context, question, model=model)
                qa_results.append({
                    'model': model,
                    'question': question,
                    'answer': answer
                })
            except Exception as e:
                logger.error(f"Error with model {model}: {e}")
    
    # 保存问答结果
    pd.DataFrame(qa_results).to_csv(results_dir / f"qa_results_{timestamp}.csv")
    
    logger.info("Testing completed successfully")
    return True

if __name__ == "__main__":
    test_ev_rag_system()