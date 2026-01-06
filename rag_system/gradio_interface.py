# # # -*- coding: utf-8 -*-

import gradio as gr
from rag_system.utils import load_rag_system, save_rag_system, save_ev_data, load_ev_data
from rag_system.generator import AnswerGenerator,MultiModelGenerator
from rag_system.retriever import Retriever
from rag_system.data_loader import ArxivLoader, CSVLoader
from rag_system.preprocessor import DataProcessor, SentenceTransformerEmbeddings

class DomainRAGSystem:
    def __init__(self):
        self.domains = {
            "arxiv": self._init_arxiv_system(),
            "ev": self._init_ev_system()
        }
        self.current_domain = "arxiv"
        self.generator = AnswerGenerator()
        # self.generator = MultiModelGenerator()
        self.embedding_type = "sentence-transformers"  # 默认embedding类型
    

    def set_embedding_type(self, embedding_type: str):
        """设置embedding类型"""
        self.embedding_type = embedding_type
        # 重新初始化处理器
        processor = DataProcessor()
        processor.embedder = SentenceTransformerEmbeddings(embedding_type=self.embedding_type)
        for domain in self.domains:
            self.domains[domain].embedder = processor.embedder

    def _init_arxiv_system(self):
        try:
            index, docs = load_rag_system()
            processor = DataProcessor()
            return Retriever(processor.embedder, docs)
        except:
            loader = ArxivLoader("data/Arxiv/arxiv-metadata-oai-snapshot.json")
            papers = loader.load(limit=1000)
            if papers and isinstance(papers[0], dict):
                processor = DataProcessor()
                index, docs = processor.process_documents(papers)
                save_rag_system(index, docs)
                return Retriever(processor.embedder, docs)
            else:
                raise ValueError("Invalid papers data format")

    def _init_ev_system(self):
        try:
            index, docs = load_ev_data()
            processor = DataProcessor()
            return Retriever(processor.embedder, docs)
        except:
            loader = CSVLoader("data/Electric_Vehicle_Population_Data.csv")
            cases = loader.load(limit=3000)
            processor = DataProcessor()
            index, docs = processor.process_documents(cases)
            save_ev_data(index, docs)
            return Retriever(processor.embedder, docs)

    def rag_system(self, question: str, domain: str="arxiv", model_name: str="gpt4o", embedding_type: str="sentence-transformers"):
        self.current_domain = domain
        self.set_embedding_type(embedding_type)
        retriever = self.domains[domain]
        
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in relevant_docs])
        answer = self.generator.generate_answer(context, question, model=model_name)
        
        return context, answer

rag = DomainRAGSystem()

iface = gr.Interface(
    fn=rag.rag_system,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter your question here..."),
        gr.Dropdown(choices=["arxiv", "ev"], label="Select Domain"),
        gr.Dropdown(choices=["gpt4o", "chatGLM", "deepseek", "local_gpt2", "claude3.5", "qwen"], label="Select Model"),
        gr.Dropdown(choices=["sentence-transformers", "huggingface"], label="Select Embedding Type"),
    ],
    outputs=[
        gr.Textbox(lines=5, label="Retrieved Documents"),
        gr.Textbox(lines=5, label="Generated Answer")
    ],
    title="Multi-Domain RAG System",
    description="Select a domain and model to answer your questions."
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)