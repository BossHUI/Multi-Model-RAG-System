# -*- coding: utf-8 -*-

import logging
import os
from rag_system.utils import load_rag_system, save_rag_system
from tests.test_arxiv_rag_system import test_rag_components
from tests.test_multi_model import test_multi_model_generation
from tests.test_model_tasks import test_models_on_tasks
from tests.test_visualization import test_visualization
from rag_system.data_loader import ArxivLoader
from rag_system.preprocessor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    @description 主函数，根据是否存在已处理数据决定是否重新处理并保存，之后运行各项测试
    """
    try:
        # 尝试从 processed_data 加载处理后的数据
        index, chunks = load_rag_system()
        print("successfully loaded processed data from processed_data")
    except Exception as e:
        logger.info("failed to load processed data, starting to process raw paper data...")
        # 如果加载失败则从原始数据处理
        loader = ArxivLoader("data/Arxiv/arxiv-metadata-oai-snapshot.json")
        papers = loader.load(limit=1000)
        processor = DataProcessor()
        index, chunks = processor.process_arxiv_papers(papers)
        # 保存处理好的数据
        save_rag_system(index, chunks)
        logger.info("successfully saved processed data to processed_data")
    
    try:
        test_rag_components()
        test_multi_model_generation()
        test_models_on_tasks()
        
    except Exception as err:
        logger.error(f"encountered an error while running tests: {str(err)}", exc_info=True)

if __name__ == "__main__":
    main()
    