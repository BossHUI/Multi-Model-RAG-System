# -*- coding: utf-8 -*-

import os
import logging
import matplotlib.pyplot as plt
from rag_system.visualization import (
    visualize_retrieval_results,
    visualize_answer_quality,
    visualize_results
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_visualization(questions, answers, relevant_docs):
    """
    @description test the visualization functions
    @param questions - a list of questions
    @param answers - a list of answers
    @param relevant_docs - a list of relevant documents
    """
    try:
        os.makedirs("results", exist_ok=True)
        
        # 测试检索结果可视化
        for question, docs in zip(questions, relevant_docs):
            visualize_retrieval_results(docs, question)
            plt.savefig(f"results/retrieval_results_{question[:30]}.png")
            plt.close()
        
        # 测试答案质量可视化
        scores = [1.0] * len(answers)
        visualize_answer_quality(questions, answers, scores)
        plt.savefig("results/answer_quality.png")
        plt.close()
        
        # 测试结果可视化
        results = [{"Question": q, "Answer": a} for q, a in zip(questions, answers)]
        visualize_results(results)
        plt.savefig("results/overall_results.png")
        plt.close()
        
        return True
    except Exception as e:
        logger.error(f"failed to test visualization：{str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # 示例数据
    test_questions = ["Question 1", "Question 2"]
    test_answers = ["Answer 1", "Answer 2"]
    test_docs = [["Doc 1"], ["Doc 2"]]
    
    success = test_visualization(test_questions, test_answers, test_docs)
    print(f"visualization test {'successful' if success else 'failed'}")