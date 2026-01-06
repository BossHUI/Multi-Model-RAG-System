
# -*- coding: utf-8 -*-

import logging
import os
import csv
from rag_system.generator import MultiModelGenerator, AnswerGenerator

# 配置日志和结果目录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs('results', exist_ok=True)

def test_single_model_generation():
    """测试单个模型生成器"""
    generator = AnswerGenerator(model_name="gpt4o")
    
    context = "Some context about machine learning..."
    question = "What is machine learning?"
    
    results = {}
    models_to_test = ["gpt4o", "chatGLM", "claude3.5", "qwen"]
    
    for model in models_to_test:
        try:
            answer = generator.generate_answer(context, question, model=model)
            results[model] = answer
        except Exception as e:
            logger.error(f"{model} testing failed：{str(e)}", exc_info=True)
            results[model] = f"Error: {e}"
    
    return results

def test_multi_model_generation():
    """测试多模型协作生成"""
    generator = MultiModelGenerator()
    
    test_cases = [
        {
            "context": "Some context about machine learning...",
            "question": "What is machine learning?",
            "domain": "AI"
        },
        # 可以添加更多测试用例...
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        context = test_case["context"]
        question = test_case["question"]
        domain = test_case["domain"]
        
        try:
            # 获取多模型结果
            final_answer = generator.generate_answer(context, question, domain)
            
            # 记录每个模型的独立结果
            model_results = {}
            for model_name, gen in generator.generators.items():
                try:
                    model_results[model_name] = gen.generate_answer(context, question, model_name, domain)
                except Exception as e:
                    model_results[model_name] = f"Error: {str(e)}"
            
            # 保存结果
            results.append({
                "test_case": i+1,
                "context": context,
                "question": question,
                "domain": domain,
                "final_answer": final_answer,
                **model_results
            })
            
        except Exception as e:
            logger.error(f"Error in test case {i+1}: {str(e)}")
    
    return results

def save_results(results, filename):
    """保存结果到CSV文件"""
    with open(f'results/{filename}', 'w', newline='', encoding='utf-8') as f:
        if isinstance(results, list):
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        else:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Answer'])
            for model, answer in results.items():
                writer.writerow([model, answer])

if __name__ == "__main__":
    # 运行单模型测试
    single_model_results = test_single_model_generation()
    save_results(single_model_results, 'single_model_results.csv')
    
    # 运行多模型测试
    multi_model_results = test_multi_model_generation()
    save_results(multi_model_results, 'multi_model_results.csv')