#-*- coding: utf-8 -*-

import logging
import pandas as pd
from rag_system.generator import AnswerGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_models_on_tasks():
    """
    @description test models on tasks
    """
    try:
        # 定义测试任务
        tasks = {
            "open_domain": [
                ("What is the capital of France?", "Context about France..."),
                ("Who wrote 'Pride and Prejudice'?", "Context about literature..."),
                ("What is the tallest mountain in the world?", "Context about geography...")
            ],
            "domain_specific": [
                ("What are the recent developments in transformer architecture?", 
                 "Context about transformer architecture..."),
                ("Explain the main contributions in quantum computing papers.", 
                 "Context about quantum computing..."),
                ("What are the trending topics in machine learning research?", 
                 "Context about machine learning...")
            ]
        }

        models = ["gpt4o", "chatGLM", "deepseek", "local_gpt2"]
        generator = AnswerGenerator()
        results = []

        for task_type, questions in tasks.items():
            for model in models:
                for question, context in questions:
                    try:
                        answer = generator.generate_answer(context, question, model=model)
                        results.append({
                            "Task Type": task_type,
                            "Model": model,
                            "Question": question,
                            "Answer": answer
                        })
                    except Exception as e:
                        logger.error(f"model {model} encounters error: {str(e)}", exc_info=True)
                        results.append({
                            "Task Type": task_type,
                            "Model": model,
                            "Question": question,
                            "Answer": f"Error: {e}"
                        })

        df = pd.DataFrame(results)
        df.to_csv("results/model_performance.csv", index=False)
        return df
    except Exception as e:
        logger.error(f"task testing encounters error: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    results = test_models_on_tasks()
    if results is not None:
        print(results)