import os
import json
import time
import numpy as np
from typing import List
from rag_system.preprocessor import EmbeddingFactory


def test_embedders(texts: List[str], output_dir="results"):
    """测试所有embedder的性能"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for embed_type in ["sentence-transformers", "huggingface"]: # 有openai api key可以加上
        try:
            print(f"Testing {embed_type}...")
            embedder = EmbeddingFactory.get_embedder(embed_type)
            # print(type(embedder("test")))  # 查看返回类型
            start_time = time.time()
            
            # 添加类型检查和转换
            if embed_type == "sentence-transformers":
                embeddings = []
                for text in texts:
                    embedding = embedder(text)
                    if isinstance(embedding, str):
                        # 如果是字符串，尝试解析
                        try:
                            embedding = np.array(json.loads(embedding))
                        except:
                            embedding = np.zeros(768)  # 使用默认维度
                    embeddings.append(np.array(embedding).flatten())
            else:
                embeddings = [np.array(embedder(text)).flatten() for text in texts]
                
            elapsed_time = time.time() - start_time
            
            # 计算向量相似度
            similarity_matrix = np.zeros((len(texts), len(texts)))
            for i in range(len(texts)):
                for j in range(len(texts)):
                    similarity_matrix[i,j] = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            
            results[embed_type] = {
                "time": elapsed_time,
                "similarity_matrix": similarity_matrix.tolist(),
                "embeddings_shape": [e.shape if hasattr(e, 'shape') else len(e) for e in embeddings]
            }
            
            with open(f"{output_dir}/{embed_type}_results.json", "w") as f:
                json.dump(results[embed_type], f, indent=2)
                
        except Exception as e:
            print(f"Error testing {embed_type}: {str(e)}")
    
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "This is a test sentence.",
        "Another example of text for embedding.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Natural language processing is a fascinating field."
    ]
    
    # 运行测试
    results = test_embedders(test_texts)
    print("Test completed. Results saved to results/")
    print("Summary:")
    for embed_type, metrics in results.items():
        print(f"{embed_type}: {metrics['time']:.2f}s")