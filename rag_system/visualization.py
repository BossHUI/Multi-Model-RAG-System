# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import time, os
import numpy as np
from wordcloud import WordCloud


def visualize_retrieval_results(relevant_docs, question):
    """
    @description 可视化检索结果
    @param relevant_docs - 检索到的相关文档列表
    @param question - 用户输入的问题
    """
    data = {
        "Question": [question] * len(relevant_docs),
        "Document Title": [doc.metadata['title'] for doc in relevant_docs],
        "Authors": [", ".join(doc.metadata['authors']) for doc in relevant_docs],
        "Abstract": [doc.page_content[:200] + "..." for doc in relevant_docs]
    }
    # 添加一个模拟的相关性分数（基于文档顺序）
    data["Relevance Score"] = [1.0 - (i * 0.2) for i in range(len(relevant_docs))]
    df = pd.DataFrame(data)
    # 打印检索结果表格
    print("\n检索结果:")
    print(df[["Document Title", "Abstract", "Relevance Score"]])

    # 绘制相关性评分柱状图
    plt.figure(figsize=(12, 6))
    y_pos = np.arange(len(df["Document Title"]))
    plt.barh(y_pos, df["Relevance Score"], color='skyblue')
    plt.yticks(y_pos, [title[:50] + "..." for title in df["Document Title"]], fontsize=8)
    plt.xlabel('Relevance Score')
    plt.title(f'Document Relevance for Query: "{question[:50]}..."')
    plt.tight_layout()


def visualize_answer_quality(questions, answers, scores):
    """
    @description 可视化生成答案的质量
    @param questions - 问题列表
    @param answers - 生成的答案列表
    @param scores - 答案质量评分列表
    """
    plt.figure(figsize=(12, 6))
    # 创建水平条形图
    y_pos = np.arange(len(questions))
    plt.barh(y_pos, scores, color='lightgreen')
    plt.yticks(y_pos, [q[:50] + "..." for q in questions], fontsize=8)
    plt.xlabel('Answer Quality Score')
    plt.title('Quality of Generated Answers')
    plt.tight_layout()


def measure_performance(retriever, generator, questions):
    """
    @description 测量检索和生成答案的时间性能
    @param retriever - 检索器
    @param generator - 生成器
    @param questions - 问题列表
    """
    retrieval_times = []
    generation_times = []

    for question in questions:
        # 测量检索时间
        start_time = time.time()
        relevant_docs = retriever.get_relevant_documents(question)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)

        # 测量生成时间
        context = "\n".join([d.page_content for d in relevant_docs])
        start_time = time.time()
        generator.generate_answer(context, question)
        generation_time = time.time() - start_time
        generation_times.append(generation_time)

    # 绘制时间性能图
    plt.figure(figsize=(10, 6))
    plt.plot(questions, retrieval_times, label='Retrieval Time', marker='o')
    plt.plot(questions, generation_times, label='Generation Time', marker='o')
    plt.xlabel('Questions')
    plt.ylabel('Time (seconds)')
    plt.title('Retrieval and Generation Time Performance')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def visualize_context_answer_relation(context, answer):
    """
    @description 可视化上下文与生成答案的关系
    @param context - 检索到的上下文
    @param answer - 生成的答案
    """
    # 生成上下文的词云
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(context)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Context Word Cloud')
    plt.show()

    # 生成答案的词云
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(answer)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Answer Word Cloud')
    plt.show()


def visualize_results(results):
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    
    # 保存结果为CSV
    df.to_csv("results/model_answers.csv", index=False)
    
    # 可视化生成时间
    plt.figure(figsize=(10, 6))
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.barh(model_data['Question'], model_data['Answer'].apply(len), label=model)
    
    plt.xlabel('Answer Length')
    plt.title('Answer Length by Model')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/answer_length_by_model.png")
    plt.close()