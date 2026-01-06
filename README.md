# Multi-Model RAG System: Large-Scale Document Retrieval and Question Answering

## Abstract

This project implements a comprehensive **Retrieval-Augmented Generation (RAG)** system designed to handle large-scale document retrieval and question answering tasks. The system integrates multiple large language models (LLMs) including GPT-4, ChatGLM, Deepseek, Claude3.5, Qwen, and local GPT-2 to generate accurate and contextually relevant answers based on retrieved documents. By combining FAISS for efficient vector search with various LLMs for answer generation, the system provides a robust solution for domain-specific knowledge retrieval and generation tasks across diverse datasets such as ArXiv research papers and electric vehicle population data.

## Features

- **Multi-Model Support**: Integrates 6+ LLMs (GPT-4, ChatGLM, Deepseek, Claude3.5, Qwen, Local GPT-2) with weighted voting mechanism
- **Flexible Retrieval**: Supports multiple embedding methods (Sentence-BERT, OpenAI Embeddings, HuggingFace Transformers)
- **Efficient Vector Search**: FAISS-based approximate nearest-neighbor search for fast document retrieval
- **Interactive Interface**: Gradio-based web UI for easy interaction with the RAG system
- **Comprehensive Evaluation**: Built-in evaluation metrics and visualization tools for performance analysis
- **Multi-Dataset Support**: Handles both structured (CSV) and unstructured (JSON) data formats

## Project Structure

```
my_rag_en/
├── data/                          # Raw datasets
│   ├── Arxiv/                     # ArXiv metadata
│   └── Electric_Vehicle_Population_Data.csv
├── processed_data/                # Processed and indexed data
│   ├── arxiv/                     # ArXiv processed data
│   ├── ev/                        # Partial EV data
│   └── full_ev/                   # Complete EV data
├── models/                        # Local model files
│   └── local_gpt2/                # GPT-2 model files
├── rag_system/                    # Core RAG system
│   ├── model_generator/           # LLM generators
│   │   ├── base.py                # Base generator interface
│   │   ├── deepseek.py
│   │   ├── gpt4o.py
│   │   ├── chatglm.py
│   │   ├── claude3.py
│   │   ├── qwen.py
│   │   ├── local_gpt2.py
│   │   ├── factory.py             # Generator factory
│   │   └── scheduler.py           # Multi-model scheduler
│   ├── main.py                    # Main entry point
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessor.py            # Data preprocessing
│   ├── generator.py               # Answer generation
│   ├── evaluation.py              # Evaluation metrics
│   ├── utils.py                   # Utility functions
│   ├── retriever.py               # Document retrieval
│   ├── gradio_interface.py        # Web UI
│   └── visualization.py           # Visualization tools
├── tests/                         # Test suite
│   ├── test_arxiv_rag_system.py
│   ├── test_ev_rag_system.py
│   ├── test_multi_model.py
│   ├── test_model_tasks.py
│   ├── test_embedding.py
│   └── test_visualization.py
├── results/                       # Output results and visualizations
├── .gradio/                       # Gradio session data
│   └── flagged/                   # User interaction logs
├── download_model.py              # Model download script
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables
└── README-en.md                   # This file
```

## Datasets

### ArXiv Dataset

**Location**: `data/Arxiv/arxiv-metadata-oai-snapshot.json`

**Source**: [ArXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

**Description**: A repository of 1.7 million articles with features including article titles, authors, categories, abstracts, and full text PDFs.

#### Metadata Fields

- `id`: ArXiv ID (used to access papers)
- `submitter`: Paper submitter
- `authors`: Authors of the paper
- `title`: Title of the paper
- `comments`: Additional info (pages, figures, etc.)
- `journal-ref`: Journal publication information
- `doi`: Digital Object Identifier
- `abstract`: Paper abstract
- `categories`: ArXiv categories/tags
- `versions`: Version history

#### Access Links

- Abstract page: `https://arxiv.org/abs/{id}`
- PDF download: `https://arxiv.org/pdf/{id}`

#### Bulk Access

The full set of PDFs is available in the GCS bucket `gs://arxiv-dataset` or through Google API. Use `gsutil` to download data to your local machine.

### Electric Vehicle Dataset

**Location**: `data/Electric_Vehicle_Population_Data.csv`

**Source**: [Electric Vehicle Population on Kaggle](https://www.kaggle.com/datasets/willianoliveiragibin/electric-vehicle-population)

**Description**: Dataset containing electric vehicle population data. BEV sales during Q2 2023 grew over 50% YoY, with one in every 10 cars sold being a pure battery electric vehicle.

#### Data Processing Methods

1. **Structured Content**
   - Organize data into clear categorical structure (vehicle info, location info, additional details)
   - Use format strings to create readable text content

2. **Complete Metadata**
   - Save important fields as metadata for retrieval and filtering
   - Includes vehicle information (VIN, manufacturer, model, etc.)
   - Includes location information (city, state, postcode)
   - Includes performance and price information (range, base price)

3. **Processing Advantages**
   - Better understanding and matching of relevant content
   - Support for specific vehicle queries, geographical distribution, and price range queries
   - Support for subsequent data analysis and visualization

#### Performance Considerations

⚠️ **Note**: The dataset contains approximately 150,000 records and may take about 3 hours to process. Consider the following optimization strategies:

1. **Reduce Data Volume**
   - Limit the number of documents loaded at once
   - Use the `limit` parameter in CSVLoader during testing

2. **Increase System Memory**
   - Add physical memory if supported
   - Close unnecessary programs to free up memory

3. **Use Memory Mapping**
   - Process large datasets without loading entire dataset into memory

4. **Batch Processing**
   - Divide data into smaller batches for processing and indexing

5. **Optimize Embedding Dimensions**
   - Use models with smaller embedding dimensions
   - Choose smaller pre-trained models

## Key Processes

### 1. Data Preparation
- Load and preprocess raw data from ArXiv and electric vehicle datasets
- Split documents into manageable chunks
- Generate embeddings using pre-trained models
- Build FAISS index for efficient nearest-neighbor search

### 2. Retrieval
- Compute query embeddings
- Perform approximate nearest-neighbor search using FAISS
- Retrieve top-K relevant documents based on query similarity

### 3. Generation
- Concatenate retrieved context with the query
- Use LLMs to generate coherent and contextually relevant answers
- Support multi-model ensemble with weighted voting

## Installation & Usage

### Prerequisites

- Python 3.8+
- Conda (recommended) or virtual environment

### Installation

1. **Install dependencies**

```shell
conda activate EnvName
pip install -r requirements.txt
```

2. **Set up environment variables**

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ZHIPUAI_API_KEY=your_zhipuai_key
# Add other API keys as needed
```

### Usage

1. **Test with small dataset** (recommended first step)

```shell
# Test RAG system components
python rag_system/main.py

# Test EV dataset RAG system
python tests/test_ev_rag_system.py

# Test ArXiv dataset RAG system
python tests/test_arxiv_rag_system.py

# Test multi-model generation
python tests/test_multi_model.py

# Test embedding methods
python tests/test_embedding.py
```

2. **Process complete dataset**

```shell
python rag_system/main.py --full-dataset
```

3. **Launch Gradio interface**

```shell
python rag_system/gradio_interface.py
```

## System Architecture

### Retriever Module

- **Multiple Vectorization Methods**: Supports Sentence-BERT, OpenAI Embeddings, and HuggingFace Transformers
- **Flexible Embedding**: Choose embedding method based on task requirements
- **Extensible Design**: Uniform interface for easy extension of new embedding methods

### Generator Module

- **Model-Specific Prompts**: Custom prompt templates designed for each model
- **Instruction Format Support**: Handles special instruction formats of different models
- **Domain Information Injection**: Supports dynamic injection of domain-specific information

### Multi-Model Collaboration

- **Weighted Voting**: Implements weighted voting mechanism for ensemble answers
- **Dynamic Configuration**: Supports dynamic weight configuration for different models
- **Fault Tolerance**: Individual model failures do not affect overall results

## Comparative Analysis

### Model Performance Comparison

| Model | Strengths | Best Use Cases |
|-------|-----------|----------------|
| **GPT-4** | High accuracy and fluency in open-domain Q&A | Customer support, education, general Q&A |
| **ChatGLM** | Strong performance in Chinese domain-specific Q&A | Chinese language Q&A tasks |
| **Deepseek** | Fast response time | Real-time applications requiring quick responses |
| **Local GPT-2** | Resource-efficient, offline capable | Offline or resource-constrained environments |
| **QWEN** | Good for long document processing | Long document answer generation |
| **Claude3.5** | Excellent long document handling | Long document answer generation |

### Application Scenarios

- **GPT-4**: High-accuracy scenarios (customer support, education)
- **ChatGLM**: Chinese language Q&A tasks
- **Deepseek**: Real-time applications with fast response requirements
- **Local GPT-2**: Offline or resource-limited environments
- **QWEN**: Long document answer generation
- **Claude3.5**: Long document answer generation with high quality

## Results & Visualization

The system provides comprehensive visualization and analysis tools:

1. **Retrieval Results**
   - Table display: Document titles, authors, abstract fragments, and relevance scores
   - Bar chart: Relevance scores for each document

2. **Generation Results**
   - Table display: Questions, generated answers, and quality scores
   - Bar chart: Answer quality scores for each question

3. **Performance Metrics**
   - Time performance chart: Retrieval and generation time per question
   - Tabular display: Question, retrieval time, and generation time

4. **Content Analysis**
   - Word cloud: Key words and context in generated answers

## Performance Analysis

The system tracks the following performance metrics:

1. **Training Speed**: Time to preprocess data and build FAISS index
2. **Retrieval Time**: Efficiency of FAISS search in returning relevant documents
3. **Generation Time**: Time taken by LLMs to generate answers
4. **Memory Usage**: Resource consumption during data processing and model inference

## Advanced Optimization

### Performance Optimization

- **GPU Acceleration**: Install `faiss-gpu` for faster vector search
  ```shell
  pip install faiss-gpu
  ```
- **Larger Models**: Try larger Sentence Transformer models (e.g., `all-mpnet-base-v2`)

### Function Enhancement

- Session history management
- Hybrid search (keyword + vector)
- Result caching mechanism

### Domain-Specific Adaptation

- Use domain-specific models (e.g., BioBERT for biomedicine)
- Build domain-specific glossaries
- Add entity recognition modules
