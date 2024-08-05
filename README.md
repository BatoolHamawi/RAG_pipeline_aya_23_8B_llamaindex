# RAG_pipeline_aya_23_8B_llamaindex
# Chat with Wikipedia Docs, Powered by LlamaIndex

Welcome to the "Chat with Wikipedia Docs" app, powered by LlamaIndex and hosted on Streamlit! This application allows users to interact with Wikipedia documents in Arabic, using advanced language models from Hugging Face.

## Features

- **Interactive Chat**: Ask questions about subjects in Wikipedia docs in Arabic.
- **Powered by LlamaIndex**: Utilizes the capabilities of LlamaIndex for efficient document indexing and querying.
- **Hugging Face Integration**: Leverages powerful models from Hugging Face for natural language understanding.
- **RAG Pipeline**: Implements the Retrieval-Augmented Generation (RAG) pipeline for accurate and contextual responses.

## RAG Pipeline Overview

### What is RAG?

Retrieval-Augmented Generation (RAG) is a framework that combines the strengths of retrieval-based models and generation-based models. It enhances the capabilities of language models by integrating a retrieval mechanism that fetches relevant documents from a large corpus, such as Wikipedia, which are then used to generate more accurate and contextually appropriate responses.

### How RAG is Implemented in This Project

1. **Document Indexing**: Wikipedia documents in Arabic are indexed using LlamaIndex. This process involves breaking down the documents into smaller chunks and storing them in a way that allows for efficient retrieval.

2. **Querying**: When a user asks a question, the query is processed and used to retrieve relevant documents from the indexed corpus. This is achieved using the `HuggingFaceEmbedding` model for semantic search.

3. **Response Generation**: The retrieved documents are then fed into the `aya-23-8B` model from Hugging Face, which generates a coherent and contextually relevant response based on the provided information.

4. **Streamlit Interface**: The entire process is wrapped in a user-friendly Streamlit interface, allowing users to interact with the model seamlessly.

## Setup Instructions

### Prerequisites

- Python 3.9 
- [Anaconda](https://www.anaconda.com/products/individual) (optional but recommended for managing environments)

### Clone the Repository

```sh
git clone https://github.com/yourusername/chat-with-wikipedia-docs.git
cd chat-with-wikipedia-docs
```
### Install Dependencies
create a virtual environment to manage dependencies:
```sh
conda create -n mychatbot python=3.8
conda activate mychatbot
```
### Install the required Python packages:
```sh
pip install -r requirements.txt
```
### Add Hugging Face and OpenAI API Keys
Create a file named secrets.toml in the .streamlit directory with the following content:
```sh
# .streamlit/secrets.toml
openai_key = "your_openai_api_key"
huggingface_token = "your_hugging_face_api_token"
```
