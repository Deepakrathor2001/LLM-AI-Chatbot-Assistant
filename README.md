# Vision AI - LLM Chatbot with RAG and Memory

An intelligent AI chatbot built using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and conversation memory. The chatbot can answer user queries using both its language model capabilities and information retrieved from custom documents, making responses more accurate, contextual, and relevant.

## Project Overview

The goal of this project was to understand how modern AI assistants work beyond simple prompt-response interactions. Instead of relying only on a language model's pre-trained knowledge, this chatbot retrieves relevant information from external documents and combines it with conversation history before generating a response.

The system is designed to maintain context during conversations, retrieve relevant knowledge from documents, and provide more reliable answers through a Retrieval-Augmented Generation pipeline.

## Key Features

* Conversational AI powered by LLM APIs
* Retrieval-Augmented Generation (RAG)
* Context-aware responses
* Conversation memory management
* Document-based question answering
* Custom prompt engineering
* Semantic search using vector embeddings
* Modular and scalable architecture

## How It Works

The chatbot follows a multi-step process:

1. The user submits a query.
2. The query is processed by the prompt engine.
3. Relevant conversation history is retrieved from memory.
4. The RAG engine searches for relevant document chunks.
5. Retrieved context is combined with chat history.
6. A structured prompt is generated.
7. The final prompt is sent to the LLM API.
8. The generated response is returned to the user.

This approach helps reduce hallucinations and improves response quality when answering questions related to custom knowledge sources.

## System Architecture

User Query

↓

Prompt Engine

↓

Memory Retrieval

↓

RAG Engine

↓

Vector Search

↓

Relevant Context

↓

LLM API

↓

Generated Response

↓

User

## Technology Stack

### Programming Language

* Python

### LLM Integration

* Gemini API 2.5 key 

### Frameworks

* LangChain

### Vector Database

* FAISS

### Embeddings

* HuggingFace Embeddings
* Sentence Transformers

### Frontend

* Streamlit

### Data Processing

* NumPy
* Pandas

## Project Structure

```text
LLM-Chatbot/
│
├── app.py
├── rag_engine.py
├── memory_system.py
├── prompt_engine.py
├── requirements.txt
├── README.md
│
├── data/
├── documents/
├── vector_store/
└── assets/
```

## Core Components

### Prompt Engine

The prompt engine is responsible for constructing structured prompts before sending requests to the language model. It combines user queries, retrieved context, and conversation history to improve response quality.

### Memory System

The memory module stores previous interactions and allows the chatbot to maintain conversational context. This helps create a more natural user experience across multiple turns.

### RAG Engine

The Retrieval-Augmented Generation engine retrieves relevant information from external documents using embeddings and vector similarity search. Retrieved content is provided to the LLM as additional context.

### LLM API Layer

This layer handles communication with the language model provider and returns generated responses to the application.

## Challenges Faced

One of the main challenges was managing context efficiently while keeping prompts within token limits. Another challenge was improving retrieval quality so that the most relevant document chunks were selected for each query.

Balancing retrieval accuracy, response quality, and API cost required multiple iterations and experimentation with chunking strategies, embeddings, and prompt design.

## What I Learned

Through this project, I gained practical experience in:

* Large Language Models
* Retrieval-Augmented Generation (RAG)
* Prompt Engineering
* Embedding Models
* Vector Databases
* Semantic Search
* API Integration
* Context Management
* End-to-End AI Application Development

## Future Improvements

* Multi-document support
* Hybrid search (keyword + semantic)
* User authentication
* Persistent memory storage
* Conversation analytics
* Source citation support
* Multi-language capabilities
* Agent-based workflows

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/llm-chatbot.git
```

Navigate to the project directory:

```bash
cd llm-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your API key and run:

```bash
streamlit run app.py
```

## Conclusion

This project was built to explore the practical implementation of modern Generative AI systems. By combining LLMs, RAG, vector search, and conversational memory, the chatbot demonstrates how AI assistants can provide more relevant, contextual, and knowledge-aware responses than traditional chatbots.

## Author

Deepak Rathor

My name is Deepak. AI/ML Engineer | Machine Learning | NLP | Large Lanuage Model | LLM Applications | RAG
I am passionate about Artificial Intelligence and Machine Learning. I enjoy solving problems and building real-world projects.

# Important Information 
I cannot share my API key so please if you want to use this project than create .env file and upload your personal API key of any AI tool like i am using Google Gemini 2.5 API key.
