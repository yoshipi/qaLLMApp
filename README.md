# qaLLMApp
Study LangChain - Comprehensive AI Application Showcase

## あなたは何ができる？ (What can you do?)

This repository demonstrates advanced AI applications built with LangChain and LangGraph, featuring:

🤖 **Multi-Role Q&A System** - Intelligent role selection with quality assurance  
🔍 **RAG (Retrieval Augmented Generation)** - Advanced document retrieval and generation  
✨ **Intelligent Prompt Generation** - Interactive prompt optimization system  
📋 **Requirements Gathering AI** - Automated stakeholder interview agent  
💬 **Conversational AI Chatbots** - State-based conversation management  
⚡ **Code Generation Agents** - AI-powered code analysis and generation  
🎯 **Specialized AI Agents** - Domain-specific problem-solving agents  

## Quick Start

**Explore all capabilities interactively:**
```bash
python capabilities_showcase.py
```

## How to setUp

### Quick Demo (No Setup Required)
```bash
# Run the capabilities showcase to see what this app can do
python capabilities_showcase.py
# OR use the launcher
python run_showcase.py
```

### Full Setup for Development

#### pip install

```shell
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install GitPython
pip install langchain-text-splitters
pip install langgraph
pip install langchain-chroma
pip install langchain-cohere
pip install rank-bm25
```

### environment variables

```shell
vim ~/.zshrc

export AZURE_OPENAI_ENDPOINT = Your environment endpoint
export AZURE_OPENAI_API_KEY = Your environment apiKey
export COHERE_API_KEY = Your cohere apiKey
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=Your environment endpoint
export LANGSMITH_API_KEY=Your langsmith api key
export LANGSMITH_PROJECT=YOur project
```

## Individual Applications

After setup, you can run specific applications:

### 🤖 Multi-Role Q&A System
```bash
python sample/qaTest.py
# Ask questions in Japanese or English - the AI will select the appropriate expert role
```

### 🔍 RAG Document Query
```bash
python sample/langchain_tutorial/rag.py
# Query LangChain documentation using retrieval-augmented generation
```

### ✨ Interactive Prompt Generation
```bash
python reverseCodeToDocument.py
# Create optimized prompts through guided conversation
```

### 📋 Requirements Gathering
```bash
python sample/requirementTest.py
# AI-driven stakeholder interview and requirement analysis
```

### 💬 Basic Chatbot
```bash
python sample/langgraph_tutorial/basicChatBot.py
# Simple conversational AI with state management
```
