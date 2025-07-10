# qaLLMApp Technical Documentation

## あなたは何ができる？ (What can you do?)

This document provides detailed technical information about the capabilities implemented in qaLLMApp.

## Architecture Overview

qaLLMApp is built using modern AI application frameworks:
- **LangChain**: Framework for building applications with Large Language Models
- **LangGraph**: Extension for creating stateful, multi-step AI workflows
- **Azure OpenAI**: Primary LLM provider (GPT-4o)
- **Chroma**: Vector database for embeddings and retrieval
- **Cohere**: Additional AI services for enhanced capabilities

## Core Capabilities

### 1. Multi-Role Q&A System (`sample/qaTest.py`)
**Japanese**: マルチロールQ&Aシステム

**Technical Implementation**:
- Automatic role classification using LLM-based selection
- Three specialized roles: General Expert, AI Expert, Counselor
- Quality assurance loop with re-processing capability
- State-based workflow using LangGraph

**Key Components**:
- `State` class: Manages conversation state and quality checks
- `selection_node`: Analyzes queries and selects appropriate expert role
- `answering_node`: Generates role-specific responses
- `check_node`: Quality validation with feedback loop

### 2. RAG (Retrieval Augmented Generation) Systems
**Japanese**: RAG (検索拡張生成) システム

**Multiple RAG Implementations**:
- **Basic RAG** (`rag.py`): Document loading and retrieval
- **Hybrid Retriever** (`rag_hybrid_retriever.py`): Vector + keyword search
- **HyDE** (`rag_hyde.py`): Hypothetical Document Embeddings
- **Query Generation** (`rag_generation_query.py`): Enhanced query processing
- **Reranking** (`rag_rerank_query.py`): Result optimization

**Technical Features**:
- Git repository document processing
- Chunking strategies for optimal retrieval
- Embedding-based similarity search
- Multi-modal retrieval approaches

### 3. Intelligent Prompt Generation (`reverseCodeToDocument.py`)
**Japanese**: インテリジェントプロンプト生成

**Workflow**:
1. Information gathering through structured conversation
2. Tool-based data extraction using Pydantic models
3. Automated prompt template generation
4. Iterative refinement process

**Components**:
- `PromptInstructions` model: Structured information capture
- State management with conversation flow
- Tool calling for structured output
- Memory-based conversation persistence

### 4. Requirements Gathering AI Agent (`sample/requirementTest.py`)
**Japanese**: 要件収集AIエージェント

**Systematic Approach**:
- Persona-based stakeholder simulation
- Automated interview generation and conduct
- Iterative information gathering
- Comprehensive requirement documentation

**Advanced Features**:
- Multi-persona interview simulation
- Evaluation-driven conversation flow
- Structured requirement extraction
- Quality assessment and iteration

### 5. Conversational AI Chatbots
**Japanese**: 対話型AIチャットボット

**Implementations**:
- Basic chatbot with state management
- Streaming response generation
- Memory and context preservation
- Extensible conversation flows

### 6. Code Generation Agents
**Japanese**: コード生成エージェント

**Capabilities**:
- Automated code generation
- Code analysis and documentation
- Multi-language support
- Intelligent optimization suggestions

### 7. Specialized AI Agents
**Japanese**: 専門AIエージェント

**Agent Types**:
- Python execution agent
- Basic autonomous agents
- Domain-specific problem solvers
- Tool integration capabilities

## State Management Patterns

All applications use consistent state management patterns:
- **Pydantic Models**: Type-safe state definitions
- **LangGraph StateGraph**: Workflow orchestration
- **Message Passing**: Inter-node communication
- **Conditional Edges**: Dynamic workflow routing

## Error Handling and Quality Assurance

- Structured output validation
- Quality check loops
- Graceful fallback mechanisms
- User input validation

## Extensibility

The architecture supports easy extension:
- New node types for additional capabilities
- Custom state models for specific use cases
- Pluggable LLM providers
- Modular component design

## Performance Considerations

- Streaming responses for better UX
- Efficient vector storage and retrieval
- Optimized prompt engineering
- Memory management for long conversations

## Security and Best Practices

- Environment variable configuration
- Secure API key management
- Input validation and sanitization
- Structured data handling