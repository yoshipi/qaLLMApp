#!/usr/bin/env python3
"""
qaLLMApp Capabilities Showcase
あなたは何ができる？(What can you do?)

This application demonstrates the comprehensive capabilities of the qaLLMApp repository.
このアプリケーションは qaLLMApp リポジトリの包括的な機能を実演します。
"""

import sys
from typing import Dict, List


class CapabilitiesShowcase:
    """Main class for showcasing qaLLMApp capabilities"""
    
    def __init__(self):
        self.capabilities = self._initialize_capabilities()
    
    def _initialize_capabilities(self) -> Dict[str, Dict]:
        """Initialize the capabilities catalog"""
        return {
            "1": {
                "name_en": "Multi-Role Q&A System",
                "name_ja": "マルチロールQ&Aシステム",
                "description_en": "Intelligent role selection and quality-checked answers",
                "description_ja": "インテリジェントな役割選択と品質チェック付き回答",
                "file": "sample/qaTest.py",
                "features": [
                    "Automatic role selection (General Expert, AI Expert, Counselor)",
                    "Quality assurance with re-processing",
                    "Japanese language support"
                ]
            },
            "2": {
                "name_en": "RAG (Retrieval Augmented Generation) Systems",
                "name_ja": "RAG (検索拡張生成) システム",
                "description_en": "Advanced document retrieval and generation capabilities",
                "description_ja": "高度な文書検索と生成機能",
                "file": "sample/langchain_tutorial/rag*.py",
                "features": [
                    "Basic RAG with document loading",
                    "Hybrid retriever (vector + keyword search)",
                    "HyDE (Hypothetical Document Embeddings)",
                    "Query generation and reranking",
                    "Git repository document processing"
                ]
            },
            "3": {
                "name_en": "Intelligent Prompt Generation",
                "name_ja": "インテリジェントプロンプト生成",
                "description_en": "Interactive system for creating optimized prompts",
                "description_ja": "最適化されたプロンプトを作成するためのインタラクティブシステム",
                "file": "reverseCodeToDocument.py",
                "features": [
                    "Guided prompt requirement gathering",
                    "Tool-based structured information extraction",
                    "State-based conversation flow",
                    "Automated prompt optimization"
                ]
            },
            "4": {
                "name_en": "Requirements Gathering AI Agent",
                "name_ja": "要件収集AIエージェント",
                "description_en": "Systematic requirement analysis through AI interviews",
                "description_ja": "AI面接による体系的な要件分析",
                "file": "sample/requirementTest.py",
                "features": [
                    "Persona-based requirement gathering",
                    "Automated stakeholder interviews",
                    "Iterative information collection",
                    "Comprehensive requirement documentation"
                ]
            },
            "5": {
                "name_en": "Conversational AI Chatbots",
                "name_ja": "対話型AIチャットボット",
                "description_en": "Advanced chatbot implementations with LangGraph",
                "description_ja": "LangGraphを使用した高度なチャットボット実装",
                "file": "sample/langgraph_tutorial/basicChatBot.py",
                "features": [
                    "State-based conversation management",
                    "Memory and context preservation",
                    "Streaming response generation",
                    "Extensible conversation flows"
                ]
            },
            "6": {
                "name_en": "Code Generation Agents",
                "name_ja": "コード生成エージェント",
                "description_en": "AI agents specialized in code generation and analysis",
                "description_ja": "コード生成と分析に特化したAIエージェント",
                "file": "sample/langgraph_tutorial/code_generation.py",
                "features": [
                    "Automated code generation",
                    "Code analysis and documentation",
                    "Multi-language support",
                    "Intelligent code optimization"
                ]
            },
            "7": {
                "name_en": "Specialized AI Agents",
                "name_ja": "専門AIエージェント",
                "description_en": "Task-specific agents for various domains",
                "description_ja": "様々なドメインのタスク固有エージェント",
                "file": "sample/langgraph_tutorial/",
                "features": [
                    "Python execution agent",
                    "Basic autonomous agents",
                    "Domain-specific problem solving",
                    "Tool integration capabilities"
                ]
            }
        }
    
    def display_welcome_message(self):
        """Display welcome message in both languages"""
        print("=" * 80)
        print("🤖 qaLLMApp Capabilities Showcase")
        print("🤖 qaLLMApp 機能紹介")
        print("=" * 80)
        print()
        print("あなたは何ができる？(What can you do?)")
        print()
        print("This application demonstrates comprehensive AI capabilities including:")
        print("このアプリケーションは以下を含む包括的なAI機能を実演します：")
        print()
    
    def display_capabilities_menu(self):
        """Display the main capabilities menu"""
        print("📋 Available Capabilities / 利用可能な機能:")
        print("-" * 50)
        
        for key, capability in self.capabilities.items():
            print(f"{key}. {capability['name_en']}")
            print(f"   {capability['name_ja']}")
            print(f"   📁 {capability['file']}")
            print()
    
    def display_capability_details(self, capability_key: str):
        """Display detailed information about a specific capability"""
        if capability_key not in self.capabilities:
            print("❌ Invalid selection / 無効な選択")
            return
        
        cap = self.capabilities[capability_key]
        print(f"🔍 {cap['name_en']} / {cap['name_ja']}")
        print("=" * 60)
        print(f"📝 Description: {cap['description_en']}")
        print(f"📝 説明: {cap['description_ja']}")
        print(f"📁 Implementation: {cap['file']}")
        print()
        print("✨ Key Features / 主要機能:")
        for feature in cap['features']:
            print(f"  • {feature}")
        print()
    
    def display_technical_details(self):
        """Display technical implementation details"""
        print("🔧 Technical Implementation / 技術実装:")
        print("-" * 50)
        print("• Framework: LangChain + LangGraph")
        print("• LLM Provider: Azure OpenAI (GPT-4o)")
        print("• Vector Store: Chroma")
        print("• Additional Services: Cohere API")
        print("• Language: Python 3.12+")
        print("• State Management: Pydantic + TypedDict")
        print("• Architecture: Agent-based workflows")
        print()
    
    def display_setup_requirements(self):
        """Display setup and configuration requirements"""
        print("⚙️  Setup Requirements / セットアップ要件:")
        print("-" * 50)
        print("Environment Variables Required:")
        print("• AZURE_OPENAI_ENDPOINT")
        print("• AZURE_OPENAI_API_KEY") 
        print("• COHERE_API_KEY")
        print("• LANGSMITH_API_KEY (optional)")
        print()
        print("Python Dependencies:")
        print("• langchain, langchain-openai, langchain-community")
        print("• langgraph, langchain-chroma, langchain-cohere")
        print("• GitPython, rank-bm25")
        print()
    
    def display_usage_examples(self):
        """Display usage examples"""
        print("💡 Usage Examples / 使用例:")
        print("-" * 50)
        print("1. Multi-Role Q&A:")
        print("   python sample/qaTest.py")
        print("   Ask: '私の人生の目的は何ですか？'")
        print()
        print("2. RAG Document Query:")
        print("   python sample/langchain_tutorial/rag.py")
        print("   Query LangChain documentation")
        print()
        print("3. Interactive Prompt Generation:")
        print("   python reverseCodeToDocument.py")
        print("   Create optimized prompts interactively")
        print()
        print("4. Requirements Gathering:")
        print("   python sample/requirementTest.py")
        print("   AI-driven stakeholder interviews")
        print()
    
    def run_interactive_showcase(self):
        """Run the interactive capabilities showcase"""
        self.display_welcome_message()
        
        while True:
            print("\n" + "=" * 60)
            print("📌 Main Menu / メインメニュー:")
            print("1. View All Capabilities / すべての機能を表示")
            print("2. Technical Details / 技術詳細")
            print("3. Setup Guide / セットアップガイド")
            print("4. Usage Examples / 使用例")
            print("5. Detailed Capability Info / 詳細機能情報")
            print("q. Quit / 終了")
            
            try:
                choice = input("\nSelect option / オプションを選択 (1-5, q): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = "q"
                
            if choice == "1":
                print("\n")
                self.display_capabilities_menu()
            elif choice == "2":
                print("\n")
                self.display_technical_details()
            elif choice == "3":
                print("\n")
                self.display_setup_requirements()
            elif choice == "4":
                print("\n")
                self.display_usage_examples()
            elif choice == "5":
                print("\n")
                self.display_capabilities_menu()
                try:
                    cap_choice = input("Select capability number (1-7): ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                print("\n")
                self.display_capability_details(cap_choice)
            elif choice.lower() == "q":
                print("\n👋 ありがとうございました！Thank you for exploring qaLLMApp!")
                print("🚀 Ready to build amazing AI applications!")
                break
            else:
                print("❌ Invalid option / 無効なオプション")


def main():
    """Main entry point"""
    showcase = CapabilitiesShowcase()
    showcase.run_interactive_showcase()


if __name__ == "__main__":
    main()