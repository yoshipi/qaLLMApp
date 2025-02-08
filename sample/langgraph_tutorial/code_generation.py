import operator
import re
import subprocess
import tempfile
import uuid
from typing import Sequence
from typing import Annotated, Any
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langsmith import traceable
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


# Javaコード生成結果を保持するためのクラス
class GeneratedJavaCode(BaseModel):
    java_class_name: str = Field(..., description="Generated Java class name")
    java_code: str = Field(..., description="Generated Java code")


# LangGraphのステータス保持クラス
class State(BaseModel):
    design_document: str = Field(..., description="Design document")
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default=[], description="message history"
    )
    java_class_name: str = Field(default="", description="Java class name")
    java_code: str = Field(default="", description="Java code")
    current_judge: bool = Field(default=False, description="Current judgement")
    judgement_reason: str = Field(default="", description="Judgement reason")
    generation_count: int = Field(default=0, description="Generation count")


def compile(java_class_name: str, java_code: str) -> tuple[bool, str]:
    # Javaファイルを作成
    with open(f"./{java_class_name}.java", encoding="utf-8", mode="w") as java_file:
        java_file.write(java_code)
        java_file_path = java_file.name

    # コンパイル
    compile_process = subprocess.run(
        ["javac", java_file_path], capture_output=True, text=True
    )
    if compile_process.returncode != 0:
        return False, f"コンパイルエラー: {compile_process.stderr}"
    else:
        return True, "コンパイルエラーなし"


llm = AzureChatOpenAI(azure_deployment="gpt-4o", api_version="2024-08-01-preview")


@traceable
def generate(state: State) -> dict[str, Any]:
    generation_output_parser = PydanticOutputParser(pydantic_object=GeneratedJavaCode)
    java_code_generation_prompt = ChatPromptTemplate(
        [
            (
                "system",
                """
            # 指示
            あなたはJavaプログラミングのプロフェッショナルです。
            設計書から実行可能なJavaコードを出力してください。

            {format_instructions}
            """,
            ),
            ("human", "{design_document}"),
        ],
        partial_variables={
            "format_instructions": generation_output_parser.get_format_instructions()
        },
    )

    generate_chain = java_code_generation_prompt | llm | generation_output_parser
    generated_java_code: GeneratedJavaCode = generate_chain.invoke(
        {"design_document": state.design_document}
    )

    # Javaコード部分のみを抽出
    generated_java_code.java_code = (
        generated_java_code.java_code.strip("```java").strip("```").strip()
    )
    return {
        "messages": generated_java_code.java_code,
        "java_class_name": generated_java_code.java_class_name,
        "java_code": generated_java_code.java_code,
        "generation_count": state.generation_count + 1,
    }


@traceable
def modify(state: State) -> dict[str, Any]:
    java_code_modification_prompt = ChatPromptTemplate(
        [
            (
                "system",
                """
            # 指示
            あなたはJavaプログラミングのプロフェッショナルです。
            条件に従って、コンパイルエラーを解決した実行可能なJavaコードを出力してください。

            ## 条件
            * 設計書に従ってください。
            * Javaコードのみ出力してください。

            ### 設計書
            {design_document}
            
            """,
            ),
            MessagesPlaceholder("chat_history", Optional=True),
            ("human", "{compile_result}"),
        ],
    )
    modification_chain = java_code_modification_prompt | llm | StrOutputParser()
    modification_result: str = modification_chain.invoke(
        {
            "design_document": state.design_document,
            "compile_result": state.judgement_reason,
            "chat_history": state.messages,
        }
    )

    # Javaコード部分のみを抽出
    modification_result = modification_result.strip("```java").strip("```").strip()

    return {
        "messages": modification_result,
        "generation_count": state.generation_count + 1,
        "java_code": modification_result,
    }


def check(state: State) -> dict[str, Any]:
    compile_result: tuple[bool, str] = compile(
        java_class_name=state.java_class_name,
        java_code=state.java_code,
    )
    if (not compile_result[0]) or state.generation_count > 3:
        return {"current_judge": True, "judgement_reason": "OK"}
    return {"current_judge": False, "judgement_reason": compile_result[1]}


graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate)
graph_builder.add_node("modify", modify)
graph_builder.add_node("check", check)

graph_builder.set_entry_point("generate")
graph_builder.add_edge("generate", "check")
graph_builder.add_conditional_edges(
    "check", lambda state: state.current_judge, {True: END, False: "modify"}
)
memory = MemorySaver()
app = graph_builder.compile(checkpointer=memory)

# 適当な設計書を生成
design_document_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            あなたはJavaプログラミングのプロフェッショナルです。
            ユーザーから指示される要件を実現するための設計書を出力してください。
            設計書のみ出力してください。
            """,
        ),
        ("human", "{input}"),
    ]
)

chain = design_document_prompt | llm | StrOutputParser()
output = chain.invoke({"input": "数当てゲームを作成してください。"})

initial_state = State(design_document=output)
thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
result = app.invoke(initial_state, config)

print(result["java_code"])
print(result["current_judge"])
print(result["judgement_reason"])
