import operator
import re
import subprocess
import tempfile
from typing import Annotated, Any
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class GeneratedJavaCode(BaseModel):
    java_class_name: str = Field(..., description="Generated Java class name")
    java_code: str = Field(..., description="Generated Java code")


class State(BaseModel):
    design_document: str = Field(..., description="Design document")
    messages: Annotated[list[GeneratedJavaCode], operator] = Field(
        [], description="Answer history"
    )
    current_judge: bool = Field(default=False, description="Current judgement")
    judgement_reason: str = Field(default="", description="Judgement reason")


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview"
)


def generate(state: State) -> dict[str, Any]:
    java_code_generation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            あなたはJavaプログラミングのプロフェッショナルです。
            設計書から実行可能なJavaコードを出力してください。
            Javaコードのみs出力してください。
            """,
            ),
            ("human", "{design_document}"),
        ]
    )
    generate_chain = java_code_generation_prompt | llm.with_structured_output(
        GeneratedJavaCode
    )
    generated_java_code: GeneratedJavaCode = generate_chain.invoke(
        {"design_document": state.design_document}
    )

    # Javaコード部分のみを抽出
    generated_java_code.java_code = (
        generated_java_code.java_code.strip("```java").strip("```").strip()
    )
    return {"messages": [generated_java_code]}


def check(state: State) -> dict[str, Any]:
    return {"current_judge": True, "judgement_reason": "OK"}


graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate)
graph_builder.add_node("check", check)

graph_builder.set_entry_point("generate")
graph_builder.add_edge("generate", "check")
graph_builder.add_conditional_edges(
    "check", lambda state: state.current_judge, {True: END, False: "generate"}
)

complied = graph_builder.compile()

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
result = complied.invoke(initial_state)

print(result["messages"][-1])


def compile_and_run_java(java_class_name: str, java_code: str) -> str:
    # Javaファイルを作成
    with open(f"./{java_class_name}.java", encoding="utf-8", mode="w") as java_file:
        java_file.write(java_code)
        java_file_path = java_file.name

    # コンパイル
    compile_process = subprocess.run(
        ["javac", java_file_path], capture_output=True, text=True
    )
    if compile_process.returncode != 0:
        return f"コンパイルエラー: {compile_process.stderr}"


compile_result = compile_and_run_java(
    java_class_name=result["messages"][-1].java_class_name,
    java_code=result["messages"][-1].java_code,
)
print(compile_result)
