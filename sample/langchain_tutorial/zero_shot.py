from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview"
)

output_parser = StrOutputParser()

cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),
    ]
)

cot_chain = cot_prompt | model | output_parser

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
        ("human", "{text}"),
    ]
)

summarize_chain = summarize_prompt | model | output_parser

cot_summarize_chain = cot_chain | summarize_chain
result = cot_summarize_chain.invoke({"question": "10+2*3"})
print(result)
