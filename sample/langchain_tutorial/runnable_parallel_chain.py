from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from operator import itemgetter

model = AzureChatOpenAI(
    azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview"
)
output_parser = StrOutputParser()

optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。",
        ),
        ("human", "{question}"),
    ]
)

optimistic_chain = optimistic_prompt | model | output_parser

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。",
        ),
        ("human", "{question}"),
    ]
)

pessimistic_chain = pessimistic_prompt | model | output_parser

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは客観的AIです。{question}について、二つの意見をまとめてください。",
        ),
        ("human", "楽観的意見:{optimistic}¥n悲観的意見:{pessimistic}"),
    ]
)

synthesize_chain = (
    RunnableParallel(
        {
            "optimistic": optimistic_chain,
            "pessimistic": pessimistic_chain,
            "question": itemgetter("question"),
        }
    )
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"question": "明日は晴れるかな？"})
print(output)
