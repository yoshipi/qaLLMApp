from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableLambda

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant。"),
        ("human", "{input}"),
    ]
)

model = AzureChatOpenAI(
    azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview"
)

output_parser = StrOutputParser()


def upper(text: str) -> str:
    return text.upper()


chain = prompt | model | output_parser | RunnableLambda(upper)
output = chain.invoke({"input": "hello"})
print(output)


@RunnableLambda
def upper_with_decorate(text: str) -> str:
    return text.upper()


decorate_chain = prompt | model | output_parser | upper_with_decorate
decorate_output = decorate_chain.invoke({"input": "who are you?"})
print(decorate_output)
