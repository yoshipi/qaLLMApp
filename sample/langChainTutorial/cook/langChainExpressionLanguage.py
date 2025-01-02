from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from recipe import recipe
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","ユーザーが入力した料理のレシピを考えてください。"),
        ("human","{dish}"),
    ]
)

model =AzureChatOpenAI(azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview")

chain = prompt | model.with_structured_output(recipe)
recipe = chain.invoke({"dish":"カレー"})
print(type(recipe))
print(recipe)