from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of dish")
    steps: list[str] = Field(description="steps to make dish")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","ユーザーが入力した料理のレシピを考えてください。"),
        ("human","{dish}"),
    ]
)

model =AzureChatOpenAI(azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview")

chain = prompt | model.with_structured_output(recipe)
recipe = chain.invoke({"dish":"カレー"})
print("with_structured_output")
print(type(recipe))
print(recipe)

output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [("system", "ユーザーが入力した料理のレシピを考えてください。¥n¥n"
     "{format_instructions}"),
     ("human","{dish}")]
)

prompt_with_format_instructions = prompt.partial(format_instructions=format_instructions)
prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})

print("PydanticOutputParser")
ai_message = model.invoke(prompt_value)
print(ai_message.content)


output_parser = StrOutputParser()
ai_message = AIMessage(content="カレーを作る")
output = output_parser.invoke(ai_message)
print("StrOutputParser")
print(type(output))
print(output)