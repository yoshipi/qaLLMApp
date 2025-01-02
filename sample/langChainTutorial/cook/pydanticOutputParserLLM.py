from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from recipe import Recipe
from langchain_openai import AzureChatOpenAI
    
output_parser = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [("system", "ユーザーが入力した料理のレシピを考えてください。¥n¥n"
     "{format_instructions}"),
     ("human","{dish}")]
)

prompt_with_format_instructions = prompt.partial(format_instructions=format_instructions)
prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})

model = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview")
ai_message = model.invoke(prompt_value)
print(ai_message.content)