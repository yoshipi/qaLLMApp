import getpass
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        MessagesPlaceholder("chat_history", optional=True),
        ("user", "{text}"),
    ]
)

model = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

print("This is prompt:" + str(prompt))
response = model.invoke(prompt)
print(response.content)
