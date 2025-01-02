from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
ai_message = AIMessage(content="カレーを作る")
output = output_parser.invoke(ai_message)
print(type(output))
print(output)