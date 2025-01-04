from langchain_community.document_loaders import GitLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loader.load()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large", openai_api_version="2023-05-15"
)

model = AzureChatOpenAI(
    azure_deployment="gpt-4o", openai_api_version="2024-08-01-preview"
)

db = Chroma.from_documents(documents, embeddings)
retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
                                          以下の文脈だけを踏まえて質問に回答してください。
                                          文脈:
                                          {context}
                                          質問:{question}
                                          """
)


hypothetical_prompt = ChatPromptTemplate.from_template(
    """
                                                        次の質問に回答する一文を生成してください。
                                                        
                                                        質問: {question}
                                                        """
)
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()


chain = (
    {"context": hypothetical_chain | retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

query = "LangChainの概要を教えてください。"
output = chain.invoke(query)
print(output)
