from typing import Any
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import GitLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.documents import Document


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    coherence_rerank = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return coherence_rerank.rerank(documents=documents, query=question)


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

chain = (
    {"question": RunnablePassthrough(), "documents": retriever}
    | RunnablePassthrough.assign(context=rerank)
    | prompt
    | model
    | StrOutputParser()
)

query = "LangChainの概要を教えてください。"
output = chain.invoke(query)
print(output)
