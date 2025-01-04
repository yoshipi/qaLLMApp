from operator import itemgetter
from typing import Any
from langchain_community.document_loaders import GitLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableParallel


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]], k: int = 60
) -> list[str]:
    content_score_mapping = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content

            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            content_score_mapping[content] += 1 / (rank + k)

    ranked = sorted(content_score_mapping.items(), key=itemgetter(1), reverse=True)
    return [content for content, _ in ranked]


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
chroma_retriever = db.as_retriever().with_config({"run_name": "chroma_retriever"})

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"}
)

prompt = ChatPromptTemplate.from_template(
    """
                                          以下の文脈だけを踏まえて質問に回答してください。
                                          文脈:
                                          {context}
                                          質問:{question}
                                          """
)

hybrid_retriever = (
    RunnableParallel(
        {"chroma_documents": chroma_retriever, "bm25_documents": bm25_retriever}
    )
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

query = "LangChainの概要を教えてください。"
output = chain.invoke(query)
print(output)
