from operator import itemgetter
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
retriever = db.as_retriever()


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="The generated queries.")


query_generation_prompt = ChatPromptTemplate.from_template(
    """
    質問に対してベクターデータベースから関連文書を検索するために、3つの異なる検索クエリを生成してください。
    距離ベースの類似性検索の限界を克服するために、ユーザーの質問に対して複数の視点を提供することが目標です。
    
    質問: {question}
"""
)

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

prompt = ChatPromptTemplate.from_template(
    """
                                          以下の文脈だけを踏まえて質問に回答してください。
                                          文脈:
                                          {context}
                                          質問:{question}
                                          """
)

chain = (
    {
        "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

query = "LangChainの概要を教えてください。"
output = chain.invoke(query)
print(output)
