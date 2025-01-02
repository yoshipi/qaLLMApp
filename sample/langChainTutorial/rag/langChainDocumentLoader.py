from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document

def file_filter(file_path: str) -> bool:
    return file_path.endswith('.mdx')

def load_documents() -> list[Document]:
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter
        )
    
    raw_docs = loader.load()
    return raw_docs