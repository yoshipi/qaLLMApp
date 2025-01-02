from langchain_text_splitters import CharacterTextSplitter
from langChainDocumentLoader import load_documents

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(load_documents())
print(len(docs))