from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
file_path = Path(__file__).parent.parent / "sets.pdf"


loader = PyPDFLoader(file_path)
# Read the pdf
docs = loader.load()
# print(docs[0])
# Chunking

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(documents=docs)
# print(texts)

# Vector Embeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

vector_store = QdrantVectorStore.from_documents(
    documents=texts,
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embeddings,
)
print("Indexing of Docuemts done...")


if __name__ == "__main__":
    pass
