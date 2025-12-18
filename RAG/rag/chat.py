# user Query

from dotenv import load_dotenv

load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embeddings,
)
query = input("> ")

search_results = vector_db.similarity_search(query=query)

print(search_results)


if __name__ == "__main__":
    pass
