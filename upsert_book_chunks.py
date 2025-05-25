import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables from .env (if running locally)
load_dotenv()

# Pinecone and OpenAI credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

index_name = "book-chunks-openai"   # new index name
dimension = 1536                    # for OpenAI embeddings (text-embedding-ada-002)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}' with dimension {dimension} ...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    # Wait for index to be fully ready
    while True:
        index_status = pc.describe_index(index_name)
        if index_status.status['ready']:
            break
        print("Waiting for index to be ready...")
        time.sleep(3)

# Connect to index
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load book chunks
with open("book_chunks.txt", "r") as file:
    chunks = file.read().split("\n\n---CHUNK---\n\n")

batch_size = 100
vectors = []

for i, chunk in enumerate(chunks):
    # Get OpenAI embedding for the chunk
    embedding_response = client.embeddings.create(
        input=chunk,
        model="text-embedding-ada-002"   # or "text-embedding-3-small" if preferred
    )
    emb = embedding_response.data[0].embedding
    vectors.append(
        {
            "id": str(i),
            "values": emb,
            "metadata": {"text": chunk}
        }
    )
    # Upsert in batches
    if len(vectors) == batch_size or i == len(chunks) - 1:
        index.upsert(vectors=vectors)
        print(f"Upserted {i + 1}/{len(chunks)} chunks...")
        vectors = []

print("All chunks uploaded to Pinecone index!")