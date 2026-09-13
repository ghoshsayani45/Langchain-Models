from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query ="Who is known for his calm demeanor and finishing skills?"

# Generate embeddings for the documents
document_embeddings = embeddings.embed_documents(documents)
print(f"Document embeddings: {document_embeddings}")

# Generate embedding for the query
query_embedding = embeddings.embed_query(query)

print(f"Query embedding: {query_embedding}")

# Calculate cosine similarity between the query and each document
similarities = cosine_similarity([query_embedding], document_embeddings)[0]

print(f"Similarities: {similarities}")


# Sorting the similarities to find the most similar document 
# (to keep the order of documents intact, enumerate is used)
index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(f"Query:{query}")

print(f"Most similar document: {documents[index]}")
print(f"Similarity score is: {score}")

