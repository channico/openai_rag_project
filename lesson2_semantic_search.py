from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

client = OpenAI()

documents = [
    "Customers may return products within 30 days of purchase.",
    "Standard shipping takes three to five business days.",
    "Password resets can be requested from the account settings page.",
    "Our customer support team is available Monday through Friday.",
]

def create_embedding(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [item.embedding for item in response.data]

def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))

document_embeddings = create_embedding(documents)

question = input("Ask a question: ")
question_embedding = create_embedding([question])[0]

results = []

for document, document_embeddings in zip(documents, document_embeddings):
    similarity = cosine_similarity(question_embedding, document_embeddings)
    results.append((similarity, document))

results.sort(reverse=True)

print("\nMost relevant documents:\n")

for similarity, document in results:
    print(f"{similarity:.4f}: {document}")

