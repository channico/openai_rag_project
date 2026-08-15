from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def chunk_text(text, chunk_size=40, overlap=10):
    if overlap >= chunk_size:
        raise ValueError('Overlap cannot be greater than chunk_size')

    words = text.split(' ')
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def create_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [item.embedding for item in response.data]


def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a)*np.linalg.norm(vector_b))

document = Path("company_handbook.txt").read_text(encoding='utf-8')

chunks = chunk_text(document, chunk_size=40, overlap=10)

print(f"Created {len(chunks)} chunks.\n")

for index, chunk in enumerate(chunks):
    print(f"CHUNK {index}")
    print(chunk)
    print()

chunk_embeddings = create_embeddings(chunks)

question = input("Ask a question: ")
question_embedding = create_embeddings([question])[0]

results = []

for index, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
    similarity = cosine_similarity(question_embedding, embedding)

    results.append({
        "chunk_id": index,
        "text": chunk,
        "similarity": similarity,
    })

results.sort(
    key=lambda result: result["similarity"],
    reverse=True,
)

print("\nMost relevant chunks:\n")

for result in results[:3]:
    print(
        f"Chunk {result['chunk_id']} "
        f"- similarity {result['similarity']:.4f} "
    )
    print(result["text"])