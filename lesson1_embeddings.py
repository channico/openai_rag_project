from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "The cat is sleeping on the sofa"

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embedding = response.data[0].embedding

print("Text:")
print(text)

print("\nEmbedding Length:")
print(len(embedding))

print("\nFirst 10 numbers:")
print(embedding[:10])