import os
from langchain_openai import OpenAIEmbeddings

api_key = ""
os.environ['OPENAI_API_KEY'] = api_key

text = "I love cookies"

embeddings = OpenAIEmbeddings()
embeddings_vector = embeddings.embed_query(text)

print(embeddings_vector)
