import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


api_key = ""
os.environ['OPENAI_API_KEY'] = api_key

db = [
    "Egypt is one of the biggest countries when it comes to the population in the middle east.",
    "Real Madrid has lost against Barcelona yesterday for a score 5 to 2 and Barcelona won the super cup. Lamen Yamal has scored a hattrick, Raffiniah has scored a goal and got a penalty that was scored by Lewandoveski.",
    "Python has finally released a new version which is 3.13 and it's much more powerful when it comes to the asynchronous processing on multiple cpu cores.",
    "Donald Trump has just announced his new crypto currency yesterday, the coin started with a very low price around 6 dollars and within 4 hours it reached the peak of 74 dollars. Afterwards, the coin crashed and the price is now around 34 dollars."
]
query = "What was the score of the last Real Madrid vs Barcelona match?"

embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_texts(texts=db, embedding=embeddings)
response = faiss_index.similarity_search(query=query, k=2)

for i, r in enumerate(response):
    print(f"Rank {i+1} document chunck:")
    print(r)