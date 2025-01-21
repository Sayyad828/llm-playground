import os
from langchain_openai import ChatOpenAI


api_key = ""
os.environ['OPENAI_API_KEY'] = api_key

prompt = "Explain in 50 words, what is egypt?"

llm = ChatOpenAI(model='gpt-4o-mini')
response = llm.invoke(prompt)
print(response.content)

print("======== Extra Info ========")
print(f"Complete Tokens {response.response_metadata["token_usage"]["completion_tokens"]}")
print(f"Prompt length: {len(prompt.split())} words == {response.response_metadata["token_usage"]["prompt_tokens"]} tokens")
print(f"Model used: {response.response_metadata["model_name"]}")