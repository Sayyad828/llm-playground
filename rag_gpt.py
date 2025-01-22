import os
import pprint
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


api_key = ""
os.environ['OPENAI_API_KEY'] = api_key
pp = pprint.PrettyPrinter(depth=4)

## Initiate LLM
llm = ChatOpenAI(model='gpt-4o-mini')

## Load the PDF
loader= PyPDFLoader(r"M:\projects\llm-playground\static\service agreement Between KDC and ESBAAR.pdf")
pages= loader.load()

## Create the creating the retriever using FAISS
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
retriever = faiss_index.as_retriever()

## Create the retriever chain to select the most relevent documents using both the FAISS and the LLM
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

## Create the document chain which takes the relevent documents, with the user query and the instructions then it generates the response
prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
    ("system", "After answering the question, you have to support your answer with a reference from the context to prove your answer. The reference should be refering to a section and a page in the given context with a quote from the context as well"),
    ("system", "If the user question is not related to the document at all just say 'This is an irrelevent question to my domain' and don't answer it"),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user","{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

## Testing
Question = "What type of data is ESBAAR requiring to collect and own?"
chat_history = [HumanMessage(content="This is an agreement between ESBAAR and KDC. ESBAAR is a service provider for KDC")]

### Method 1 (Manually invoke the two chains)
context = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": Question})
response = document_chain.invoke({
    "chat_history": chat_history,
    "context": context,
    "input": Question})
print("============= Method 1 =============")
print("="*5 + "Question" + "="*5 + "\n" + Question)
print("="*5 + "Answer" + "="*5 + "\n" + response)


### Method 2 (Use a full retrieval chain)
# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain) # Create retrieval chain that retrieves documents and then passes them on.
# response = retrieval_chain.invoke({
# "chat_history":chat_history,
# "input":Question})
# print("============= Method 2 =============")
# print("="*5 + "Question" + "="*5 + "\n" + Question)
# print("="*5 + "Answer" + "="*5 + "\n" + response["answer"])
