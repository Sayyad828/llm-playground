import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


api_key = "[ADD-API-KEY]"
os.environ['OPENAI_API_KEY'] = api_key

llm = ChatOpenAI(model='gpt-4o-mini')
llm.invoke("what can u help me with?")
print(llm)

loader= PyPDFLoader("M:\projects\llm-playground\static\service agreement Between KDC and ESBAAR.pdf")
pages= loader.load()

print(f"document has {len(pages)} pages")

text_splitter= CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
docs = text_splitter.split_documents(pages)

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
# faiss_index.save_local("kdc_agreement")


embeddings = OpenAIEmbeddings()
retriever = faiss_index.as_retriever()


prompt_search_query = ChatPromptTemplate.from_messages([
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])


retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)


prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user","{input}"),
])


document_chain = create_stuff_documents_chain(llm, prompt_get_answer)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


## Testing

chat_history = [HumanMessage(content="Does ESBAAR own the dataset?"), AIMessage(content="Yes")]
response = retrieval_chain.invoke({
"chat_history":chat_history,
"input":"How?"
})
print (response['answer'])