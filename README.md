# README: Understanding the Loop of Retrieval-Augmented Generation (RAG)

## **Introduction**
Retrieval-Augmented Generation (RAG) is a powerful method that combines retrieval from external knowledge sources with generative language models. This enables the model to provide accurate, context-aware, and grounded answers by leveraging both pre-trained knowledge and external, dynamically retrieved information.

This document provides a detailed explanation of the RAG loop, including its components, workflow, and practical implementation, with a focus on the provided code example.

---
 
## **What is RAG?**
RAG is a hybrid approach where a language model (LLM) generates responses based on retrieved external documents rather than relying solely on its pre-trained data. This is achieved by splitting the workflow into two primary stages:

1. **Retrieval**: Relevant documents are retrieved from a knowledge base (e.g., vector database) based on a query.
2. **Generation**: The retrieved documents are used as additional context to generate a grounded response.

---

## **Components of the RAG Pipeline**

### 1. **Data Source**
The knowledge base is created from documents (e.g., PDFs, text files) that are processed into smaller chunks for efficient retrieval. In the example code:
- A PDF document is loaded using `PyPDFLoader`.
- Text chunks are created with a text splitter (`CharacterTextSplitter`).
- A vector index is constructed using FAISS and OpenAI embeddings.

### 2. **Retriever**
The retriever is responsible for querying the knowledge base and finding the most relevant documents based on the input prompt. 
- A retriever is built on top of the FAISS vector store.
- It uses embeddings to find documents that semantically match the query.

### 3. **LLM and Chains**
#### a. **Language Model (LLM)**
The language model (e.g., GPT-4) is the core generative engine. It produces answers based on input prompts and retrieved context.
- In the example, `ChatOpenAI` is used with the `gpt-4o-mini` model.

#### b. **Chains**
Chains are pipelines that orchestrate the interactions between the retriever and the LLM. The two key chains are:
- **Retriever Chain**: Processes user input and retrieves relevant documents from the vector store.
- **Document Chain**: Uses the retrieved documents to generate a final, context-aware response.

---

## **RAG Workflow: Step-by-Step**

### Step 1: Load and Preprocess Data
1. Load documents into memory (e.g., using `PyPDFLoader`).
2. Split documents into manageable chunks (e.g., 10 characters).
3. Create a vector index of these chunks with FAISS and embeddings (e.g., `OpenAIEmbeddings`).

### Step 2: Build the Retrieval Chain
1. **Search Query Generation**:
   - Use the `retriever_chain` to generate a search query based on user input and conversation history.
   - This involves prompting the LLM to convert user intent into a document search query.

2. **Retrieve Relevant Documents**:
   - The retriever searches the vector index for chunks matching the generated search query.

### Step 3: Build the Document Chain
1. Format the retrieved documents into a context for the LLM.
2. Use a prompt template (e.g., `ChatPromptTemplate`) to ask the LLM to synthesize an answer based on the retrieved context.

### Step 4: Generate the Final Response
1. The `retrieval_chain` integrates the outputs of the `retriever_chain` and `document_chain`.
2. The final response includes an answer grounded in the retrieved documents, optionally with references or quotes for validation.

---

## **Key Functions in the Code**
1. **`PyPDFLoader`**: Loads the PDF and converts it into a list of pages.
2. **`CharacterTextSplitter`**: Splits the pages into smaller chunks for indexing.
3. **`FAISS.from_documents`**: Builds a FAISS vector store with document embeddings.
4. **`create_history_aware_retriever`**: Creates a chain that generates search queries based on conversation history.
5. **`create_stuff_documents_chain`**: Combines retrieved documents into a single context for the LLM.
6. **`create_retrieval_chain`**: Combines the retriever and document chains into a full RAG pipeline.

---

## **Visualization of Outputs**
To better understand the intermediate steps, the code has been modified to:
1. Print the **search query** generated by the retriever chain.
2. Display the **retrieved documents**.
3. Output the **final synthesized answer**.

---

## **Practical Example**
### User Question:
> "What type of data is ESBAAR requiring to collect and own?"

### Workflow Outputs:
#### 1. Search Query:
> "What data is ESBAAR requiring to collect and own in the agreement?"

#### 2. Retrieved Documents:
- **Document 1**: "ESBAAR will collect and own data related to customer behavior analytics and operational metrics. (Page 5, Section 3)"

#### 3. Final Answer:
> "ESBAAR requires the collection and ownership of data related to customer behavior analytics and operational metrics. Reference: 'ESBAAR will collect and own data related to customer behavior analytics and operational metrics.' (Page 5, Section 3)"

---

## **Conclusion**
The RAG pipeline seamlessly combines retrieval and generation to answer complex, context-dependent queries. By breaking the process into clear stages, it:
- Ensures responses are accurate and grounded in the provided documents.
- Allows for modular experimentation (e.g., testing different retrievers, embeddings, or prompt templates).

This pipeline is particularly useful in scenarios requiring both the depth of pre-trained LLMs and the precision of domain-specific knowledge bases.