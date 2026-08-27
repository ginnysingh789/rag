import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

persistent_directory="db/chroma_db"
#Load the same embedding model
embedding_model=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )
# Open the Vector database, So you can perform searching in it
vectorDatabase=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)
#User query
query="Who wrote the book Adventures of Sherlock Holmes"
#Create a Retriever
retriever = vectorDatabase.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
#Retrieve the relevant chunks
relevant_docs=retriever.invoke(query)
#Call the LLM with the relevant chunks to generate the response
llm=ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
# Given the instruction what to do with the relevant chunks
context = "\n\n".join(doc.page_content for doc in relevant_docs)
combinedInput = f"""
Answer the question using only the information provided in the context.

Question:
{query}

Context:
{context}

Instructions:
- Give only the direct answer.
- Keep the answer to 1-2 sentences.
- Do not mention the documents, context, chunks, metadata, or retrieval process.
- Do not explain how you found the answer.
- If the answer cannot be found in the context, say "I don't know."
"""
messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combinedInput),
]
result=llm.invoke(messages)
print("Response "+result.content)
