
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv
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
print(f"User query {query} ")
for i , doc in enumerate(relevant_docs,1):
    print(f"Document {i} :\n {doc.page_content}")