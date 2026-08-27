import os
# This library is used to load the data from the text files
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader,DirectoryLoader
# This library is used to create the chunks 
from langchain_text_splitters import CharacterTextSplitter
#Embedding Model to generate embeddings from the chunks
from langchain_huggingface import HuggingFaceEmbeddings
#For storing the embeddings in the vectorDB 
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

def load_document(docs_path="docs"):
    #check if text_data is exist or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist")
    # Load only the docs files from the folder
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents=loader.load()
    if len(documents)==0:
        raise FileNotFoundError(f"No .txt file in the {docs_path}")
    return documents

# Split the texts into the chunks
def split_documents(documents,chunk_size=800,chunk_overlap=50):
    print("Generate splitter function is being called")
    text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks =text_splitter.split_documents(documents)
    return chunks
    
def generate_embedding(chunks,persist_directory="db/chroma_db"):
    print("Generate Embedding function is being called")
    embedding_model=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )
    # Create a vector store locally
    print(f"Embedding are store in the DB")
    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        #Which Algo will being used during the searching part
        collection_metadata={"hnsw:space":"cosine"}
    )
    return vectorstore
    
def main():
    # Load the data from the text files
    print("main working")
    documents=load_document(docs_path="docs")
    
    #Generate the chunks
    chunks=split_documents(documents)
    #generate the embeddings
    embeddings=generate_embedding(chunks)
    return embeddings
if __name__ == "__main__":
    main()