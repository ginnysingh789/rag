#Import all the langchain libraries
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
load_dotenv()

#store our conversation as message
chat_history=[]
#connect to your vector database
persistent_directory="./db/chroma_db"

embedding_model=HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
    )
vector_database=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)
#Call the LLM with the relevant chunks to generate the response
llm=ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
def ask_questions(user_question):
    #If there is a chat_history , ask the LLM to rewrite the new question using the past history , so vector search can be done 
    if chat_history:
        #Ask AI to make new question
        messages=[
            SystemMessage(content="Given the chat History, rewrite the new question to be standalone and searchable ")
        ]+chat_history+[
            HumanMessage(content=f"New Question : {user_question}")
        ]
        result=llm.invoke(messages)
        search_question=result.content.strip()
        
    else:
        
        search_question=user_question
    
    # Find the relevant chunks from vector db
    retriever = vector_database.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
    relevant_docs=retriever.invoke(search_question)
    
    
    #Call the llm model with the new  searchable question and with the relevant chunks 
        # Given the instruction what to do with the relevant chunks
    
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    combinedInput = f"""
    Answer the question using only the information provided in the context.

    Question:
    {search_question}

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
    answer=llm.invoke(messages)
    #Append the result in the chatHistory
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer.content))
    print("Response "+answer.content)

def start_chat():
    print("Ask me question.Type 'quit' to exit")
    while True:
        question=input("\n Your questions: ")
        if question.lower()=='quit':
            print("GoodBye")
            break
        ask_questions(question)

if __name__=="__main__":
    start_chat()
    
    
# Who is the author of The Adventure of Sherlock Holmes