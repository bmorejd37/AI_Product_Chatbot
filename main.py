import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

#Load the CSV data file
loader = CSVLoader(
    file_path="Brambleberry_FAQ.csv",
    encoding="utf-8"
)
docs = loader.load()

# Create a "Vector Store" or searchable database
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

#Set up the model
llm = OllamaLLM(model="mistral")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

#Chatbot user interface
print("\nBrambleberry Product Assistant")
print("\nHi, how can I help you today?")
print("To exit, type 'exit.'\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    
    response = qa_chain.invoke(user_input)
    print(f"\nAI: {response['result']}\n")