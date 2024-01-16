import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


if __name__ == "__main__":
    print("Hello")
    loader = PyPDFLoader(
        "/Users/raymondlow/Documents/langchain-medium-analyser/medium_blogs/mediumblog1.pdf"
    )
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_medium")

    new_vectorstore = FAISS.load_local("faiss_index_medium", embeddings=embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever()
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner"
    result = qa.run(query)
    print(result)
