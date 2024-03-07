import os
from fastapi import APIRouter, WebSocket
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


router = APIRouter(prefix="/lookup")


@router.websocket("/completion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
    os.environ["OPENAI_MODEL_NAME"] = "gemma:7b"
    os.environ["OPENAI_API_KEY"] = ""

    # Vec DB 
    vectorstore = Chroma(
        collection_name="doxai",
        persist_directory='vecdb',
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    # Retrieve and generate 
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    # Use a local model through Ollama
    llm = Ollama(model="gemma:7b")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)

    # input
    input = await websocket.receive_text()
    # stream output
    for chunks in chain.stream(input):
        # print(chunks)
        await websocket.send_text(chunks)

    # close stream
    await websocket.close(reason='finish')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
