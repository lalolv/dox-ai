import os
from fastapi import APIRouter, WebSocket
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from fastapi import WebSocket, WebSocketDisconnect
from dotenv import load_dotenv


# load env
load_dotenv()
# Router
router = APIRouter(prefix="/lookup")


# Use a local model through Ollama
llm = Ollama(model=os.environ.get('OPENAI_MODEL_NAME'))
# Model name
emb_model = os.environ.get('EMBEDDINGS_MODEL')
device = os.environ.get('DEVICE')
# Vec DB
vectorstore = Chroma(
    collection_name="doxai",
    persist_directory='vecdb',
    embedding_function=HuggingFaceEmbeddings(
        model_name=emb_model, model_kwargs={'device': device})
)
# Retrieve and generate
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template('''
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
''')


# Connection Manager
class ConnectionManager:
    def __init__(self):
        # 存放激活的ws连接对象
        self.active_connections: List[WebSocket] = []
        # Chain
        self.chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    async def connect(self, ws: WebSocket):
        # 等待连接
        await ws.accept()
        # 存储ws连接对象
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        # 关闭时 移除ws对象
        self.active_connections.remove(ws)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    async def send_message(self, message: str, ws: WebSocket):
        # stream output
        for chunks in self.chain.stream(message):
            # print(chunks)
            await ws.send_text(chunks)
        # END
        await ws.send_text("[end]")

    async def broadcast(self, message: str):
        # 广播消息
        for connection in self.active_connections:
            await connection.send_text(message)


# Manager Object
manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    # Connect
    await manager.connect(websocket)
    # Broadcast message
    await manager.broadcast(f"Welcome!")

    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_message(data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Bye!")

