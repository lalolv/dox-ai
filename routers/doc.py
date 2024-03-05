import os
from fastapi import APIRouter, UploadFile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


router = APIRouter(prefix="/doc")

# Upload doc
# thenlper/gte-large-zh
# all-MiniLM-L6-v2
@router.post("/upload")
async def upload_doc(file: UploadFile):
    # Upload file
    if file.filename is not None:
        # Save file
        save_file = os.path.join("data/", file.filename)
        f = open(save_file, 'wb')
        data = await file.read()
        f.write(data)
        f.close()
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        raw_doc = TextLoader(save_file).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_doc)

        Chroma.from_documents(
            documents=documents,
            persist_directory='vecdb',
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name='doxai'
        )

    return file.filename


# 测试
@router.get("/lookup")
async def lookup_from_vecdb():
    db = Chroma(
        collection_name='doxai',
        persist_directory='vecdb',
        embedding_function=OllamaEmbeddings(model="nomic-embed-text")
    )
    query = "如果我是使用 Vue 的新手，可以给我哪些建议？"
    docs = db.similarity_search(query)
    return docs[0].page_content
