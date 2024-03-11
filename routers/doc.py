import os
from fastapi import APIRouter, UploadFile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from transformers import AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel


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
        # Model name
        emb_model = os.environ.get('EMBEDDINGS_MODEL')
        device = os.environ.get('DEVICE')
        # text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        #     AutoTokenizer.from_pretrained(emb_model),
        #     chunk_size=1000, chunk_overlap=200
        # )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_doc)

        Chroma.from_documents(
            documents=documents,
            persist_directory='vecdb',
            embedding=HuggingFaceEmbeddings(
                model_name=emb_model, model_kwargs={'device': device}),
            collection_name='doxai'
        )

    return file.filename


class QueryItem(BaseModel):
    query: str

# 测试
@router.post("/lookup")
async def lookup_from_vecdb(item: QueryItem):
    # Model name
    emb_model = os.environ.get('EMBEDDINGS_MODEL')
    device = os.environ.get('DEVICE')
    # Chrome vecdb
    db = Chroma(
        collection_name='doxai',
        persist_directory='vecdb',
        embedding_function=HuggingFaceEmbeddings(
            model_name=emb_model, model_kwargs={'device': device})
    )
    docs = db.similarity_search(item.query)
    return docs[0].page_content
