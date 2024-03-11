from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from routers import system, doc, lookup
from loguru import logger

from routers import lookup


# load env
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Startup!')
    yield
    logger.info('Closed!')

# create app
app = FastAPI(lifespan=lifespan)

# 允许的访问域
origins = ["*"]


# 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ping for test
@app.get("/")
async def read_root():
    return {"Hello": "World"}


# Routers
app.include_router(system.router)
app.include_router(doc.router)
app.include_router(lookup.router)
