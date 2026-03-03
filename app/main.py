from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Tennis AI MVP")

app.include_router(router)