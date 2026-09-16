from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.database import init_db
from auth.router import router as auth_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(auth_router)


@app.get("/")
def home():
    return {"message": "Backend running!"}