from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import auth, outlets
from models import User, NewsOutlet, NewsDigest # Import to register models
import os

app = FastAPI(title="Urbanous API")

# CORS
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://urbanous.vercel.app",
    "https://urbanous.net",
    "https://www.urbanous.net",
    os.getenv("FRONTEND_URL", "https://urbanous.vercel.app") 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
async def startup():
    # In dev/standalone, create tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(outlets.router, tags=["News Agents"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Urbanous API"}
