from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.comparisons import router as comparisons_router
from app.api.experiments import router as experiments_router
from app.api.seed_analysis import router as seed_analysis_router
from app.storage.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Intelligent ML Experiment Assistant",
    version="0.1.0",
    description="Run, track, evaluate, diagnose, and explain ML experiments.",
    lifespan=lifespan,
)

app.include_router(experiments_router)
app.include_router(comparisons_router)
app.include_router(seed_analysis_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}