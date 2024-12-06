# src/ml_data_pipeline/main.py
from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from ml_data_pipeline.endpoints.health import router as health_router
from ml_data_pipeline.endpoints.pipeline import router as pipeline_router

app = FastAPI(title="ML Data Pipeline API", version="1.0")

# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(pipeline_router, prefix="/api", tags=["Pipeline"])

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

logger.add("logs/ml_service.log", rotation="10 MB")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
