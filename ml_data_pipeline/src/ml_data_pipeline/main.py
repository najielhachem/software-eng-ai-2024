# src/ml_data_pipeline/main.py
from fastapi import FastAPI

from ml_data_pipeline.endpoints.health import router as health_router

app = FastAPI(title="ML Data Pipeline API", version="1.0")

# Include API routes
app.include_router(health_router, prefix="/api", tags=["Health"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
