# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data_enrichment_FAISS.views import router as enrichment_router

app = FastAPI(
    title="Automated Data Enrichment - One API Deduplication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(enrichment_router)

@app.get("/")
async def root():
    return {"message": "One-API Data Enrichment Ready!", "endpoint": "POST /enrichment/process"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)