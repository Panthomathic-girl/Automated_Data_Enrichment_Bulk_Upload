# app/data_enrichment_faiss/views.py
import asyncio
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from app.data_enrichment_FAISS.rag_faiss import LeadFAISSIndex  # Note: your folder name has capital FAISS

router = APIRouter(tags=["Automated Data Enrichment"])

BATCH_SIZE = 100


# ===================== CRUD ENDPOINTS =====================

@router.post("/records")
async def add_records_batch(records: List[Dict]):
    """Add new records (skip if string 'id' already exists)"""
    if not records:
        raise HTTPException(400, "Records list cannot be empty")

    valid_records = [rec for rec in records if isinstance(rec.get("id"), str) and rec["id"].strip()]
    if not valid_records:
        raise HTTPException(400, "Each record must contain a valid non-empty string 'id' field")

    # Run in thread to keep FastAPI responsive
    actually_added = await asyncio.to_thread(LeadFAISSIndex.add_records, records)

    return {
        "status": "success",
        "submitted": len(records),
        "actually_added": actually_added,
        "skipped": len(records) - actually_added
    }


@router.put("/records")
async def upsert_records(records: List[Dict]):
    """Upsert records (update if 'id' exists, otherwise skip - no insert on upsert)"""
    if not records:
        raise HTTPException(400, "Records list cannot be empty")

    valid_records = [rec for rec in records if isinstance(rec.get("id"), str) and rec["id"].strip()]
    if not valid_records:
        raise HTTPException(400, "Each record must contain a valid non-empty string 'id' field")

    result = await asyncio.to_thread(LeadFAISSIndex.upsert_records, records)

    return {
        "status": "success",
        "message": "Upsert operation completed",
        **result
    }


@router.get("/records")
async def fetch_all_records():
    """Fetch all indexed records"""
    records = LeadFAISSIndex.get_all_records()  # Synchronous, fast for metadata
    return {
        "count": len(records),
        "records": records
    }


@router.get("/records/{record_id}")
async def fetch_by_id(record_id: str):
    """Fetch a single record by string id"""
    if not record_id:
        raise HTTPException(400, "Record ID cannot be empty")

    record = LeadFAISSIndex.get_by_id(record_id)
    if not record:
        raise HTTPException(404, f"Record with id='{record_id}' not found")

    return {"record": record}


@router.delete("/records/{record_id}")
async def delete_by_id(record_id: str):
    """Delete a record by string id"""
    if not record_id:
        raise HTTPException(400, "Record ID cannot be empty")

    success = LeadFAISSIndex.delete_by_id(record_id)
    if not success:
        raise HTTPException(404, f"Record with id='{record_id}' not found")

    return {"status": "deleted", "id": record_id}


@router.delete("/records")
async def delete_all_records():
    """Delete all records and reset index"""
    LeadFAISSIndex.delete_all()
    return {"status": "all records deleted"}


# ===================== DEDUPLICATION =====================

@router.post("/duplicates")
async def process_deduplication(leads: List[Dict]):
    try:
        if not leads:
            raise HTTPException(400, "leads list is required")

        total_indexed = len(LeadFAISSIndex.get_all_records())
        if total_indexed == 0:
            raise HTTPException(400, "No master records indexed yet. Use POST /records first.")

        device_used = LeadFAISSIndex.ensure_model_loaded()
        print(f"Deduplication running on device: {device_used}")

        all_results = []
        for i in range(0, len(leads), BATCH_SIZE):
            batch = leads[i:i + BATCH_SIZE]
            batch_results = LeadFAISSIndex.search_batch(
                query_leads=batch,
                top_k=total_indexed + 50,
                threshold=0.75
            )
            all_results.extend(batch_results)

        return {
            "status": "completed",
            "indexed_master_records": total_indexed,
            "leads_processed": len(leads),
            "similarity_threshold": 0.75,
            "note": "Returns all matches >= 75% similarity",
            "results": all_results
        }
    except Exception as e:
        raise HTTPException(500, f"Error during deduplication: {str(e)}")
