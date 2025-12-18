# app/data_enrichment_faiss/rag_faiss.py
import logging
from pathlib import Path
from typing import List, Dict, Optional
import faiss
import numpy as np
import pickle
import threading
import torch
from sentence_transformers import SentenceTransformer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path("faiss_leads_index.faiss")
META_PATH = Path("faiss_leads_meta.pkl")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class LeadFAISSIndex:
    _model = None
    _index = None
    _meta: List[Dict] = None
    _str_id_to_idx: Dict[str, int] = None      # string 'id' → position in _meta
    _next_internal_id: int = 0                  # sequential int for FAISS internal use
    _model_lock = threading.Lock()
    _device = None

    @classmethod
    def ensure_model_loaded(cls):
        cls._load_model()
        return cls._device

    @classmethod
    def _load_model(cls):
        with cls._model_lock:
            if cls._model is None:
                if torch.backends.mps.is_available():
                    cls._device = "mps"
                else:
                    cls._device = "cpu"

                print(f"Loading SentenceTransformer model '{MODEL_NAME}' on {cls._device}...")
                logger.info(f"Loading model on {cls._device}")
                cls._model = SentenceTransformer(MODEL_NAME, device=cls._device)
        return cls._model

    @classmethod
    def _record_to_text(cls, record: Dict) -> str:
        if not record or not isinstance(record, dict):
            return ""

        parts = []
        skip_keys = {"id"}

        for key, value in record.items():
            if key in skip_keys:
                continue
            if value is None:
                continue
            if isinstance(value, (str, int, float)):
                parts.append(str(value).strip())
            elif isinstance(value, (list, dict)):
                parts.append(str(value)[:1000])

        return " | ".join(filter(None, parts)) if parts else ""

    @classmethod
    def _ensure_index(cls):
        if cls._index is not None:
            return

        if INDEX_PATH.exists() and META_PATH.exists():
            cls.load_index()
        else:
            dim = 384
            base_index = faiss.IndexFlatIP(dim)
            cls._index = faiss.IndexIDMap(base_index)
            cls._meta = []
            cls._str_id_to_idx = {}
            cls._next_internal_id = 0

    @classmethod
    def load_index(cls):
        if cls._index is not None:
            return

        cls._index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
            cls._meta = data["meta"]
            cls._str_id_to_idx = data["str_id_to_idx"]
            cls._next_internal_id = data.get("next_internal_id", len(cls._meta))

        logger.info(f"Loaded FAISS index with {cls._index.ntotal} records")

    @classmethod
    def _save_index(cls):
        faiss.write_index(cls._index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump({
                "meta": cls._meta,
                "str_id_to_idx": cls._str_id_to_idx,
                "next_internal_id": cls._next_internal_id
            }, f)
        logger.info(f"Saved FAISS index with {len(cls._meta)} records")

    @classmethod
    def _rebuild_index(cls):
        """Rebuild index after deletions to keep internal IDs valid"""
        if not cls._meta:
            cls._index = None
            cls._str_id_to_idx = {}
            cls._next_internal_id = 0
            for path in [INDEX_PATH, META_PATH]:
                if path.exists():
                    path.unlink()
            return

        model = cls._load_model()
        dim = 384
        base_index = faiss.IndexFlatIP(dim)
        new_index = faiss.IndexIDMap(base_index)

        vectors = []
        internal_ids = []
        valid_meta = []
        new_str_id_to_idx = {}

        for idx, record in enumerate(cls._meta):
            text = cls._record_to_text(record)
            if not text.strip():
                continue
            vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
            internal_id = len(valid_meta)  # new sequential ID
            vectors.append(vec)
            internal_ids.append(internal_id)
            valid_meta.append(record)
            record_id = record["id"]
            new_str_id_to_idx[record_id] = len(valid_meta) - 1

        cls._meta = valid_meta
        cls._str_id_to_idx = new_str_id_to_idx
        cls._next_internal_id = len(valid_meta)

        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(internal_ids, dtype=np.int64)
            new_index.add_with_ids(vectors_np, ids_np)

        cls._index = new_index
        cls._save_index()

    # ===================== CRUD =====================

    @classmethod
    def add_records(cls, records: List[Dict], batch_size: int = 32) -> int:
        cls._ensure_index()
        model = cls._load_model()

        new_vectors = []
        new_internal_ids = []
        added = 0

        valid_records = []
        for record in records:
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                continue
            if record_id in cls._str_id_to_idx:
                continue  # Skip existing
            text = cls._record_to_text(record)
            if not text.strip():
                continue
            valid_records.append((record_id, text, record))

        if not valid_records:
            logger.info("No new valid records to add")
            return 0

        for i in range(0, len(valid_records), batch_size):
            batch = valid_records[i:i + batch_size]
            texts = [t for _, t, _ in batch]
            record_ids = [rid for rid, _, _ in batch]
            full_records = [rec for _, _, rec in batch]

            vecs = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=len(texts),
                show_progress_bar=False
            ).astype(np.float32)

            for vec, rid, rec in zip(vecs, record_ids, full_records):
                internal_id = cls._next_internal_id
                cls._meta.append(rec.copy())
                cls._str_id_to_idx[rid] = len(cls._meta) - 1
                new_vectors.append(vec)
                new_internal_ids.append(internal_id)
                cls._next_internal_id += 1
                added += 1

        if new_vectors:
            vectors_np = np.array(new_vectors, dtype=np.float32)
            ids_np = np.array(new_internal_ids, dtype=np.int64)
            cls._index.add_with_ids(vectors_np, ids_np)
            cls._save_index()

        logger.info(f"Added {added} new records (string UUIDs)")
        return added

    @classmethod
    def upsert_records(cls, records: List[Dict]) -> Dict[str, int]:
        cls._ensure_index()
        model = cls._load_model()

        updated = skipped = 0
        to_update = []

        for record in records:
            record_id = record.get("id")
            if not isinstance(record_id, str) or record_id not in cls._str_id_to_idx:
                skipped += 1
                continue
            text = cls._record_to_text(record)
            if not text.strip():
                skipped += 1
                continue
            vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
            to_update.append((record_id, vec, record))

        if not to_update:
            return {"updated": 0, "skipped": skipped}

        # Remove old vectors (by internal ID)
        internal_ids_to_remove = []
        for record_id, _, _ in to_update:
            idx = cls._str_id_to_idx[record_id]
            internal_ids_to_remove.append(idx)  # use current position as proxy (will rebuild anyway)

        if internal_ids_to_remove:
            cls._index.remove_ids(np.array(internal_ids_to_remove, dtype=np.int64))

        # Add updated versions
        new_vectors = []
        new_internal_ids = []
        for record_id, vec, rec in to_update:
            idx = cls._str_id_to_idx[record_id]
            cls._meta[idx] = rec.copy()
            internal_id = cls._next_internal_id
            new_vectors.append(vec)
            new_internal_ids.append(internal_id)
            cls._next_internal_id += 1
            updated += 1

        if new_vectors:
            vectors_np = np.array(new_vectors, dtype=np.float32)
            ids_np = np.array(new_internal_ids, dtype=np.int64)
            cls._index.add_with_ids(vectors_np, ids_np)
            cls._save_index()

        logger.info(f"Upsert: {updated} updated, {skipped} skipped")
        return {"updated": updated, "skipped": skipped}

    @classmethod
    def get_all_records(cls) -> List[Dict]:
        cls._ensure_index()
        return [r.copy() for r in cls._meta]

    @classmethod
    def get_by_id(cls, record_id: str) -> Optional[Dict]:
        cls._ensure_index()
        if record_id not in cls._str_id_to_idx:
            return None
        idx = cls._str_id_to_idx[record_id]
        return cls._meta[idx].copy()

    @classmethod
    def delete_by_id(cls, record_id: str) -> bool:
        cls._ensure_index()
        if record_id not in cls._str_id_to_idx:
            return False

        del cls._meta[cls._str_id_to_idx[record_id]]
        cls._str_id_to_idx.pop(record_id)
        cls._rebuild_index()
        logger.info(f"Deleted record with id={record_id}")
        return True

    @classmethod
    def delete_all(cls):
        cls._index = None
        cls._meta = None
        cls._str_id_to_idx = None
        cls._next_internal_id = 0
        for path in [INDEX_PATH, META_PATH]:
            if path.exists():
                path.unlink()
        logger.info("All records and index deleted")

    # ===================== SEARCH =====================

    @classmethod
    def search_similar(cls, query_record: Dict, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        cls._ensure_index()
        if cls._index.ntotal == 0:
            return []

        model = cls._load_model()
        text = cls._record_to_text(query_record)
        if not text.strip():
            return []

        q_vec = model.encode(text, normalize_embeddings=True).astype(np.float32)[np.newaxis]
        scores, ids = cls._index.search(q_vec, top_k + 20)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < threshold:
                continue
            record = cls.get_by_id(str(idx))
            if record:
                rec = record.copy()
                rec["similarity_score"] = round(float(score) * 100, 2)
                results.append(rec)
            if len(results) >= top_k:
                break
        return results

    @classmethod
    def search_batch(cls, query_leads: List[Dict], top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        cls._ensure_index()
        if cls._index.ntotal == 0:
            return [{"input_lead": lead, "duplicates_found": 0, "duplicates": []} for lead in query_leads]

        model = cls._load_model()
        texts = [cls._record_to_text(lead) for lead in query_leads]
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]

        results_map = {}

        if valid_texts:
            embeddings = model.encode(
                valid_texts,
                normalize_embeddings=True,
                batch_size=len(valid_texts),
                show_progress_bar=False
            ).astype(np.float32)

            D, I = cls._index.search(embeddings, top_k + 20)

            for i, orig_idx in enumerate(valid_indices):
                duplicates = []
                for score, idx in zip(D[i], I[i]):
                    if idx == -1 or score < threshold:
                        continue
                    record = cls.get_by_id(str(idx))
                    if record:
                        rec = record.copy()
                        rec["similarity_score"] = round(float(score) * 100, 2)
                        duplicates.append(rec)
                    if len(duplicates) >= top_k:
                        break

                results_map[orig_idx] = {
                    "input_lead": query_leads[orig_idx],
                    "duplicates_found": len(duplicates),
                    "duplicates": duplicates
                }

        final = []
        for i, lead in enumerate(query_leads):
            final.append(results_map.get(i) or {
                "input_lead": lead,
                "duplicates_found": 0,
                "duplicates": []
            })
        return final