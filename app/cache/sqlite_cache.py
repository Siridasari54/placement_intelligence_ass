import os
import sqlite3
import numpy as np
from typing import Tuple, Dict, Any, Optional
from app.embeddings.embedding_factory import embedding_model
from app.utils.logger import logger

class SQLitePersistentCache:
    def __init__(self, db_path: str = "data/cache/semantic_cache.db", threshold: float = 0.85):
        self.db_path = db_path
        self.threshold = threshold
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_db()
        logger.info(f"SQLite Persistent Cache initialized at: {db_path} with similarity threshold: {threshold}")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Exact match cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exact_cache (
                query TEXT PRIMARY KEY,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Semantic cache table with vector blob
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def get(self, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Retrieves cached response. 
        First checks exact match table (extremely fast O(1)).
        If missed, performs semantic vector search using NumPy cosine similarity.
        """
        query_clean = query.strip()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Exact Match Check
        cursor.execute("SELECT response FROM exact_cache WHERE query = ?", (query_clean,))
        row = cursor.fetchone()
        if row:
            conn.close()
            logger.info("Exact match cache HIT.")
            return row[0], {"score": 1.0, "type": "exact"}
            
        # 2. Semantic Match Check
        cursor.execute("SELECT query, response, embedding FROM semantic_cache")
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return None, {}
            
        # Generate query embedding using the shared singleton model
        query_embedding = np.array(embedding_model.embed_query(query_clean), dtype=np.float32)
        
        best_score = -1.0
        best_response = None
        best_query = None
        
        # Calculate similarity using NumPy vector operations
        for db_query, db_response, db_embedding_blob in rows:
            db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float32)
            
            # Cosine similarity formula: (A . B) / (||A|| * ||B||)
            dot_product = np.dot(query_embedding, db_embedding)
            norm_q = np.linalg.norm(query_embedding)
            norm_db = np.linalg.norm(db_embedding)
            
            if norm_q > 0 and norm_db > 0:
                score = dot_product / (norm_q * norm_db)
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                best_response = db_response
                best_query = db_query
                
        conn.close()
        
        if best_score >= self.threshold:
            logger.info(f"Semantic cache HIT (Score: {best_score:.4f}, Query: '{best_query}')")
            return best_response, {"score": float(best_score), "type": "semantic"}
            
        return None, {}

    def set(self, query: str, response: str) -> None:
        """Saves a query and response to both exact and semantic cache tables."""
        query_clean = query.strip()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Save to exact match cache
            cursor.execute(
                "INSERT OR REPLACE INTO exact_cache (query, response) VALUES (?, ?)",
                (query_clean, response)
            )
            
            # 2. Save to semantic cache
            query_embedding = np.array(embedding_model.embed_query(query_clean), dtype=np.float32)
            embedding_blob = query_embedding.tobytes()
            
            # Avoid inserting duplicate semantic records
            cursor.execute("SELECT id FROM semantic_cache WHERE query = ?", (query_clean,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute(
                    "UPDATE semantic_cache SET response = ?, embedding = ? WHERE id = ?",
                    (response, embedding_blob, existing[0])
                )
            else:
                cursor.execute(
                    "INSERT INTO semantic_cache (query, response, embedding) VALUES (?, ?, ?)",
                    (query_clean, response, embedding_blob)
                )
                
            conn.commit()
            logger.info(f"Saved query to SQLite Persistent Cache.")
        except Exception as e:
            logger.error(f"Error saving to SQLite persistent cache: {e}")
        finally:
            conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM exact_cache")
        cursor.execute("DELETE FROM semantic_cache")
        conn.commit()
        conn.close()
        logger.info("SQLite cache database cleared.")
