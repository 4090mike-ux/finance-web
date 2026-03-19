"""
JARVIS 메모리 시스템
- SQLite: 대화 히스토리, 사용자 프로파일, 태스크 이력
- ChromaDB: 벡터 기반 시맨틱 검색
- In-memory: 세션 컨텍스트
"""

import sqlite3
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    JARVIS 통합 메모리 관리자
    - 대화 히스토리 (SQLite)
    - 벡터 의미 검색 (ChromaDB)
    - 지식 베이스 (SQLite + 벡터)
    - 사용자 프로파일 (SQLite)
    """

    def __init__(self, db_path: str, chroma_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.embedding_model_name = embedding_model

        # SQLite 초기화
        self._init_sqlite()

        # ChromaDB + 임베딩 초기화 (비동기로 로드)
        self.chroma_client = None
        self.collection = None
        self.embedder = None
        self._init_vector_db()

        # 세션 메모리 (인메모리)
        self.session_context = []
        self.session_id = f"session_{int(time.time())}"

        logger.info(f"MemoryManager initialized - session: {self.session_id}")

    def _init_sqlite(self):
        """SQLite 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                title TEXT,
                content TEXT,
                source TEXT,
                timestamp REAL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                description TEXT,
                result TEXT,
                status TEXT,
                timestamp REAL,
                duration REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_category ON knowledge_base(category)
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite database initialized")

    def _init_vector_db(self):
        """ChromaDB 벡터 DB 초기화"""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="jarvis_memory",
                metadata={"hnsw:space": "cosine"}
            )
            self.embedder = SentenceTransformer(self.embedding_model_name)
            logger.info("ChromaDB vector store initialized")
        except Exception as e:
            logger.warning(f"ChromaDB init failed (will use keyword search): {e}")
            self.chroma_client = None

    # ==================== 대화 히스토리 ====================

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """대화 메시지 저장"""
        timestamp = time.time()

        # SQLite 저장
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, timestamp, metadata) VALUES (?,?,?,?,?)",
            (self.session_id, role, content, timestamp, json.dumps(metadata or {}))
        )
        conn.commit()
        conn.close()

        # 세션 컨텍스트 업데이트
        self.session_context.append({"role": role, "content": content, "timestamp": timestamp})

        # 벡터 DB에 저장
        if self.collection and self.embedder and role == "assistant":
            try:
                embedding = self.embedder.encode([content])[0].tolist()
                doc_id = f"conv_{self.session_id}_{int(timestamp)}"
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"role": role, "session": self.session_id, "timestamp": timestamp}]
                )
            except Exception as e:
                logger.debug(f"Vector store add failed: {e}")

    def get_conversation_history(self, limit: int = 50, session_id: str = None) -> List[Dict]:
        """대화 히스토리 조회"""
        sid = session_id or self.session_id
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT role, content, timestamp FROM conversations WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (sid, limit)
        ).fetchall()
        conn.close()

        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    def get_recent_context(self, n: int = 20) -> List[Dict]:
        """최근 N개 메시지 (세션 내)"""
        return self.session_context[-n:] if self.session_context else []

    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """시맨틱 유사도 검색"""
        if not self.collection or not self.embedder:
            return self._keyword_search(query, n_results)

        try:
            query_embedding = self.embedder.encode([query])[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count() or 1)
            )

            output = []
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                })
            return output
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
            return self._keyword_search(query, n_results)

    def _keyword_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """키워드 기반 폴백 검색"""
        conn = sqlite3.connect(self.db_path)
        words = query.lower().split()
        conditions = " OR ".join([f"LOWER(content) LIKE '%{w}%'" for w in words[:3]])
        rows = conn.execute(
            f"SELECT role, content, timestamp FROM conversations WHERE {conditions} ORDER BY timestamp DESC LIMIT ?",
            (n_results,)
        ).fetchall()
        conn.close()

        return [{"content": r[1], "metadata": {"role": r[0], "timestamp": r[2]}, "distance": 0.5}
                for r in rows]

    # ==================== 지식 베이스 ====================

    def add_knowledge(self, category: str, title: str, content: str, source: str = "", metadata: Dict = None):
        """지식 추가"""
        timestamp = time.time()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO knowledge_base (category, title, content, source, timestamp, metadata) VALUES (?,?,?,?,?,?)",
            (category, title, content, source, timestamp, json.dumps(metadata or {}))
        )
        conn.commit()
        conn.close()

        # 벡터 저장
        if self.collection and self.embedder:
            try:
                combined = f"{title}\n{content}"
                embedding = self.embedder.encode([combined])[0].tolist()
                doc_id = f"kb_{category}_{int(timestamp)}"
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[combined],
                    metadatas=[{"category": category, "title": title, "source": source}]
                )
            except Exception as e:
                logger.debug(f"Knowledge vector store failed: {e}")

    def get_knowledge(self, category: str = None, limit: int = 20) -> List[Dict]:
        """지식 조회"""
        conn = sqlite3.connect(self.db_path)
        if category:
            rows = conn.execute(
                "SELECT category, title, content, source, timestamp FROM knowledge_base WHERE category=? ORDER BY timestamp DESC LIMIT ?",
                (category, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT category, title, content, source, timestamp FROM knowledge_base ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()

        return [{"category": r[0], "title": r[1], "content": r[2], "source": r[3], "timestamp": r[4]}
                for r in rows]

    # ==================== 사용자 프로파일 ====================

    def set_profile(self, key: str, value: Any):
        """사용자 프로파일 설정"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO user_profile (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value), time.time())
        )
        conn.commit()
        conn.close()

    def get_profile(self, key: str, default=None) -> Any:
        """사용자 프로파일 조회"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT value FROM user_profile WHERE key=?", (key,)
        ).fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return default

    def get_all_profile(self) -> Dict:
        """전체 프로파일 조회"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT key, value FROM user_profile").fetchall()
        conn.close()
        return {r[0]: json.loads(r[1]) for r in rows}

    # ==================== 태스크 이력 ====================

    def log_task(self, task_type: str, description: str, result: str, status: str, duration: float = 0):
        """태스크 실행 이력 기록"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO task_history (task_type, description, result, status, timestamp, duration) VALUES (?,?,?,?,?,?)",
            (task_type, description, result[:2000], status, time.time(), duration)
        )
        conn.commit()
        conn.close()

    def get_task_history(self, limit: int = 20) -> List[Dict]:
        """태스크 이력 조회"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT task_type, description, result, status, timestamp, duration FROM task_history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()

        return [
            {"task_type": r[0], "description": r[1], "result": r[2],
             "status": r[3], "timestamp": r[4], "duration": r[5]}
            for r in rows
        ]

    # ==================== 통계 ====================

    def get_stats(self) -> Dict:
        """메모리 통계"""
        conn = sqlite3.connect(self.db_path)
        total_conv = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        total_kb = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        total_tasks = conn.execute("SELECT COUNT(*) FROM task_history").fetchone()[0]
        conn.close()

        vector_count = 0
        if self.collection:
            try:
                vector_count = self.collection.count()
            except:
                pass

        return {
            "conversations": total_conv,
            "knowledge_base": total_kb,
            "tasks": total_tasks,
            "vector_embeddings": vector_count,
            "session_context_size": len(self.session_context),
            "session_id": self.session_id,
        }
