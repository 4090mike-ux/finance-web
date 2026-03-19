"""
JARVIS 자기 개선 시스템
- 피드백 학습
- 성능 메트릭 추적
- 자동 프롬프트 최적화
- 지식 갱신
- 실수 분석 및 교정
"""

import json
import time
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SelfImprovementSystem:
    """
    JARVIS 자기 개선 시스템
    지속적으로 더 나은 응답을 학습하고 개선
    """

    def __init__(self, db_path: str, llm_manager):
        self.db_path = db_path
        self.llm = llm_manager
        self._init_db()
        logger.info("SelfImprovementSystem initialized")

    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                response TEXT,
                rating INTEGER,
                feedback_text TEXT,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                context TEXT,
                timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                examples TEXT,
                effectiveness REAL,
                created_at REAL,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT,
                context TEXT,
                resolution TEXT,
                timestamp REAL
            )
        """)
        conn.commit()
        conn.close()

    # ==================== 피드백 처리 ====================

    def record_feedback(
        self,
        session_id: str,
        query: str,
        response: str,
        rating: int,  # 1-5
        feedback_text: str = "",
    ):
        """사용자 피드백 기록"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO feedback (session_id, query, response, rating, feedback_text, timestamp) VALUES (?,?,?,?,?,?)",
            (session_id, query, response[:2000], rating, feedback_text, time.time())
        )
        conn.commit()
        conn.close()

        # 낮은 평점은 즉시 분석
        if rating <= 2:
            self._analyze_poor_response(query, response, feedback_text)
        elif rating >= 4:
            self._learn_good_pattern(query, response)

        logger.info(f"Feedback recorded: rating={rating}/5")

    def _analyze_poor_response(self, query: str, response: str, feedback: str):
        """낮은 평점 응답 분석"""
        if not self.llm.primary:
            return

        try:
            from jarvis.llm.manager import Message
            messages = [
                Message(
                    role="user",
                    content=f"""다음 대화에서 무엇이 잘못되었는지 분석하세요:

사용자 질문: {query}
AI 응답: {response[:1000]}
사용자 피드백: {feedback}

무엇이 문제였고 어떻게 개선할 수 있을지 분석해주세요."""
                )
            ]
            analysis = self.llm.chat(messages, max_tokens=1024)

            # 에러 로그 저장
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO error_log (error_type, context, resolution, timestamp) VALUES (?,?,?,?)",
                ("poor_response", f"Q:{query[:200]}", analysis.content[:1000], time.time())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Analysis failed: {e}")

    def _learn_good_pattern(self, query: str, response: str):
        """좋은 응답 패턴 학습"""
        try:
            # 쿼리 유형 파악
            query_type = self._classify_query(query)

            conn = sqlite3.connect(self.db_path)
            existing = conn.execute(
                "SELECT id, examples, effectiveness FROM learned_patterns WHERE pattern_type=?",
                (query_type,)
            ).fetchone()

            if existing:
                # 기존 패턴 업데이트
                examples = json.loads(existing[1])
                examples.append({"q": query[:100], "a": response[:200]})
                examples = examples[-20:]  # 최근 20개만 유지

                conn.execute(
                    "UPDATE learned_patterns SET examples=?, effectiveness=?, updated_at=? WHERE id=?",
                    (json.dumps(examples, ensure_ascii=False),
                     min(existing[2] + 0.01, 1.0), time.time(), existing[0])
                )
            else:
                conn.execute(
                    "INSERT INTO learned_patterns (pattern_type, description, examples, effectiveness, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (query_type, f"{query_type} 유형 좋은 응답",
                     json.dumps([{"q": query[:100], "a": response[:200]}], ensure_ascii=False),
                     0.7, time.time(), time.time())
                )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Pattern learning failed: {e}")

    def _classify_query(self, query: str) -> str:
        """쿼리 유형 분류"""
        q = query.lower()
        if any(k in q for k in ["코드", "python", "프로그래밍", "함수"]): return "coding"
        if any(k in q for k in ["검색", "찾아", "최신", "뉴스"]): return "research"
        if any(k in q for k in ["계획", "전략", "어떻게 할"]): return "planning"
        if any(k in q for k in ["설명", "이해", "왜", "무엇"]): return "explanation"
        if any(k in q for k in ["분석", "비교", "평가"]): return "analysis"
        return "general"

    # ==================== 성능 메트릭 ====================

    def record_metric(self, metric_name: str, value: float, context: str = ""):
        """성능 지표 기록"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO performance_metrics (metric_name, metric_value, context, timestamp) VALUES (?,?,?,?)",
            (metric_name, value, context, time.time())
        )
        conn.commit()
        conn.close()

    def get_performance_report(self) -> Dict:
        """성능 보고서 생성"""
        conn = sqlite3.connect(self.db_path)

        # 평균 평점
        avg_rating = conn.execute(
            "SELECT AVG(rating) FROM feedback WHERE timestamp > ?",
            (time.time() - 86400 * 7,)  # 최근 7일
        ).fetchone()[0] or 0

        # 총 피드백 수
        total_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        # 학습된 패턴 수
        patterns = conn.execute("SELECT COUNT(*) FROM learned_patterns").fetchone()[0]

        # 에러 수
        errors = conn.execute(
            "SELECT COUNT(*) FROM error_log WHERE timestamp > ?",
            (time.time() - 86400,)
        ).fetchone()[0]

        # 응답 시간 메트릭
        avg_duration = conn.execute(
            "SELECT AVG(metric_value) FROM performance_metrics WHERE metric_name='response_duration'"
        ).fetchone()[0] or 0

        conn.close()

        return {
            "avg_rating": round(avg_rating, 2),
            "total_feedback": total_feedback,
            "learned_patterns": patterns,
            "recent_errors": errors,
            "avg_response_time": round(avg_duration, 2),
            "improvement_score": self._calculate_improvement_score(avg_rating, errors),
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_improvement_score(self, avg_rating: float, errors: int) -> float:
        """개선 점수 계산 (0-100)"""
        rating_score = (avg_rating / 5.0) * 70 if avg_rating else 50
        error_penalty = min(errors * 2, 30)
        return max(0, min(100, rating_score - error_penalty + 30))

    # ==================== 지식 갱신 ====================

    def update_knowledge_from_web(self, web_intel, topics: List[str] = None):
        """웹에서 최신 지식 갱신"""
        if not topics:
            topics = ["AI 최신 기술", "LLM 모델", "머신러닝 트렌드"]

        updates = []
        for topic in topics[:3]:  # 최대 3개
            try:
                results = web_intel.search_web(topic, max_results=3)
                for r in results:
                    if "error" not in r and r.get("snippet"):
                        updates.append({
                            "topic": topic,
                            "title": r.get("title", ""),
                            "content": r.get("snippet", ""),
                            "url": r.get("url", ""),
                        })
            except Exception as e:
                logger.error(f"Knowledge update failed for {topic}: {e}")

        return updates

    def suggest_improvements(self) -> List[str]:
        """자기 개선 제안 생성"""
        conn = sqlite3.connect(self.db_path)

        # 낮은 평점 쿼리 분석
        poor = conn.execute(
            "SELECT query, feedback_text FROM feedback WHERE rating <= 2 ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()

        conn.close()

        suggestions = []
        if poor:
            patterns = set()
            for query, feedback in poor:
                q_type = self._classify_query(query)
                patterns.add(q_type)

            for p in patterns:
                suggestions.append(f"{p} 유형 질문에 대한 응답 품질 개선 필요")

        if not suggestions:
            suggestions = [
                "응답 시간 최적화",
                "더 구체적인 예시 제공",
                "최신 정보 더 자주 검색",
                "코드 예시 품질 향상",
            ]

        return suggestions[:5]

    def get_learned_patterns(self) -> List[Dict]:
        """학습된 패턴 목록"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT pattern_type, description, effectiveness, updated_at FROM learned_patterns ORDER BY effectiveness DESC LIMIT 20"
        ).fetchall()
        conn.close()

        return [
            {
                "type": r[0],
                "description": r[1],
                "effectiveness": round(r[2], 3),
                "last_updated": datetime.fromtimestamp(r[3]).isoformat() if r[3] else "",
            }
            for r in rows
        ]
