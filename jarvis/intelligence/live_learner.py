"""
JARVIS 실시간 학습기 — Iteration 10
최신 AI 논문/GitHub/뉴스에서 실시간으로 학습한다

영감:
  - 지속적 학습 (Continual Learning)
  - 온라인 학습 (Online Learning)
  - 지식 베이스 갱신 (Knowledge Base Updates)
  - 메타 학습 (Meta-Learning from Papers)

핵심 개념:
  JARVIS는 잠들어 있지 않는다 — 항상 최신 지식을 흡수한다
  ArXiv → GitHub → HuggingFace → 내부 지식 베이스
  LLM으로 핵심 통찰 추출 → 중요도 점수 부여 → SQLite 저장
"""

import json
import time
import sqlite3
import threading
import logging
import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeSource:
    """학습 소스 정보"""
    source_id: str
    name: str                           # 소스 이름 (예: "arxiv_cs_ai")
    url: str                            # 소스 URL
    source_type: str                    # "arxiv", "github", "huggingface", "news"
    last_fetched: float = 0.0           # 마지막 페치 타임스탬프
    fetch_count: int = 0                # 총 페치 횟수
    insight_count: int = 0              # 추출된 통찰 수

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LearnedInsight:
    """추출된 지식 통찰"""
    insight_id: str
    topic: str                          # 주제
    insight: str                        # 핵심 통찰 내용
    source_url: str                     # 출처 URL
    source_type: str                    # "arxiv", "github", "huggingface"
    importance_score: float             # 중요도 (0~1)
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    paper_title: str = ""               # 논문/리포 제목

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ════════════════════════════════════════════════════════════════
# 실시간 학습기
# ════════════════════════════════════════════════════════════════

class LiveLearner:
    """
    JARVIS 실시간 학습기

    ArXiv, GitHub Trending, HuggingFace Papers에서 주기적으로
    최신 AI/ML 지식을 수집하고 LLM으로 핵심 통찰을 추출한다.
    """

    # ArXiv API 엔드포인트
    ARXIV_API = "http://export.arxiv.org/api/query"
    # GitHub Trending (비공식 스크래핑)
    GITHUB_TRENDING = "https://github.com/trending/python?since=daily"
    # Semantic Scholar
    SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        db_path: str = "",
    ):
        """
        Args:
            llm_manager: LLMManager 인스턴스
            event_callback: 학습 이벤트 콜백
            db_path: SQLite DB 경로
        """
        self.llm = llm_manager
        self.event_callback = event_callback
        self._lock = threading.Lock()

        # 학습 소스 목록
        self.sources: Dict[str, KnowledgeSource] = {
            "arxiv_cs_ai": KnowledgeSource(
                source_id="arxiv_cs_ai",
                name="ArXiv CS.AI",
                url="http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=10&sortBy=lastUpdatedDate",
                source_type="arxiv",
            ),
            "arxiv_cs_lg": KnowledgeSource(
                source_id="arxiv_cs_lg",
                name="ArXiv CS.LG",
                url="http://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results=10&sortBy=lastUpdatedDate",
                source_type="arxiv",
            ),
            "arxiv_cs_cl": KnowledgeSource(
                source_id="arxiv_cs_cl",
                name="ArXiv CS.CL",
                url="http://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=10&sortBy=lastUpdatedDate",
                source_type="arxiv",
            ),
            "github_trending": KnowledgeSource(
                source_id="github_trending",
                name="GitHub Trending Python",
                url=self.GITHUB_TRENDING,
                source_type="github",
            ),
        }

        # DB 초기화
        if not db_path:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "live_learner.db")
        self.db_path = db_path
        self._init_db()

        # 백그라운드 스레드
        self._bg_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info("LiveLearner 초기화 완료")

    # ── DB 초기화 ──────────────────────────────────────────────

    def _init_db(self):
        """SQLite 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    topic TEXT,
                    insight TEXT,
                    source_url TEXT,
                    source_type TEXT,
                    importance_score REAL,
                    tags TEXT,
                    timestamp REAL,
                    paper_title TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,
                    fetched_at REAL,
                    count INTEGER,
                    success INTEGER
                )
            """)
            conn.commit()

    # ── ArXiv 페치 ────────────────────────────────────────────

    def fetch_arxiv_papers(self, category: str = "cs.AI", max_papers: int = 10) -> List[Dict]:
        """
        ArXiv API에서 최신 논문을 가져온다

        Args:
            category: ArXiv 카테고리 (cs.AI, cs.LG, cs.CL)
            max_papers: 최대 논문 수

        Returns:
            논문 정보 딕셔너리 목록
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests 라이브러리 없음 — ArXiv 페치 불가")
            return []

        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query=cat:{category}"
            f"&start=0&max_results={max_papers}"
            f"&sortBy=lastUpdatedDate&sortOrder=descending"
        )

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return self._parse_arxiv_xml(resp.text)
        except Exception as e:
            logger.warning(f"ArXiv 페치 실패 ({category}): {e}")
            return []

    def _parse_arxiv_xml(self, xml_text: str) -> List[Dict]:
        """ArXiv Atom XML 파싱 (표준 라이브러리 xml.etree만 사용)"""
        try:
            import xml.etree.ElementTree as ET
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            root = ET.fromstring(xml_text)
            papers = []

            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                updated_el = entry.find("atom:updated", ns)

                title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
                summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                paper_id = id_el.text.strip() if id_el is not None else ""
                updated = updated_el.text.strip() if updated_el is not None else ""

                if title and summary:
                    papers.append({
                        "title": title,
                        "abstract": summary[:1000],
                        "url": paper_id,
                        "updated": updated,
                        "source": "arxiv",
                    })

            return papers
        except Exception as e:
            logger.warning(f"ArXiv XML 파싱 실패: {e}")
            return []

    # ── GitHub Trending 페치 ───────────────────────────────────

    def fetch_github_trending(self) -> List[Dict]:
        """
        GitHub Trending에서 인기 Python AI 리포지토리를 가져온다

        Returns:
            리포 정보 딕셔너리 목록
        """
        if not REQUESTS_AVAILABLE:
            return []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml",
            }
            resp = requests.get(self.GITHUB_TRENDING, timeout=15, headers=headers)
            resp.raise_for_status()
            return self._parse_github_html(resp.text)
        except Exception as e:
            logger.warning(f"GitHub Trending 페치 실패: {e}")
            return []

    def _parse_github_html(self, html: str) -> List[Dict]:
        """GitHub HTML에서 리포 정보 파싱 (정규식 기반)"""
        repos = []
        try:
            # repo 이름 패턴
            repo_pattern = re.compile(
                r'<h2[^>]*class="[^"]*h3[^"]*"[^>]*>\s*<a[^>]*href="/([^"]+)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            desc_pattern = re.compile(
                r'<p[^>]*class="[^"]*col-9[^"]*"[^>]*>(.*?)</p>',
                re.DOTALL,
            )

            repo_matches = repo_pattern.findall(html)
            desc_matches = desc_pattern.findall(html)

            for i, (repo_path, repo_name) in enumerate(repo_matches[:15]):
                clean_name = re.sub(r'\s+', ' ', repo_name).strip()
                desc = ""
                if i < len(desc_matches):
                    desc = re.sub(r'<[^>]+>', '', desc_matches[i]).strip()
                    desc = re.sub(r'\s+', ' ', desc).strip()

                if clean_name and "/" in repo_path:
                    repos.append({
                        "title": clean_name,
                        "description": desc[:200],
                        "url": f"https://github.com/{repo_path}",
                        "source": "github",
                    })

            # 파싱 실패 시 최소한의 정보라도 반환
            if not repos:
                # 간단한 대안 패턴
                simple_pattern = re.compile(r'href="/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)"[^>]*class="[^"]*Link[^"]*"')
                simple_matches = simple_pattern.findall(html)
                for repo_path in simple_matches[:10]:
                    if repo_path.count("/") == 1:
                        repos.append({
                            "title": repo_path,
                            "description": "GitHub Trending Repository",
                            "url": f"https://github.com/{repo_path}",
                            "source": "github",
                        })

        except Exception as e:
            logger.warning(f"GitHub HTML 파싱 실패: {e}")

        return repos

    # ── 통찰 추출 ──────────────────────────────────────────────

    def extract_insights(self, raw_content: List[Dict]) -> List[LearnedInsight]:
        """
        원시 콘텐츠에서 LLM으로 핵심 통찰을 추출한다

        Args:
            raw_content: fetch_arxiv_papers 또는 fetch_github_trending의 결과

        Returns:
            LearnedInsight 목록
        """
        insights = []

        for item in raw_content[:10]:  # 한 번에 최대 10개
            try:
                insight = self._extract_single_insight(item)
                if insight:
                    insights.append(insight)
            except Exception as e:
                logger.warning(f"통찰 추출 실패: {e}")

        return insights

    def _extract_single_insight(self, item: Dict) -> Optional[LearnedInsight]:
        """단일 콘텐츠에서 통찰 추출"""
        title = item.get("title", "")
        abstract = item.get("abstract", item.get("description", ""))
        url = item.get("url", "")
        source_type = item.get("source", "unknown")

        if not title:
            return None

        # 이미 처리된 항목인지 확인 (URL 기반)
        if self._is_already_learned(url):
            return None

        # 중요도 점수 계산 (간단한 휴리스틱)
        importance = self._calculate_importance(title, abstract)

        # LLM으로 핵심 통찰 추출
        if self.llm and abstract:
            insight_text = self._llm_extract_insight(title, abstract)
        else:
            # LLM 없이 제목 기반 요약
            insight_text = f"{title}: {abstract[:200]}..." if abstract else title

        # 태그 추출
        tags = self._extract_tags(title + " " + abstract)

        insight_id = hashlib.md5(url.encode() if url else title.encode()).hexdigest()[:12]

        return LearnedInsight(
            insight_id=insight_id,
            topic=self._extract_topic(title),
            insight=insight_text,
            source_url=url,
            source_type=source_type,
            importance_score=importance,
            tags=tags,
            paper_title=title,
        )

    def _llm_extract_insight(self, title: str, abstract: str) -> str:
        """LLM으로 논문 핵심 통찰 추출"""
        try:
            prompt = f"""다음 AI/ML 논문/리포에서 핵심 통찰 1-2문장을 한국어로 추출하라.

제목: {title}
내용: {abstract[:600]}

핵심 통찰 (한국어, 2문장 이내):"""

            result = self.llm.complete(prompt, max_tokens=200)
            return result.strip()
        except Exception:
            return f"{title}: {abstract[:150]}..."

    def _calculate_importance(self, title: str, abstract: str) -> float:
        """중요도 점수 계산 (0~1)"""
        text = (title + " " + abstract).lower()
        score = 0.3  # 기본 점수

        # 핵심 AI 키워드 가중치
        high_value_terms = [
            "state-of-the-art", "sota", "breakthrough", "novel", "transformer",
            "large language model", "llm", "gpt", "claude", "gemini",
            "reinforcement learning", "multimodal", "agent", "reasoning",
            "새로운", "획기적", "성능 향상",
        ]
        medium_value_terms = [
            "improve", "efficient", "faster", "benchmark", "evaluation",
            "dataset", "fine-tuning", "quantization", "training",
        ]

        for term in high_value_terms:
            if term in text:
                score += 0.12

        for term in medium_value_terms:
            if term in text:
                score += 0.05

        return min(1.0, score)

    def _extract_topic(self, title: str) -> str:
        """제목에서 주제 카테고리 추출"""
        title_lower = title.lower()
        topic_map = {
            "Large Language Model": ["llm", "language model", "gpt", "claude", "gemini", "bert"],
            "Reinforcement Learning": ["reinforcement", "rl", "reward", "policy"],
            "Computer Vision": ["vision", "image", "visual", "cnn", "detection"],
            "Natural Language Processing": ["nlp", "text", "translation", "summarization"],
            "Multimodal AI": ["multimodal", "vision-language", "audio", "speech"],
            "AI Agents": ["agent", "autonomous", "tool use", "planning"],
            "AI Safety": ["safety", "alignment", "bias", "fairness", "robustness"],
            "Efficient AI": ["efficient", "quantization", "pruning", "compression", "fast"],
            "GitHub Repository": [],
        }

        for topic, keywords in topic_map.items():
            for kw in keywords:
                if kw in title_lower:
                    return topic

        return "General AI Research"

    def _extract_tags(self, text: str) -> List[str]:
        """텍스트에서 태그 추출"""
        text_lower = text.lower()
        all_tags = [
            "llm", "transformer", "reinforcement-learning", "computer-vision",
            "nlp", "multimodal", "agent", "fine-tuning", "benchmark",
            "dataset", "efficient", "safety", "alignment", "generation",
            "reasoning", "planning", "tool-use", "knowledge-graph",
        ]
        return [tag for tag in all_tags if tag in text_lower][:5]

    def _is_already_learned(self, url: str) -> bool:
        """이미 학습한 항목인지 확인"""
        if not url:
            return False
        try:
            insight_id = hashlib.md5(url.encode()).hexdigest()[:12]
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT 1 FROM insights WHERE insight_id = ?", (insight_id,)
                ).fetchone()
            return row is not None
        except Exception:
            return False

    # ── 지식 베이스 업데이트 ───────────────────────────────────

    def update_knowledge_base(self, insights: List[LearnedInsight]) -> int:
        """
        추출된 통찰을 SQLite 지식 베이스에 저장한다

        Args:
            insights: 저장할 통찰 목록

        Returns:
            실제로 추가된 통찰 수
        """
        added = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                for insight in insights:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO insights
                            (insight_id, topic, insight, source_url, source_type,
                             importance_score, tags, timestamp, paper_title)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            insight.insight_id,
                            insight.topic,
                            insight.insight,
                            insight.source_url,
                            insight.source_type,
                            insight.importance_score,
                            json.dumps(insight.tags),
                            insight.timestamp,
                            insight.paper_title,
                        ))
                        added += 1
                    except sqlite3.IntegrityError:
                        pass  # 이미 존재
                conn.commit()

            if added > 0:
                logger.info(f"지식 베이스 업데이트: +{added}개 통찰")
                self._emit_event("knowledge_updated", {"added": added})

        except Exception as e:
            logger.warning(f"지식 베이스 업데이트 실패: {e}")

        return added

    # ── 학습 실행 ──────────────────────────────────────────────

    def run_learning_cycle(self) -> Dict[str, Any]:
        """
        한 번의 완전한 학습 사이클을 실행한다
        (ArXiv + GitHub 페치 → 통찰 추출 → DB 저장)

        Returns:
            학습 결과 요약
        """
        logger.info("학습 사이클 시작...")
        self._emit_event("learning_started", {"timestamp": time.time()})

        all_raw = []
        results = {"arxiv": 0, "github": 0, "total_insights": 0}

        # ArXiv 페치
        for category in ["cs.AI", "cs.LG", "cs.CL"]:
            papers = self.fetch_arxiv_papers(category, max_papers=5)
            all_raw.extend(papers)
            results["arxiv"] += len(papers)
            if papers:
                logger.info(f"ArXiv {category}: {len(papers)}편 로드")

        # GitHub Trending 페치
        repos = self.fetch_github_trending()
        all_raw.extend(repos)
        results["github"] += len(repos)
        if repos:
            logger.info(f"GitHub Trending: {len(repos)}개 리포 로드")

        # 통찰 추출
        insights = self.extract_insights(all_raw)

        # DB 저장
        added = self.update_knowledge_base(insights)
        results["total_insights"] = added

        self._emit_event("learning_completed", results)
        logger.info(f"학습 사이클 완료 — 총 {added}개 통찰 추가")
        return results

    # ── 조회 ──────────────────────────────────────────────────

    def get_recent_insights(self, n: int = 10) -> List[LearnedInsight]:
        """
        최근 학습된 통찰 목록 반환

        Args:
            n: 반환할 통찰 수

        Returns:
            LearnedInsight 목록 (최신순)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT insight_id, topic, insight, source_url, source_type,
                           importance_score, tags, timestamp, paper_title
                    FROM insights
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (n,)).fetchall()

            insights = []
            for row in rows:
                insights.append(LearnedInsight(
                    insight_id=row[0],
                    topic=row[1],
                    insight=row[2],
                    source_url=row[3],
                    source_type=row[4],
                    importance_score=row[5],
                    tags=json.loads(row[6]) if row[6] else [],
                    timestamp=row[7],
                    paper_title=row[8] or "",
                ))
            return insights
        except Exception as e:
            logger.warning(f"통찰 조회 실패: {e}")
            return []

    def get_learning_summary(self) -> Dict[str, Any]:
        """학습 통계 요약 반환"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
                by_source = conn.execute(
                    "SELECT source_type, COUNT(*) FROM insights GROUP BY source_type"
                ).fetchall()
                by_topic = conn.execute(
                    "SELECT topic, COUNT(*) FROM insights GROUP BY topic ORDER BY COUNT(*) DESC LIMIT 5"
                ).fetchall()
                avg_importance = conn.execute(
                    "SELECT AVG(importance_score) FROM insights"
                ).fetchone()[0]

            return {
                "total_insights": total,
                "by_source": dict(by_source),
                "top_topics": [{"topic": t, "count": c} for t, c in by_topic],
                "avg_importance": round(avg_importance or 0, 3),
                "sources_active": len(self.sources),
                "last_updated": time.time(),
            }
        except Exception as e:
            logger.warning(f"학습 요약 실패: {e}")
            return {
                "total_insights": 0,
                "by_source": {},
                "top_topics": [],
                "avg_importance": 0.0,
                "sources_active": len(self.sources),
            }

    # ── 자동 학습 시작 ─────────────────────────────────────────

    def start_auto_learn(self, interval_minutes: int = 60):
        """
        백그라운드에서 주기적으로 학습을 실행한다

        Args:
            interval_minutes: 학습 간격 (분)
        """
        if self._running:
            logger.info("LiveLearner 이미 실행 중")
            return

        self._running = True
        self._bg_thread = threading.Thread(
            target=self._auto_learn_loop,
            args=(interval_minutes * 60,),
            daemon=True,
            name="live-learner-bg",
        )
        self._bg_thread.start()
        logger.info(f"자동 학습 시작 (간격: {interval_minutes}분)")

        # 최초 학습 즉시 실행 (비동기)
        init_thread = threading.Thread(
            target=self._safe_learn_cycle,
            daemon=True,
            name="live-learner-init",
        )
        init_thread.start()

    def _auto_learn_loop(self, interval_seconds: float):
        """자동 학습 백그라운드 루프"""
        while self._running:
            time.sleep(interval_seconds)
            if not self._running:
                break
            self._safe_learn_cycle()

    def _safe_learn_cycle(self):
        """오류 안전한 학습 사이클 실행"""
        try:
            self.run_learning_cycle()
        except Exception as e:
            logger.warning(f"학습 사이클 오류: {e}")

    def stop(self):
        """자동 학습 중지"""
        self._running = False
        logger.info("LiveLearner 중지")

    # ── 내부 유틸 ──────────────────────────────────────────────

    def _emit_event(self, event_type: str, data: Dict):
        """이벤트 콜백 호출"""
        if self.event_callback:
            try:
                self.event_callback({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
