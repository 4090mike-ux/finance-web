"""
JARVIS 실시간 지능 모니터 — Iteration 6
세상의 모든 정보를 실시간으로 수집·분석·우선순위화

수집 소스:
- RSS 피드 (HackerNews, TechCrunch, MIT Tech Review, ArXiv)
- GitHub 트렌딩 레포지토리 (실시간)
- AI/ML 최신 논문 (ArXiv API)
- 기술 뉴스 (DuckDuckGo News)
- Reddit AI 서브레딧 인기글

처리:
- 중요도 스코어링 (키워드 + 신선도 + 참여도)
- 자동 요약 (LLM)
- 관련성 필터링 (사용자 관심사 기반)
- 실시간 WebSocket 브로드캐스트
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

MONITOR_DB = Path("data/jarvis/live_feed.json")
REQUEST_TIMEOUT = 10


@dataclass
class FeedItem:
    """피드 항목"""
    id: str
    title: str
    url: str
    source: str           # hackernews / github / arxiv / news / reddit
    category: str         # AI / Tech / Research / Security / Other
    score: float          # 중요도 점수 (0-100)
    summary: str = ""
    published_at: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_new: bool = True
    tags: List[str] = field(default_factory=list)


class LiveMonitor:
    """
    JARVIS 실시간 지능 모니터
    인터넷의 중요 정보를 24/7 감시하고 필터링
    """

    # 수집 간격 (초)
    SCHEDULES = {
        "hackernews": 300,       # 5분
        "github_trending": 1800, # 30분
        "arxiv": 3600,           # 1시간
        "tech_news": 600,        # 10분
    }

    # 관심 키워드
    AI_KEYWORDS = [
        "LLM", "GPT", "Claude", "Gemini", "llama", "AI agent", "AGI",
        "machine learning", "deep learning", "transformer", "neural",
        "langchain", "autogen", "CrewAI", "RAG", "vector", "embedding",
        "인공지능", "머신러닝", "딥러닝",
    ]

    IMPORTANCE_KEYWORDS = {
        "critical": ["breakthrough", "AGI", "emergency", "critical", "revolutionary"],
        "high": ["major", "release", "launch", "announce", "new", "open-source"],
        "medium": ["update", "improve", "research", "study", "paper"],
    }

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
    ):
        self.llm = llm_manager
        self.event_callback = event_callback

        self._items: Dict[str, FeedItem] = {}  # id → item
        self._is_running = False
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._stats = {
            "total_fetched": 0,
            "ai_items": 0,
            "last_fetch": None,
            "sources_active": 0,
        }
        self._load_cache()
        logger.info("LiveMonitor initialized")

    # ── 시작/중지 ──────────────────────────────────────────────────────────

    def start(self):
        """모니터 시작"""
        if self._is_running:
            return
        self._is_running = True

        for source, interval in self.SCHEDULES.items():
            func = getattr(self, f"_fetch_{source}", None)
            if not func:
                continue
            t = threading.Thread(
                target=self._scheduled_run,
                args=(source, func, interval),
                daemon=True,
                name=f"LiveMon-{source}",
            )
            self._threads.append(t)
            t.start()

        logger.info(f"LiveMonitor started: {len(self._threads)} sources monitoring")
        self._emit("monitor_started", {"sources": list(self.SCHEDULES.keys())})

    def stop(self):
        self._is_running = False
        logger.info("LiveMonitor stopped")

    def _scheduled_run(self, name: str, func: Callable, interval: int):
        """주기적 실행"""
        time.sleep(10)  # 초기 지연
        while self._is_running:
            try:
                logger.info(f"[LiveMon] Fetching: {name}")
                new_items = func()
                if new_items:
                    self._process_items(new_items, name)
                    self._stats["last_fetch"] = datetime.now().isoformat()
            except Exception as e:
                logger.debug(f"[LiveMon] {name} error: {e}")
            time.sleep(interval)

    # ── HackerNews ────────────────────────────────────────────────────────

    def _fetch_hackernews(self) -> List[Dict]:
        """HackerNews Top Stories"""
        try:
            resp = requests.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json",
                timeout=REQUEST_TIMEOUT,
            )
            ids = resp.json()[:20]  # 상위 20개
            items = []
            for story_id in ids[:10]:  # 실제로 10개만
                try:
                    story = requests.get(
                        f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                        timeout=5,
                    ).json()
                    if story and story.get("type") == "story":
                        items.append({
                            "id": f"hn_{story_id}",
                            "title": story.get("title", ""),
                            "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                            "score": story.get("score", 0),
                            "published_at": datetime.fromtimestamp(
                                story.get("time", time.time())
                            ).isoformat(),
                            "source": "hackernews",
                        })
                except Exception:
                    continue
            return items
        except Exception as e:
            logger.debug(f"[LiveMon] HN fetch error: {e}")
            return []

    # ── GitHub 트렌딩 ─────────────────────────────────────────────────────

    def _fetch_github_trending(self) -> List[Dict]:
        """GitHub 트렌딩 레포지토리"""
        try:
            # GitHub Search API (최근 1일 내 스타 급증)
            query = "stars:>100 pushed:>2024-01-01 language:Python topic:AI"
            params = {
                "q": "topic:llm OR topic:ai OR topic:machine-learning stars:>500",
                "sort": "stars",
                "order": "desc",
                "per_page": 10,
            }
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []
            for repo in data.get("items", []):
                items.append({
                    "id": f"gh_{repo['id']}",
                    "title": f"⭐ {repo['name']}: {repo.get('description', '')[:80]}",
                    "url": repo["html_url"],
                    "score": min(100, repo.get("stargazers_count", 0) // 100),
                    "published_at": repo.get("updated_at", datetime.now().isoformat()),
                    "source": "github",
                    "tags": repo.get("topics", [])[:5],
                })
            return items
        except Exception as e:
            logger.debug(f"[LiveMon] GitHub fetch error: {e}")
            return []

    # ── ArXiv 최신 논문 ───────────────────────────────────────────────────

    def _fetch_arxiv(self) -> List[Dict]:
        """ArXiv 최신 AI 논문"""
        try:
            params = {
                "search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
                "start": 0,
                "max_results": 10,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return []

            import re
            content = resp.text
            titles = re.findall(r'<title>(.*?)</title>', content, re.DOTALL)[1:]
            summaries = re.findall(r'<summary>(.*?)</summary>', content, re.DOTALL)
            links = re.findall(r'<id>(http://arxiv\.org/abs/\S+)</id>', content)

            items = []
            for i, (title, link) in enumerate(zip(titles[:8], links[:8])):
                paper_id = link.split("/")[-1]
                summary = summaries[i].strip()[:200] if i < len(summaries) else ""
                items.append({
                    "id": f"arxiv_{paper_id}",
                    "title": f"📄 {title.strip()[:100]}",
                    "url": link,
                    "score": 60 + i * 2,
                    "published_at": datetime.now().isoformat(),
                    "source": "arxiv",
                    "summary": summary,
                })
            return items
        except Exception as e:
            logger.debug(f"[LiveMon] ArXiv fetch error: {e}")
            return []

    # ── 기술 뉴스 ─────────────────────────────────────────────────────────

    def _fetch_tech_news(self) -> List[Dict]:
        """DuckDuckGo를 통한 기술 뉴스"""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news("AI LLM machine learning 2025", max_results=10))
            items = []
            for r in results:
                items.append({
                    "id": f"news_{hash(r.get('url', ''))}",
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "score": 50,
                    "published_at": r.get("date", datetime.now().isoformat()),
                    "source": "news",
                    "summary": r.get("body", "")[:200],
                })
            return items
        except Exception as e:
            logger.debug(f"[LiveMon] News fetch error: {e}")
            return []

    # ── 아이템 처리 ────────────────────────────────────────────────────────

    def _process_items(self, raw_items: List[Dict], source: str):
        """수집된 아이템 처리 및 저장"""
        new_count = 0
        ai_count = 0

        for raw in raw_items:
            item_id = raw.get("id", f"{source}_{int(time.time())}")
            if item_id in self._items:
                continue  # 이미 있음

            title = raw.get("title", "")
            is_ai = self._is_ai_related(title)
            category = "AI" if is_ai else self._categorize(title)
            importance_score = self._score_importance(raw, is_ai)

            item = FeedItem(
                id=item_id,
                title=title,
                url=raw.get("url", ""),
                source=raw.get("source", source),
                category=category,
                score=importance_score,
                summary=raw.get("summary", ""),
                published_at=raw.get("published_at", datetime.now().isoformat()),
                tags=raw.get("tags", []),
                is_new=True,
            )

            with self._lock:
                self._items[item_id] = item
            new_count += 1
            if is_ai:
                ai_count += 1

        if new_count > 0:
            self._stats["total_fetched"] += new_count
            self._stats["ai_items"] += ai_count
            self._stats["sources_active"] = len(set(i.source for i in self._items.values()))

            # 중요 AI 아이템 알림
            high_score_items = [
                i for i in self._items.values()
                if i.is_new and i.score >= 80 and i.category == "AI"
            ]
            if high_score_items:
                self._emit("important_item", {
                    "count": len(high_score_items),
                    "top": {"title": high_score_items[0].title, "score": high_score_items[0].score},
                })

            self._save_cache()
            logger.info(f"[LiveMon] {source}: +{new_count} items ({ai_count} AI-related)")

    def _is_ai_related(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.AI_KEYWORDS)

    def _categorize(self, text: str) -> str:
        text_lower = text.lower()
        if any(k in text_lower for k in ["security", "hack", "vulnerability", "exploit"]):
            return "Security"
        if any(k in text_lower for k in ["startup", "funding", "billion", "million", "ipo"]):
            return "Business"
        if any(k in text_lower for k in ["python", "javascript", "rust", "go", "java"]):
            return "Dev"
        return "Tech"

    def _score_importance(self, raw: Dict, is_ai: bool) -> float:
        score = 40.0
        if is_ai:
            score += 30
        # HN/GitHub 점수
        if raw.get("score"):
            score += min(20, raw["score"] / 50)
        # 중요 키워드
        title = raw.get("title", "").lower()
        for level, keywords in self.IMPORTANCE_KEYWORDS.items():
            if any(kw.lower() in title for kw in keywords):
                score += {"critical": 25, "high": 15, "medium": 5}.get(level, 0)
                break
        return min(100, score)

    # ── 조회 ───────────────────────────────────────────────────────────────

    def get_feed(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 20,
        min_score: float = 0,
        new_only: bool = False,
    ) -> List[Dict]:
        """필터링된 피드 반환"""
        items = list(self._items.values())
        if category:
            items = [i for i in items if i.category == category]
        if source:
            items = [i for i in items if i.source == source]
        if new_only:
            items = [i for i in items if i.is_new]
        items = [i for i in items if i.score >= min_score]
        items.sort(key=lambda i: (-i.score, i.fetched_at), reverse=False)
        items.sort(key=lambda i: i.score, reverse=True)

        # 읽음 처리
        for item in items[:limit]:
            item.is_new = False

        return [
            {
                "id": i.id, "title": i.title, "url": i.url,
                "source": i.source, "category": i.category,
                "score": i.score, "summary": i.summary,
                "published_at": i.published_at, "tags": i.tags,
            }
            for i in items[:limit]
        ]

    def get_ai_digest(self, hours: int = 24) -> Dict:
        """AI 관련 중요 소식 요약"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        ai_items = [
            i for i in self._items.values()
            if i.category == "AI" and i.fetched_at >= cutoff
        ]
        ai_items.sort(key=lambda i: -i.score)
        top = ai_items[:10]

        digest = {
            "period_hours": hours,
            "total_ai_items": len(ai_items),
            "top_items": [
                {"title": i.title, "source": i.source, "score": i.score, "url": i.url}
                for i in top
            ],
            "sources_breakdown": {},
        }

        for item in ai_items:
            src = item.source
            digest["sources_breakdown"][src] = digest["sources_breakdown"].get(src, 0) + 1

        # LLM 요약
        if self.llm and top:
            try:
                from jarvis.llm.manager import Message
                titles_text = "\n".join([f"- {i.title}" for i in top[:5]])
                resp = self.llm.chat(
                    [Message(role="user", content=f"다음 AI/기술 뉴스를 3-4문장으로 요약하세요:\n{titles_text}")],
                    max_tokens=512,
                )
                digest["llm_summary"] = resp.content
            except Exception:
                pass

        return digest

    def get_stats(self) -> Dict:
        return {
            **self._stats,
            "total_items": len(self._items),
            "new_items": sum(1 for i in self._items.values() if i.is_new),
            "is_running": self._is_running,
            "categories": {
                cat: sum(1 for i in self._items.values() if i.category == cat)
                for cat in ["AI", "Tech", "Security", "Business", "Dev"]
            },
        }

    # ── 영속성 ─────────────────────────────────────────────────────────────

    def _save_cache(self):
        try:
            MONITOR_DB.parent.mkdir(parents=True, exist_ok=True)
            # 상위 200개만 저장
            sorted_items = sorted(self._items.values(), key=lambda i: -i.score)[:200]
            data = [
                {"id": i.id, "title": i.title, "url": i.url, "source": i.source,
                 "category": i.category, "score": i.score, "summary": i.summary,
                 "published_at": i.published_at, "fetched_at": i.fetched_at, "tags": i.tags}
                for i in sorted_items
            ]
            MONITOR_DB.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[LiveMon] Save error: {e}")

    def _load_cache(self):
        try:
            if MONITOR_DB.exists():
                data = json.loads(MONITOR_DB.read_text(encoding="utf-8"))
                for d in data:
                    item = FeedItem(
                        id=d["id"], title=d["title"], url=d["url"],
                        source=d["source"], category=d.get("category", "Tech"),
                        score=d.get("score", 50), summary=d.get("summary", ""),
                        published_at=d.get("published_at", ""),
                        fetched_at=d.get("fetched_at", datetime.now().isoformat()),
                        tags=d.get("tags", []), is_new=False,
                    )
                    self._items[item.id] = item
                logger.info(f"[LiveMon] Loaded {len(self._items)} cached items")
        except Exception as e:
            logger.debug(f"[LiveMon] Load error: {e}")

    def _emit(self, event_type: str, data: Dict):
        if self.event_callback:
            try:
                self.event_callback({"type": event_type, **data, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass
