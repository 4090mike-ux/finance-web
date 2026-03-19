"""
JARVIS 지식 자동 갱신 시스템
- 주기적 ArXiv 논문 수집
- GitHub 트렌드 모니터링
- 뉴스 피드 처리
- LLM 모델 업데이트 추적
- 지식 요약 및 인덱싱
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class KnowledgeUpdater:
    """
    자율 지식 갱신 시스템
    백그라운드에서 최신 정보를 지속적으로 수집
    """

    UPDATE_TOPICS = {
        "ai_research": ["large language model", "transformer architecture", "reinforcement learning from human feedback"],
        "ai_tools": ["AI agent framework", "vector database", "prompt engineering"],
        "programming": ["python best practices", "async programming", "API design"],
        "security": ["AI safety", "red teaming LLM", "adversarial robustness"],
    }

    def __init__(self, web_intel, memory_manager, llm_manager):
        self.web = web_intel
        self.memory = memory_manager
        self.llm = llm_manager
        self.is_running = False
        self.last_update = {}
        self.update_thread = None
        self.update_interval = 3600  # 1시간마다
        logger.info("KnowledgeUpdater initialized")

    def start_auto_update(self):
        """자동 갱신 시작"""
        self.is_running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="JARVIS-KnowledgeUpdater"
        )
        self.update_thread.start()
        logger.info("Auto knowledge update started")

    def stop_auto_update(self):
        """자동 갱신 중지"""
        self.is_running = False
        logger.info("Auto knowledge update stopped")

    def _update_loop(self):
        """갱신 루프"""
        while self.is_running:
            try:
                self.run_update_cycle()
            except Exception as e:
                logger.error(f"Update cycle error: {e}")
            time.sleep(self.update_interval)

    def run_update_cycle(self) -> Dict:
        """한 번의 갱신 사이클 실행"""
        results = {
            "papers_added": 0,
            "github_repos_added": 0,
            "news_added": 0,
            "timestamp": datetime.now().isoformat(),
        }

        # 최신 ArXiv 논문
        try:
            papers = self.web.search_arxiv("large language model agent 2025", max_results=5)
            for paper in papers:
                if "error" not in paper:
                    self.memory.add_knowledge(
                        category="arxiv_paper",
                        title=paper.get("title", ""),
                        content=paper.get("abstract", "")[:1000],
                        source=paper.get("url", ""),
                        metadata={
                            "authors": paper.get("authors", []),
                            "published": paper.get("published", ""),
                            "categories": paper.get("categories", []),
                        }
                    )
                    results["papers_added"] += 1
            logger.info(f"Added {results['papers_added']} papers")
        except Exception as e:
            logger.error(f"ArXiv update failed: {e}")

        # GitHub 트렌딩
        try:
            repos = self.web.get_github_trending(language="python")
            for repo in repos[:5]:
                if "error" not in repo:
                    self.memory.add_knowledge(
                        category="github_trending",
                        title=repo.get("name", ""),
                        content=f"{repo.get('description', '')} | Stars: {repo.get('stars', 0)}",
                        source=repo.get("url", ""),
                        metadata={"language": repo.get("language", ""), "stars": repo.get("stars", 0)}
                    )
                    results["github_repos_added"] += 1
            logger.info(f"Added {results['github_repos_added']} GitHub repos")
        except Exception as e:
            logger.error(f"GitHub update failed: {e}")

        # AI 뉴스
        try:
            news = self.web.search_news("artificial intelligence breakthrough 2025", max_results=5)
            for item in news:
                if "error" not in item and item.get("title"):
                    self.memory.add_knowledge(
                        category="ai_news",
                        title=item.get("title", ""),
                        content=item.get("snippet", "")[:500],
                        source=item.get("url", ""),
                        metadata={"date": item.get("date", ""), "source": item.get("source", "")}
                    )
                    results["news_added"] += 1
            logger.info(f"Added {results['news_added']} news items")
        except Exception as e:
            logger.error(f"News update failed: {e}")

        self.last_update = results
        return results

    def get_knowledge_summary(self) -> str:
        """수집된 지식 요약"""
        if not self.llm.primary:
            return "API key 없음 - 요약 불가"

        try:
            # 최신 지식 수집
            papers = self.memory.get_knowledge(category="arxiv_paper", limit=5)
            repos = self.memory.get_knowledge(category="github_trending", limit=5)
            news = self.memory.get_knowledge(category="ai_news", limit=5)

            if not any([papers, repos, news]):
                return "아직 수집된 지식이 없습니다. 갱신을 실행해주세요."

            # 요약 생성
            knowledge_text = ""
            if papers:
                knowledge_text += "최신 논문:\n"
                for p in papers:
                    knowledge_text += f"- {p['title']}: {p['content'][:200]}\n"

            if repos:
                knowledge_text += "\nGitHub 트렌드:\n"
                for r in repos:
                    knowledge_text += f"- {r['title']}: {r['content'][:100]}\n"

            if news:
                knowledge_text += "\nAI 뉴스:\n"
                for n in news:
                    knowledge_text += f"- {n['title']}: {n['content'][:150]}\n"

            from jarvis.llm.manager import Message
            messages = [
                Message(
                    role="user",
                    content=f"다음 최신 AI 정보를 간결하게 요약해주세요:\n\n{knowledge_text[:3000]}"
                )
            ]
            response = self.llm.chat(messages, max_tokens=1024)
            return response.content

        except Exception as e:
            return f"요약 생성 오류: {e}"

    def get_status(self) -> Dict:
        """갱신 상태"""
        stats = self.memory.get_stats()
        return {
            "is_running": self.is_running,
            "last_update": self.last_update,
            "update_interval_hours": self.update_interval / 3600,
            "total_knowledge": stats.get("knowledge_base", 0),
            "categories": self._get_categories(),
        }

    def _get_categories(self) -> Dict:
        """카테고리별 지식 수"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.memory.db_path)
            rows = conn.execute(
                "SELECT category, COUNT(*) FROM knowledge_base GROUP BY category"
            ).fetchall()
            conn.close()
            return {r[0]: r[1] for r in rows}
        except:
            return {}
