"""
JARVIS 웹 인텔리전스 모듈
- DuckDuckGo 검색 (API 없이)
- GitHub API (최신 트렌드, 논문 코드, 레포 검색)
- ArXiv API (최신 AI/ML 논문)
- Wikipedia API
- 뉴스 수집
"""

import re
import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WebIntelligence:
    """JARVIS 웹 인텔리전스 시스템"""

    def __init__(self, github_token: str = ""):
        self.github_token = github_token
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "JARVIS-AI/1.0 (Intelligent Assistant)",
            "Accept": "application/json",
        })

        if github_token:
            self.session.headers["Authorization"] = f"token {github_token}"
            logger.info("GitHub API token configured")

    # ==================== 웹 검색 ====================

    def _get_ddgs(self):
        """DDGS 클라이언트 - 최신 패키지명 자동 감지"""
        try:
            from ddgs import DDGS
            return DDGS()
        except ImportError:
            from duckduckgo_search import DDGS
            return DDGS()

    def search_web(self, query: str, max_results: int = 10) -> List[Dict]:
        """DuckDuckGo 웹 검색"""
        try:
            results = []
            with self._get_ddgs() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("url", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                        "source": "duckduckgo",
                    })
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return [{"error": str(e), "query": query}]

    def search_news(self, query: str, max_results: int = 10) -> List[Dict]:
        """최신 뉴스 검색"""
        try:
            results = []
            with self._get_ddgs() as ddgs:
                for r in ddgs.news(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", r.get("href", "")),
                        "snippet": r.get("body", r.get("excerpt", "")),
                        "date": r.get("date", ""),
                        "source": r.get("source", ""),
                    })
            return results
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return [{"error": str(e)}]

    # ==================== GitHub ====================

    def search_github(self, query: str, type: str = "repositories", max_results: int = 10) -> List[Dict]:
        """GitHub 검색 (레포, 코드, 이슈 등)"""
        try:
            url = f"https://api.github.com/search/{type}"
            params = {"q": query, "per_page": min(max_results, 30), "sort": "stars", "order": "desc"}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            results = []

            if type == "repositories":
                for item in items:
                    results.append({
                        "name": item["full_name"],
                        "url": item["html_url"],
                        "description": item.get("description", ""),
                        "stars": item["stargazers_count"],
                        "forks": item["forks_count"],
                        "language": item.get("language", ""),
                        "updated": item.get("updated_at", ""),
                        "topics": item.get("topics", []),
                    })
            elif type == "code":
                for item in items:
                    results.append({
                        "name": item["name"],
                        "path": item["path"],
                        "url": item["html_url"],
                        "repo": item["repository"]["full_name"],
                    })

            return results
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return [{"error": str(e)}]

    def get_github_trending(self, language: str = "", since: str = "daily") -> List[Dict]:
        """GitHub 트렌딩 레포 (비공식 API)"""
        try:
            url = "https://api.github.com/search/repositories"
            # 최근 1주일 내 스타 많이 받은 레포
            since_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            q = f"created:>{since_date} stars:>10"
            if language:
                q += f" language:{language}"

            params = {"q": q, "sort": "stars", "order": "desc", "per_page": 20}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "name": item["full_name"],
                    "url": item["html_url"],
                    "description": item.get("description", ""),
                    "stars": item["stargazers_count"],
                    "forks": item["forks_count"],
                    "language": item.get("language", ""),
                    "created": item.get("created_at", ""),
                })
            return results
        except Exception as e:
            logger.error(f"GitHub trending failed: {e}")
            return [{"error": str(e)}]

    def get_repo_info(self, repo: str) -> Dict:
        """특정 레포 정보"""
        try:
            url = f"https://api.github.com/repos/{repo}"
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            return {
                "name": data["full_name"],
                "url": data["html_url"],
                "description": data.get("description", ""),
                "stars": data["stargazers_count"],
                "forks": data["forks_count"],
                "watchers": data["watchers_count"],
                "language": data.get("language", ""),
                "topics": data.get("topics", []),
                "license": data.get("license", {}).get("name", "") if data.get("license") else "",
                "created": data.get("created_at", ""),
                "updated": data.get("updated_at", ""),
                "open_issues": data["open_issues_count"],
                "default_branch": data["default_branch"],
            }
        except Exception as e:
            return {"error": str(e), "repo": repo}

    def get_repo_readme(self, repo: str) -> Dict:
        """레포 README 내용"""
        try:
            url = f"https://api.github.com/repos/{repo}/readme"
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            import base64
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return {"repo": repo, "content": content[:3000], "url": data["html_url"]}
        except Exception as e:
            return {"error": str(e), "repo": repo}

    # ==================== ArXiv 논문 ====================

    def search_arxiv(self, query: str, max_results: int = 10, category: str = "") -> List[Dict]:
        """ArXiv 논문 검색"""
        try:
            import arxiv
            search_query = query
            if category:
                search_query = f"cat:{category} AND ({query})"

            client = arxiv.Client()
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors[:5]],
                    "abstract": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": str(paper.published.date()) if paper.published else "",
                    "categories": paper.categories,
                    "arxiv_id": paper.get_short_id(),
                })
            return results
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return [{"error": str(e)}]

    def get_latest_ai_papers(self, limit: int = 10) -> List[Dict]:
        """최신 AI/ML 논문 조회"""
        categories = ["cs.AI", "cs.LG", "cs.CL", "stat.ML"]
        query = " OR ".join([f"cat:{c}" for c in categories])
        return self.search_arxiv("artificial intelligence machine learning", max_results=limit)

    # ==================== Wikipedia ====================

    def search_wikipedia(self, query: str, lang: str = "ko") -> Dict:
        """Wikipedia 검색"""
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia(
                language=lang,
                user_agent="JARVIS-AI/1.0"
            )
            page = wiki.page(query)

            if page.exists():
                return {
                    "title": page.title,
                    "summary": page.summary[:2000],
                    "url": page.fullurl,
                    "language": lang,
                    "sections": [s.title for s in page.sections[:10]],
                }
            else:
                # 영어로 폴백
                if lang != "en":
                    return self.search_wikipedia(query, lang="en")
                return {"error": f"Wikipedia page not found: {query}"}
        except Exception as e:
            return {"error": str(e)}

    # ==================== 통합 검색 ====================

    def comprehensive_search(self, query: str) -> Dict:
        """웹 + 뉴스 + GitHub + Wikipedia 통합 검색"""
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "web": [],
            "news": [],
            "github": [],
            "wikipedia": {},
        }

        # 웹 검색
        try:
            results["web"] = self.search_web(query, max_results=5)
        except Exception as e:
            results["web"] = [{"error": str(e)}]

        # 뉴스
        try:
            results["news"] = self.search_news(query, max_results=5)
        except Exception as e:
            results["news"] = [{"error": str(e)}]

        # GitHub (기술 쿼리일 때)
        tech_keywords = ["python", "javascript", "ai", "ml", "api", "docker", "code", "framework", "library"]
        if any(kw in query.lower() for kw in tech_keywords):
            try:
                results["github"] = self.search_github(query, max_results=5)
            except Exception as e:
                results["github"] = [{"error": str(e)}]

        # Wikipedia
        try:
            results["wikipedia"] = self.search_wikipedia(query)
        except Exception as e:
            results["wikipedia"] = {"error": str(e)}

        return results

    # ==================== LLM/AI 트렌드 ====================

    def get_ai_trends(self) -> Dict:
        """현재 AI 트렌드 정보 수집"""
        return {
            "latest_models": self._get_latest_llm_models(),
            "trending_repos": self.get_github_trending(language="python"),
            "recent_papers": self.get_latest_ai_papers(limit=5),
            "news": self.search_news("artificial intelligence LLM 2025", max_results=5),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_latest_llm_models(self) -> List[Dict]:
        """알려진 최신 LLM 모델 목록"""
        # 하드코딩된 최신 모델 정보 (2025 기준)
        return [
            {
                "name": "Claude Sonnet 4.6 / Opus 4.6",
                "provider": "Anthropic",
                "context": "200K tokens",
                "multimodal": True,
            },
            {
                "name": "GPT-4o / o1 / o3",
                "provider": "OpenAI",
                "context": "128K tokens",
                "multimodal": True,
            },
            {
                "name": "Gemini 2.0 Pro",
                "provider": "Google",
                "context": "1M tokens",
                "multimodal": True,
            },
            {
                "name": "Llama 3.3 70B",
                "provider": "Meta (Open Source)",
                "context": "128K tokens",
                "multimodal": False,
            },
            {
                "name": "DeepSeek V3 / R1",
                "provider": "DeepSeek",
                "context": "64K tokens",
                "multimodal": False,
            },
            {
                "name": "Mistral Large 3",
                "provider": "Mistral AI",
                "context": "128K tokens",
                "multimodal": False,
            },
        ]
