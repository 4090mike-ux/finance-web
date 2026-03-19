"""
JARVIS 딥 리서치 엔진 — Iteration 3
멀티홉 검색으로 한 주제를 깊고 넓게 탐구하는 연구 에이전트
- 멀티홉 쿼리 (첫 검색 결과로 다음 검색 생성)
- 웹 + ArXiv + GitHub + Wikipedia 통합
- 전체 웹 페이지 스크래핑 (스니펫 넘어서)
- 출처 추적 및 사실 검증
- 구조화된 연구 보고서 생성
- 지식 갭 자동 발견
"""

import re
import json
import time
import logging
import requests
from typing import Any, Dict, Generator, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ResearchSource:
    url: str
    title: str
    content: str
    source_type: str  # web, arxiv, github, wikipedia
    relevance: float = 0.0
    verified: bool = False
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ResearchReport:
    topic: str
    executive_summary: str
    key_findings: List[str]
    sources: List[ResearchSource]
    knowledge_gaps: List[str]
    follow_up_questions: List[str]
    confidence: float
    depth: int  # 연구 깊이 (홉 수)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DeepResearcher:
    """
    JARVIS 딥 리서치 엔진
    인간 연구자를 뛰어넘는 속도와 깊이로 정보를 수집/분석
    """

    MAX_HOPS = 4          # 최대 멀티홉 깊이
    MAX_SOURCES = 20      # 최대 소스 수집
    SCRAPE_TIMEOUT = 10   # 페이지 스크래핑 타임아웃

    def __init__(self, web_intelligence, llm_manager, memory_manager=None):
        self.web = web_intelligence
        self.llm = llm_manager
        self.memory = memory_manager
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "JARVIS-Research/3.0 Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        logger.info("DeepResearcher initialized")

    # ── 메인 리서치 ───────────────────────────────────────────────────────
    def research(
        self,
        topic: str,
        depth: int = 2,
        include_papers: bool = True,
        include_code: bool = True,
        language: str = "ko",
    ) -> ResearchReport:
        """
        주제에 대한 심층 연구 실행
        depth: 멀티홉 깊이 (1=기본, 4=최대)
        """
        logger.info(f"Deep research started: '{topic}' (depth={depth})")
        start_time = time.time()
        sources: List[ResearchSource] = []
        all_queries = {topic}
        current_queries = [topic]

        for hop in range(min(depth, self.MAX_HOPS)):
            if len(sources) >= self.MAX_SOURCES:
                break

            hop_sources = []

            # 웹 검색
            for query in current_queries[:3]:
                results = self.web.search_web(query, max_results=5)
                for r in results:
                    if not r.get("error") and r.get("url"):
                        src = ResearchSource(
                            url=r.get("url", ""),
                            title=r.get("title", ""),
                            content=r.get("snippet", ""),
                            source_type="web",
                        )
                        # 전체 페이지 스크래핑 (홉 0에서만 — 속도 고려)
                        if hop == 0 and len(hop_sources) < 3:
                            full_content = self._scrape_page(r.get("url", ""))
                            if full_content:
                                src.content = full_content[:3000]
                        hop_sources.append(src)

            # ArXiv 논문 검색
            if include_papers and hop < 2:
                papers = self.web.search_arxiv(topic, max_results=3)
                for p in papers:
                    if not p.get("error"):
                        hop_sources.append(ResearchSource(
                            url=p.get("url", ""),
                            title=p.get("title", ""),
                            content=f"저자: {', '.join(p.get('authors', [])[:3])}\n{p.get('abstract', '')}",
                            source_type="arxiv",
                        ))

            # GitHub 검색
            if include_code and hop < 2:
                repos = self.web.search_github(topic, max_results=3)
                for r in repos:
                    if not r.get("error"):
                        hop_sources.append(ResearchSource(
                            url=r.get("url", ""),
                            title=r.get("name", ""),
                            content=f"⭐{r.get('stars', 0)} | {r.get('description', '')} | 언어: {r.get('language', '')}",
                            source_type="github",
                        ))

            # Wikipedia
            if hop == 0:
                wiki = self.web.search_wikipedia(topic)
                if wiki and not wiki.get("error"):
                    hop_sources.append(ResearchSource(
                        url=wiki.get("url", ""),
                        title=wiki.get("title", topic),
                        content=wiki.get("summary", "")[:2000],
                        source_type="wikipedia",
                    ))

            sources.extend(hop_sources[:10])

            # 다음 홉 쿼리 생성 (LLM 기반)
            if hop < depth - 1 and self.llm and hop_sources:
                next_queries = self._generate_next_queries(topic, hop_sources, all_queries)
                for q in next_queries:
                    all_queries.add(q)
                current_queries = next_queries

        # 보고서 생성
        report = self._generate_report(topic, sources, depth, language)
        duration = round(time.time() - start_time, 1)
        logger.info(f"Research complete: {len(sources)} sources in {duration}s")

        # 기억에 저장
        if self.memory:
            self.memory.add_knowledge(
                "research_report",
                f"주제: {topic}\n요약: {report.executive_summary[:500]}",
                topic,
            )

        return report

    def research_streaming(self, topic: str, depth: int = 2) -> Generator[Dict, None, None]:
        """스트리밍 연구 (실시간 진행 상황 반환)"""
        yield {"type": "start", "topic": topic, "depth": depth}

        sources = []
        current_queries = [topic]

        for hop in range(min(depth, self.MAX_HOPS)):
            yield {"type": "hop_start", "hop": hop + 1, "queries": current_queries}

            hop_sources = []
            for query in current_queries[:2]:
                results = self.web.search_web(query, max_results=4)
                for r in results:
                    if not r.get("error"):
                        src = ResearchSource(
                            url=r.get("url", ""),
                            title=r.get("title", ""),
                            content=r.get("snippet", ""),
                            source_type="web",
                        )
                        hop_sources.append(src)
                        yield {"type": "source_found", "title": r.get("title", ""), "url": r.get("url", "")}

            sources.extend(hop_sources)
            yield {"type": "hop_done", "hop": hop + 1, "sources_count": len(sources)}

            if hop < depth - 1 and self.llm:
                next_queries = self._generate_next_queries(topic, hop_sources, {topic})
                current_queries = next_queries
                yield {"type": "next_queries", "queries": next_queries}

        yield {"type": "generating_report"}
        report = self._generate_report(topic, sources, depth, "ko")
        yield {
            "type": "done",
            "report": {
                "topic": report.topic,
                "summary": report.executive_summary,
                "findings": report.key_findings,
                "sources_count": len(report.sources),
                "knowledge_gaps": report.knowledge_gaps,
                "follow_up": report.follow_up_questions,
                "confidence": report.confidence,
            }
        }

    # ── 페이지 스크래핑 ───────────────────────────────────────────────────
    def _scrape_page(self, url: str) -> Optional[str]:
        """웹 페이지 전체 텍스트 추출"""
        if not url or not url.startswith("http"):
            return None
        try:
            resp = self._session.get(url, timeout=self.SCRAPE_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()

            # BeautifulSoup으로 텍스트 추출
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.content, "lxml")
                # 불필요한 태그 제거
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "ads"]):
                    tag.decompose()
                # 주요 컨텐츠 추출
                for selector in ["article", "main", ".content", "#content", ".post-body"]:
                    main = soup.select_one(selector)
                    if main:
                        return re.sub(r'\n{3,}', '\n\n', main.get_text(separator="\n")).strip()
                return re.sub(r'\n{3,}', '\n\n', soup.get_text(separator="\n")).strip()[:5000]
            except ImportError:
                # BS4 없으면 기본 정규식
                text = re.sub(r'<[^>]+>', ' ', resp.text)
                return re.sub(r'\s+', ' ', text).strip()[:3000]

        except Exception as e:
            logger.debug(f"Scrape failed for {url}: {e}")
            return None

    # ── 멀티홉 쿼리 생성 ─────────────────────────────────────────────────
    def _generate_next_queries(
        self,
        topic: str,
        sources: List[ResearchSource],
        seen_queries: set,
    ) -> List[str]:
        """현재 소스에서 다음 탐색 쿼리 생성"""
        if not self.llm:
            return []

        content_sample = "\n".join([s.content[:200] for s in sources[:5]])
        from jarvis.llm.manager import Message

        prompt = f"""주제 "{topic}"에 대한 다음 정보를 바탕으로 더 깊게 탐구할 3개의 검색 쿼리를 생성하세요.

현재 수집된 정보 요약:
{content_sample}

이미 탐색한 쿼리: {', '.join(list(seen_queries)[:5])}

규칙:
- 새로운 각도나 세부 사항을 탐색하는 쿼리
- 이미 탐색한 것과 다른 쿼리
- 구체적이고 검색 가능한 쿼리
- 영어로 작성 (국제 검색 최적화)

JSON 배열로만 반환: ["query1", "query2", "query3"]"""

        messages = [Message(role="user", content=prompt)]
        try:
            response = self.llm.chat(messages, system="JSON 배열만 반환하세요.", max_tokens=256)
            match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return [q for q in queries if q not in seen_queries][:3]
        except Exception as e:
            logger.debug(f"Query generation failed: {e}")
        return []

    # ── 보고서 생성 ───────────────────────────────────────────────────────
    def _generate_report(
        self,
        topic: str,
        sources: List[ResearchSource],
        depth: int,
        language: str,
    ) -> ResearchReport:
        """수집된 소스에서 구조화된 보고서 생성"""
        if not self.llm or not sources:
            return ResearchReport(
                topic=topic,
                executive_summary=f"{len(sources)}개 소스 수집됨. LLM 없이 보고서 생성 불가.",
                key_findings=[s.content[:200] for s in sources[:3]],
                sources=sources,
                knowledge_gaps=[],
                follow_up_questions=[],
                confidence=0.3,
                depth=depth,
            )

        # 소스 내용 결합
        source_texts = []
        for i, src in enumerate(sources[:15]):
            source_texts.append(
                f"[소스 {i+1}: {src.source_type.upper()} - {src.title}]\n{src.content[:600]}"
            )
        combined = "\n\n".join(source_texts)

        from jarvis.llm.manager import Message
        prompt = f"""다음 소스들을 바탕으로 "{topic}"에 대한 심층 연구 보고서를 작성하세요.

소스 데이터 ({len(sources)}개):
{combined[:8000]}

다음 형식의 JSON으로 반환하세요:
{{
  "executive_summary": "핵심 요약 (300자 이내)",
  "key_findings": ["발견 1", "발견 2", "발견 3", "발견 4", "발견 5"],
  "knowledge_gaps": ["아직 불분명한 점 1", "불분명한 점 2"],
  "follow_up_questions": ["후속 탐구 질문 1", "질문 2", "질문 3"],
  "confidence": 0.85
}}

언어: {"한국어" if language == "ko" else "English"}"""

        messages = [Message(role="user", content=prompt)]
        try:
            response = self.llm.chat(messages, system="당신은 전문 연구 분석가입니다. JSON만 출력하세요.", max_tokens=3000)
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return ResearchReport(
                    topic=topic,
                    executive_summary=data.get("executive_summary", ""),
                    key_findings=data.get("key_findings", []),
                    sources=sources,
                    knowledge_gaps=data.get("knowledge_gaps", []),
                    follow_up_questions=data.get("follow_up_questions", []),
                    confidence=float(data.get("confidence", 0.7)),
                    depth=depth,
                )
        except Exception as e:
            logger.error(f"Report generation error: {e}")

        # 폴백 보고서
        return ResearchReport(
            topic=topic,
            executive_summary="\n".join([s.content[:150] for s in sources[:3]]),
            key_findings=[s.title for s in sources[:5]],
            sources=sources,
            knowledge_gaps=["LLM 보고서 생성 실패"],
            follow_up_questions=[],
            confidence=0.5,
            depth=depth,
        )

    def format_report_markdown(self, report: ResearchReport) -> str:
        """보고서를 마크다운으로 포맷"""
        lines = [
            f"# 연구 보고서: {report.topic}",
            f"*{report.created_at[:19]} | 깊이: {report.depth}홉 | 소스: {len(report.sources)}개 | 신뢰도: {report.confidence:.0%}*",
            "",
            "## 핵심 요약",
            report.executive_summary,
            "",
            "## 주요 발견",
        ]
        for i, f in enumerate(report.key_findings, 1):
            lines.append(f"{i}. {f}")

        if report.knowledge_gaps:
            lines.extend(["", "## 지식 갭 (추가 탐구 필요)"])
            for g in report.knowledge_gaps:
                lines.append(f"- {g}")

        if report.follow_up_questions:
            lines.extend(["", "## 후속 탐구 질문"])
            for q in report.follow_up_questions:
                lines.append(f"- {q}")

        lines.extend(["", "## 참조 소스"])
        seen_urls = set()
        for src in report.sources[:10]:
            if src.url and src.url not in seen_urls:
                lines.append(f"- [{src.title or src.url}]({src.url}) *[{src.source_type}]*")
                seen_urls.add(src.url)

        return "\n".join(lines)
