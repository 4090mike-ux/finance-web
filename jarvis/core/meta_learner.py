"""
JARVIS 메타 학습 엔진 — Iteration 5
모든 상호작용에서 학습하여 자신의 추론 전략·프롬프트·도구 선택을 자동 개선
- 쿼리 유형별 최적 전략 학습 (Bandit 알고리즘)
- 시스템 프롬프트 자동 진화
- 도구 선택 최적화
- 응답 품질 자기 평가
- 실패 패턴 자동 진단 및 수정
"""

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/jarvis/meta_state.json")


@dataclass
class StrategyRecord:
    name: str
    successes: int = 0
    failures: int = 0
    total_duration: float = 0.0
    avg_rating: float = 3.0   # 1–5
    last_used: str = ""

    @property
    def win_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.5

    @property
    def ucb_score(self, c: float = 1.4) -> float:
        total = self.successes + self.failures
        if total == 0:
            return float("inf")
        exploitation = self.win_rate
        exploration = c * math.sqrt(math.log(max(total, 1)) / total)
        return exploitation + exploration


@dataclass
class QueryPattern:
    pattern_id: str
    keywords: List[str]
    best_strategy: str
    best_tools: List[str]
    sample_queries: List[str] = field(default_factory=list)
    success_count: int = 0


@dataclass
class LearningEvent:
    query: str
    strategy: str
    tools_used: List[str]
    success: bool
    rating: float   # 1–5
    duration: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetaLearner:
    """
    JARVIS 메타 학습 시스템
    매 상호작용에서 자신을 개선하는 자기 진화 엔진
    """

    QUERY_CATEGORIES = {
        "factual": ["뭐야", "알려줘", "what", "explain", "정의", "개념"],
        "analytical": ["분석", "비교", "평가", "장단점", "pros", "cons", "analyze"],
        "creative": ["만들어", "작성", "write", "create", "설계", "design"],
        "research": ["논문", "연구", "arxiv", "최신", "트렌드", "paper"],
        "coding": ["코드", "함수", "class", "bug", "debug", "python", "구현"],
        "planning": ["계획", "단계", "how to", "절차", "방법", "plan"],
        "mathematical": ["계산", "수식", "증명", "compute", "math", "통계"],
        "conversational": ["안녕", "어때", "how are", "뭐해", "재미"],
    }

    STRATEGIES = ["react", "cot", "tot", "self_reflection", "socratic", "direct"]

    TOOL_AFFINITY = {
        "factual": ["web_search", "wikipedia"],
        "analytical": ["reason_deeply", "web_search"],
        "creative": [],
        "research": ["arxiv_search", "web_search", "github_search"],
        "coding": ["execute_python", "web_search"],
        "planning": ["reason_deeply"],
        "mathematical": ["execute_python", "reason_deeply"],
        "conversational": [],
    }

    def __init__(self, llm_manager=None):
        self.llm = llm_manager
        self._strategy_records: Dict[str, StrategyRecord] = {
            s: StrategyRecord(name=s) for s in self.STRATEGIES
        }
        self._category_strategy: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: 1.0 for s in self.STRATEGIES}
        )
        self._query_patterns: List[QueryPattern] = []
        self._events: List[LearningEvent] = []
        self._evolved_prompts: Dict[str, str] = {}
        self._total_interactions: int = 0
        self._load_state()
        logger.info(f"MetaLearner initialized — {self._total_interactions} interactions learned")

    # ── 전략 선택 (UCB1 알고리즘) ────────────────────────────────────────────
    def select_strategy(self, query: str) -> str:
        """쿼리에 최적인 추론 전략 선택"""
        category = self._classify_query(query)
        weights = self._category_strategy.get(category, {s: 1.0 for s in self.STRATEGIES})

        # 탐색/활용 균형 (epsilon-greedy + UCB)
        if random.random() < 0.05 and self._total_interactions > 20:
            # 5% 무작위 탐색
            return random.choice(self.STRATEGIES)

        best = max(weights.items(), key=lambda x: x[1])
        logger.debug(f"[MetaLearner] category={category} → strategy={best[0]} (weight={best[1]:.2f})")
        return best[0]

    def select_tools(self, query: str) -> List[str]:
        """쿼리에 최적인 도구 목록 추천"""
        category = self._classify_query(query)
        return self.TOOL_AFFINITY.get(category, [])

    def _classify_query(self, query: str) -> str:
        q = query.lower()
        scores = {cat: 0 for cat in self.QUERY_CATEGORIES}
        for cat, keywords in self.QUERY_CATEGORIES.items():
            for kw in keywords:
                if kw in q:
                    scores[cat] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "factual"

    # ── 학습 ─────────────────────────────────────────────────────────────────
    def record_outcome(
        self,
        query: str,
        strategy: str,
        tools_used: List[str],
        success: bool,
        rating: float = 3.0,
        duration: float = 0.0,
    ):
        """결과 기록 및 가중치 업데이트"""
        category = self._classify_query(query)

        # 전략 기록 업데이트
        rec = self._strategy_records.get(strategy)
        if rec:
            if success:
                rec.successes += 1
            else:
                rec.failures += 1
            rec.total_duration += duration
            rec.avg_rating = (rec.avg_rating * 0.9 + rating * 0.1)  # EWMA
            rec.last_used = datetime.now().isoformat()

        # 카테고리별 전략 가중치 업데이트 (강화학습 스타일)
        reward = (rating / 5.0) * (1.5 if success else 0.5)
        w = self._category_strategy[category]
        lr = 0.1  # 학습률
        w[strategy] = w.get(strategy, 1.0) * (1 - lr) + reward * lr
        # Softmax 정규화
        total = sum(math.exp(v) for v in w.values())
        self._category_strategy[category] = {
            k: math.exp(v) / total for k, v in w.items()
        }

        # 이벤트 기록
        event = LearningEvent(
            query=query[:200],
            strategy=strategy,
            tools_used=tools_used,
            success=success,
            rating=rating,
            duration=duration,
        )
        self._events.append(event)
        if len(self._events) > 500:
            self._events = self._events[-500:]

        self._total_interactions += 1

        # 100번마다 자동 분석
        if self._total_interactions % 100 == 0:
            self._auto_analyze()

        self._save_state()

    # ── 시스템 프롬프트 진화 ─────────────────────────────────────────────────
    def evolve_system_prompt(self, base_prompt: str, performance_data: Dict) -> str:
        """성능 데이터 기반 시스템 프롬프트 자동 개선"""
        if not self.llm or self._total_interactions < 20:
            return base_prompt

        from jarvis.llm.manager import Message

        weak_areas = self._find_weak_areas()
        if not weak_areas:
            return self._evolved_prompts.get("main", base_prompt)

        prompt = f"""현재 JARVIS 시스템 프롬프트:
{base_prompt[:2000]}

성능 분석:
- 약점 영역: {', '.join(weak_areas)}
- 최고 전략: {self._get_best_strategy()}
- 평균 평점: {self._get_avg_rating():.1f}/5.0
- 총 상호작용: {self._total_interactions}

약점 영역을 개선하는 더 나은 시스템 프롬프트를 작성해주세요.
핵심 정체성(JARVIS)은 유지하되, 약점을 보완하는 지침을 추가하세요.
결과: 개선된 시스템 프롬프트 전체 텍스트만 반환"""

        try:
            messages = [Message(role="user", content=prompt)]
            resp = self.llm.chat(messages, max_tokens=2048)
            evolved = resp.content.strip()
            if len(evolved) > 200:  # 유효한 응답인 경우만
                self._evolved_prompts["main"] = evolved
                logger.info(f"[MetaLearner] System prompt evolved ({len(evolved)} chars)")
                return evolved
        except Exception as e:
            logger.debug(f"Prompt evolution error: {e}")

        return base_prompt

    def get_evolved_prompt(self, base_prompt: str) -> str:
        """진화된 프롬프트 반환 (없으면 기본값)"""
        return self._evolved_prompts.get("main", base_prompt)

    # ── 자동 분석 ────────────────────────────────────────────────────────────
    def _auto_analyze(self):
        """자동 성능 분석 및 최적화"""
        weak = self._find_weak_areas()
        best_strat = self._get_best_strategy()
        logger.info(
            f"[MetaLearner] Auto-analysis: {self._total_interactions} interactions, "
            f"best_strategy={best_strat}, weak_areas={weak}"
        )

    def _find_weak_areas(self) -> List[str]:
        """성능이 낮은 영역 탐지"""
        if len(self._events) < 10:
            return []
        recent = self._events[-50:]
        by_category: Dict[str, List[float]] = defaultdict(list)
        for e in recent:
            cat = self._classify_query(e.query)
            by_category[cat].append(e.rating)
        weak = []
        for cat, ratings in by_category.items():
            if ratings and sum(ratings) / len(ratings) < 3.0:
                weak.append(cat)
        return weak

    def _get_best_strategy(self) -> str:
        if not self._strategy_records:
            return "direct"
        return max(
            self._strategy_records.values(),
            key=lambda r: r.win_rate,
        ).name

    def _get_avg_rating(self) -> float:
        if not self._events:
            return 3.0
        recent = self._events[-50:]
        return sum(e.rating for e in recent) / len(recent)

    # ── 자가 평가 ────────────────────────────────────────────────────────────
    def self_evaluate(self, query: str, response: str) -> float:
        """LLM 기반 응답 품질 자가 평가 (1-5점)"""
        if not self.llm:
            return 3.0
        from jarvis.llm.manager import Message

        prompt = f"""다음 질문에 대한 답변의 품질을 1-5점으로 평가하세요.

질문: {query[:300]}

답변: {response[:500]}

평가 기준: 정확성, 완전성, 명확성, 도움 정도
숫자만 반환 (1, 2, 3, 4, 5 중 하나)"""

        try:
            messages = [Message(role="user", content=prompt)]
            resp = self.llm.chat(messages, system="엄격한 품질 평가자입니다. 숫자만 출력.", max_tokens=10)
            score = float(resp.content.strip().split()[0])
            return max(1.0, min(5.0, score))
        except Exception:
            return 3.0

    # ── 통계 ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        strategy_stats = {
            name: {
                "win_rate": round(rec.win_rate, 3),
                "successes": rec.successes,
                "failures": rec.failures,
                "avg_rating": round(rec.avg_rating, 2),
            }
            for name, rec in self._strategy_records.items()
        }
        return {
            "total_interactions": self._total_interactions,
            "best_strategy": self._get_best_strategy(),
            "avg_rating": round(self._get_avg_rating(), 2),
            "weak_areas": self._find_weak_areas(),
            "strategies": strategy_stats,
            "prompt_evolved": "main" in self._evolved_prompts,
        }

    def get_learning_report(self) -> str:
        stats = self.get_stats()
        lines = [
            "## JARVIS 메타 학습 리포트",
            f"총 학습 상호작용: {stats['total_interactions']}",
            f"최적 전략: {stats['best_strategy']}",
            f"평균 응답 품질: {stats['avg_rating']}/5.0",
            f"프롬프트 진화: {'완료' if stats['prompt_evolved'] else '미완료'}",
            "",
            "### 전략별 성능",
        ]
        for name, s in stats["strategies"].items():
            lines.append(f"- **{name}**: 승률 {s['win_rate']:.0%}, 평점 {s['avg_rating']}, ({s['successes']}승/{s['failures']}패)")
        if stats["weak_areas"]:
            lines.append(f"\n### 개선 필요 영역\n{', '.join(stats['weak_areas'])}")
        return "\n".join(lines)

    def get_status(self) -> Dict:
        """API 호환 상태 반환 (get_stats 별칭 + 추가 정보)"""
        stats = self.get_stats()
        stats["recent_events"] = [
            {"strategy": e.strategy, "success": e.success, "rating": e.rating, "timestamp": e.timestamp}
            for e in self._events[-10:]
        ]
        return stats

    def optimize_strategies(self) -> Dict:
        """전략 가중치 재최적화 실행"""
        weak = self._find_weak_areas()
        best = self._get_best_strategy()
        avg = self._get_avg_rating()
        # 약한 영역의 가중치를 최고 전략으로 강화
        for cat in weak:
            w = self._category_strategy[cat]
            if best in w:
                w[best] = min(w[best] * 1.5, 5.0)
        self._save_state()
        return {
            "optimized": True,
            "weak_areas_treated": weak,
            "best_strategy_boosted": best,
            "current_avg_rating": round(avg, 2),
            "total_interactions": self._total_interactions,
        }

    # ── 영속성 ───────────────────────────────────────────────────────────────
    def _save_state(self):
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_interactions": self._total_interactions,
                "strategy_records": {
                    k: {"successes": v.successes, "failures": v.failures, "avg_rating": v.avg_rating}
                    for k, v in self._strategy_records.items()
                },
                "category_strategy": {k: v for k, v in self._category_strategy.items()},
                "evolved_prompts": self._evolved_prompts,
                "events": [
                    {"query": e.query, "strategy": e.strategy, "success": e.success,
                     "rating": e.rating, "duration": e.duration, "timestamp": e.timestamp}
                    for e in self._events[-100:]
                ],
            }
            STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"MetaLearner save error: {e}")

    def _load_state(self):
        try:
            if not STATE_PATH.exists():
                return
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            self._total_interactions = data.get("total_interactions", 0)
            for k, v in data.get("strategy_records", {}).items():
                if k in self._strategy_records:
                    self._strategy_records[k].successes = v.get("successes", 0)
                    self._strategy_records[k].failures = v.get("failures", 0)
                    self._strategy_records[k].avg_rating = v.get("avg_rating", 3.0)
            for k, v in data.get("category_strategy", {}).items():
                self._category_strategy[k] = v
            self._evolved_prompts = data.get("evolved_prompts", {})
            for e in data.get("events", []):
                self._events.append(LearningEvent(
                    query=e.get("query", ""), strategy=e.get("strategy", ""),
                    tools_used=[], success=e.get("success", True),
                    rating=e.get("rating", 3.0), duration=e.get("duration", 0.0),
                    timestamp=e.get("timestamp", ""),
                ))
        except Exception as e:
            logger.debug(f"MetaLearner load error: {e}")
