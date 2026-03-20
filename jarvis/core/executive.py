"""
JARVIS 총사령관 (Executive Controller) — Iteration 6
모든 AI 시스템을 지휘하는 최상위 지능 계층

역할:
- 임의의 입력에 대해 최적의 시스템 조합 자동 선택
- 병렬/순차 실행 전략 결정
- 결과 통합 및 품질 보장
- 시스템 간 지식 공유 조율
- 실행 계획 수립 및 적응적 수정

인간 대비 우위:
- 동시에 모든 AI 시스템의 능력을 평가하고 조합
- 실행 중 결과를 보고 전략을 실시간 변경
- 어떤 유형의 문제도 최적의 방법으로 해결
"""

import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """실행 계획"""
    problem: str
    complexity: str          # simple / moderate / complex / ultra
    selected_systems: List[str]
    parallel_groups: List[List[str]]   # 병렬 실행 그룹
    sequential_order: List[str]        # 순차 실행 순서
    estimated_time: float
    rationale: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionResult:
    """실행 결과"""
    problem: str
    plan: ExecutionPlan
    system_results: Dict[str, Any]
    final_answer: str
    confidence: float
    total_duration: float
    systems_used: List[str]
    insights: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# 복잡도 판단 키워드
COMPLEXITY_SIGNALS = {
    "simple": ["안녕", "뭐야", "알려줘", "hello", "what is", "define", "설명해"],
    "moderate": ["분석해", "비교해", "방법", "how to", "analyze", "compare", "explain"],
    "complex": ["연구해", "개발해", "설계해", "전략", "research", "develop", "design", "strategy"],
    "ultra": ["초월", "혁신", "혁명", "완전", "자율", "AGI", "superhuman", "breakthrough"],
}

# 시스템-문제 유형 매핑
SYSTEM_AFFINITY = {
    "tot": ["분석", "추론", "판단", "선택", "reasoning", "decide", "evaluate", "복잡한"],
    "swarm": ["전략", "종합", "다각도", "strategy", "comprehensive", "multi-aspect"],
    "kg": ["관계", "연결", "개념", "지식", "relationship", "concept", "knowledge"],
    "debate": ["논쟁", "찬반", "검증", "fact", "verify", "pros cons", "debate"],
    "deep_researcher": ["논문", "연구", "최신", "paper", "research", "arxiv", "trend"],
    "code_intelligence": ["코드", "프로그램", "함수", "code", "function", "script", "bug"],
    "web": ["검색", "정보", "뉴스", "search", "news", "latest", "current"],
    "goal_hierarchy": ["목표", "계획", "단계", "goal", "plan", "step", "achieve"],
    "consciousness": ["자체", "검증", "품질", "self", "verify", "quality", "check"],
}


class ExecutiveController:
    """
    JARVIS 총사령관
    모든 AI 능력을 최적으로 조합하여 어떤 문제도 해결
    """

    PLAN_PROMPT = """문제/요청: {problem}

사용 가능한 AI 시스템들:
- tot: Tree of Thoughts 다중 추론 (복잡한 분석)
- swarm: 에이전트 스웜 (다각도 전문가 분석)
- kg: 지식 그래프 추론 (개념 관계 분석)
- debate: 멀티 에이전트 토론 (검증/팩트체크)
- deep_researcher: 딥 리서치 (학술/기술 연구)
- code_intelligence: 코드 생성/분석
- web: 웹 검색 (최신 정보)
- goal_hierarchy: 목표 분해 실행
- direct: 직접 LLM 응답 (간단한 질문)

JSON으로 최적 실행 계획 반환:
{{
  "complexity": "simple|moderate|complex|ultra",
  "selected_systems": ["system1", "system2"],
  "parallel_groups": [["system1", "system2"], ["system3"]],
  "sequential_order": ["system1", "system2", "system3"],
  "estimated_time": 30,
  "rationale": "이 시스템들을 선택한 이유"
}}

JSON만 출력하세요."""

    SYNTHESIS_PROMPT = """원래 문제: {problem}

각 AI 시스템의 분석 결과:
{results}

이 모든 결과를 통합하여 최종 답변을 작성하세요:
- 모든 관점의 핵심 인사이트 포함
- 모순되는 부분은 명확히 표시
- 구체적이고 실행 가능한 결론
- 불확실한 부분은 솔직하게 표현

최고 수준의 종합 답변을 제공하세요."""

    def __init__(self, jarvis_engine, progress_callback: Optional[Callable] = None):
        self.jarvis = jarvis_engine
        self._progress_cb = progress_callback
        self._history: List[ExecutionResult] = []
        self._executor_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="EXEC")
        logger.info("ExecutiveController initialized — all systems under command")

    # ── 메인 실행 ──────────────────────────────────────────────────────────

    def execute(self, problem: str, force_ultra: bool = False) -> ExecutionResult:
        """문제 분석 → 계획 수립 → 실행 → 통합"""
        start = time.time()
        self._emit("exec_start", {"problem": problem[:100]})

        # 1. 복잡도 분석 및 계획 수립
        complexity = self._analyze_complexity(problem) if not force_ultra else "ultra"
        plan = self._create_plan(problem, complexity)
        self._emit("plan_ready", {
            "complexity": plan.complexity,
            "systems": plan.selected_systems,
            "rationale": plan.rationale,
        })
        logger.info(f"[Executive] Plan: {plan.complexity}, systems={plan.selected_systems}")

        # 2. 시스템 실행 (병렬/순차 혼합)
        system_results = self._execute_plan(problem, plan)

        # 3. 결과 통합
        final_answer, confidence, insights = self._synthesize(problem, system_results)

        result = ExecutionResult(
            problem=problem,
            plan=plan,
            system_results=system_results,
            final_answer=final_answer,
            confidence=confidence,
            total_duration=round(time.time() - start, 2),
            systems_used=list(system_results.keys()),
            insights=insights,
        )
        self._history.append(result)
        self._emit("exec_done", {
            "confidence": confidence,
            "duration": result.total_duration,
            "systems": result.systems_used,
        })
        return result

    def execute_streaming(self, problem: str) -> Generator[Dict, None, None]:
        """스트리밍 실행 — 각 단계 결과를 실시간 전달"""
        start = time.time()
        yield {"type": "analyzing", "message": "문제 분석 중..."}

        complexity = self._analyze_complexity(problem)
        plan = self._create_plan(problem, complexity)
        yield {"type": "plan", "complexity": plan.complexity, "systems": plan.selected_systems, "rationale": plan.rationale}

        system_results = {}
        for system in plan.selected_systems[:4]:  # 스트리밍은 순차적으로
            yield {"type": "running", "system": system}
            result = self._run_single_system(problem, system)
            system_results[system] = result
            summary = str(result)[:200] if result else "실패"
            yield {"type": "system_done", "system": system, "summary": summary}

        yield {"type": "synthesizing", "message": "결과 통합 중..."}
        final_answer, confidence, insights = self._synthesize(problem, system_results)
        yield {
            "type": "done",
            "answer": final_answer,
            "confidence": confidence,
            "insights": insights,
            "duration": round(time.time() - start, 2),
            "systems_used": list(system_results.keys()),
        }

    # ── 복잡도 분석 ────────────────────────────────────────────────────────

    def _analyze_complexity(self, problem: str) -> str:
        p = problem.lower()
        for level in ["ultra", "complex", "moderate", "simple"]:
            signals = COMPLEXITY_SIGNALS.get(level, [])
            if any(s in p for s in signals):
                return level
        # 길이 기반 추가 판단
        if len(problem) > 200:
            return "complex"
        elif len(problem) > 50:
            return "moderate"
        return "simple"

    def _create_plan(self, problem: str, complexity: str) -> ExecutionPlan:
        """실행 계획 수립 — LLM 또는 규칙 기반"""
        # 규칙 기반 빠른 계획
        if complexity == "simple":
            return ExecutionPlan(
                problem=problem, complexity=complexity,
                selected_systems=["direct"],
                parallel_groups=[["direct"]],
                sequential_order=["direct"],
                estimated_time=5,
                rationale="간단한 질문 — 직접 응답",
            )
        elif complexity == "moderate":
            systems = self._select_systems_by_keyword(problem, n=2)
            return ExecutionPlan(
                problem=problem, complexity=complexity,
                selected_systems=systems,
                parallel_groups=[systems],
                sequential_order=systems,
                estimated_time=20,
                rationale=f"중간 복잡도 — {', '.join(systems)} 사용",
            )
        elif complexity == "complex":
            systems = self._select_systems_by_keyword(problem, n=3)
            systems.append("consciousness")  # 품질 검증
            return ExecutionPlan(
                problem=problem, complexity=complexity,
                selected_systems=systems,
                parallel_groups=[systems[:2], systems[2:]],
                sequential_order=systems,
                estimated_time=60,
                rationale=f"고복잡도 — {', '.join(systems)} 사용",
            )
        else:  # ultra
            systems = ["tot", "swarm", "deep_researcher", "kg", "debate"]
            return ExecutionPlan(
                problem=problem, complexity=complexity,
                selected_systems=systems,
                parallel_groups=[["tot", "swarm"], ["deep_researcher", "kg"], ["debate"]],
                sequential_order=systems,
                estimated_time=120,
                rationale="초고복잡도 — 전체 시스템 동원",
            )

    def _select_systems_by_keyword(self, problem: str, n: int) -> List[str]:
        p = problem.lower()
        scores: Dict[str, int] = {}
        for system, keywords in SYSTEM_AFFINITY.items():
            scores[system] = sum(1 for kw in keywords if kw in p)
        ranked = sorted(scores, key=scores.get, reverse=True)
        selected = [s for s in ranked if scores[s] > 0][:n]
        if not selected:
            selected = ["tot", "web"][:n]
        return selected

    # ── 실행 ───────────────────────────────────────────────────────────────

    def _execute_plan(self, problem: str, plan: ExecutionPlan) -> Dict[str, Any]:
        """병렬/순차 혼합 실행"""
        results = {}

        for group in plan.parallel_groups:
            if len(group) == 1:
                # 단일 시스템 직접 실행
                system = group[0]
                r = self._run_single_system(problem, system)
                results[system] = r
            else:
                # 그룹 병렬 실행
                futures = {
                    self._executor_pool.submit(self._run_single_system, problem, s): s
                    for s in group
                }
                for future in as_completed(futures, timeout=90):
                    system = futures[future]
                    try:
                        results[system] = future.result(timeout=90)
                    except FuturesTimeout:
                        results[system] = {"error": "timeout"}
                        logger.warning(f"[Executive] {system} timed out")
                    except Exception as e:
                        results[system] = {"error": str(e)}

        return results

    def _run_single_system(self, problem: str, system: str) -> Any:
        """단일 시스템 실행"""
        j = self.jarvis
        try:
            if system == "direct":
                result = j.chat(problem, use_tools=False)
                return {"text": result.get("response", ""), "confidence": 0.7}

            elif system == "tot" and j.tot:
                tree = j.tot.think(problem, strategy="beam", branching=3, max_depth=3)
                return {"text": tree.final_answer, "confidence": tree.confidence, "thoughts": tree.total_thoughts}

            elif system == "swarm" and j.swarm:
                r = j.swarm.execute(problem, max_agents=4)
                return {"text": r.synthesis, "confidence": r.confidence, "consensus": r.consensus_points}

            elif system == "kg" and j.kg:
                r = j.kg.reason(problem)
                return {"text": r.get("answer", ""), "concepts": r.get("relevant_concepts", [])}

            elif system == "debate" and j.debate_engine:
                r = j.debate_engine.debate(problem, fast_mode=True)
                return {"text": r.synthesis, "confidence": r.confidence, "consensus": r.consensus_level}

            elif system == "deep_researcher" and j.deep_researcher:
                r = j.deep_researcher.research(problem, depth=1)
                return {"text": r.executive_summary, "findings": r.key_findings, "confidence": r.confidence}

            elif system == "code_intelligence" and j.code_intelligence:
                r = j.code_intelligence.generate(requirement=problem, language="python")
                return {"code": r.code, "explanation": r.explanation}

            elif system == "web" and j.web:
                results = j.web.search_web(problem, max_results=5)
                snippets = [r.get("snippet", "") for r in results if isinstance(r, dict)][:3]
                return {"text": " ".join(snippets), "sources": len(results)}

            elif system == "goal_hierarchy" and j.goals:
                goal = j.goals.create_goal(problem, auto_decompose=True)
                return {"goal_id": goal.id, "children": len(goal.children)}

            elif system == "consciousness" and j.consciousness:
                # 이전 결과들 평가에 사용
                return {"status": "monitoring"}

        except Exception as e:
            logger.error(f"[Executive] System {system} error: {e}")
            return {"error": str(e)}

        return {"error": f"System '{system}' not available"}

    # ── 결과 통합 ──────────────────────────────────────────────────────────

    def _synthesize(
        self, problem: str, system_results: Dict[str, Any]
    ) -> Tuple[str, float, List[str]]:
        """모든 결과를 통합하여 최종 답변 생성"""
        from jarvis.llm.manager import Message

        # 결과 텍스트 수집
        result_texts = []
        confidences = []
        insights = []

        for system, result in system_results.items():
            if not result or result.get("error"):
                continue
            text = result.get("text") or result.get("code") or result.get("explanation", "")
            if text and len(text) > 10:
                result_texts.append(f"[{system.upper()}]\n{text[:600]}")
                if "confidence" in result:
                    confidences.append(float(result["confidence"]))
                # 인사이트 수집
                for key in ["findings", "consensus", "concepts"]:
                    if key in result and isinstance(result[key], list):
                        insights.extend([str(i) for i in result[key][:2]])

        if not result_texts:
            # 폴백: 직접 LLM 응답
            direct = self.jarvis.chat(problem)
            return direct.get("response", "답변 생성 실패"), 0.5, []

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.7

        # 하나의 결과만 있으면 그대로 반환
        if len(result_texts) == 1:
            text = result_texts[0].split("\n", 1)[1] if "\n" in result_texts[0] else result_texts[0]
            return text, avg_confidence, insights[:5]

        # 여러 결과 통합
        try:
            results_combined = "\n\n".join(result_texts[:4])
            prompt = self.SYNTHESIS_PROMPT.format(problem=problem, results=results_combined)
            resp = self.jarvis.llm.chat([Message(role="user", content=prompt)], max_tokens=4096)
            return resp.content, avg_confidence, insights[:5]
        except Exception as e:
            logger.error(f"[Executive] Synthesis error: {e}")
            return result_texts[0], avg_confidence, insights[:3]

    # ── 통계 ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        if not self._history:
            return {"total_executions": 0}
        complexities = [r.plan.complexity for r in self._history]
        from collections import Counter
        return {
            "total_executions": len(self._history),
            "complexity_distribution": dict(Counter(complexities)),
            "avg_confidence": round(sum(r.confidence for r in self._history) / len(self._history), 3),
            "avg_duration": round(sum(r.total_duration for r in self._history) / len(self._history), 2),
            "most_used_systems": dict(Counter(s for r in self._history for s in r.systems_used).most_common(5)),
        }

    def get_history(self) -> List[Dict]:
        return [
            {
                "problem": r.problem[:80],
                "complexity": r.plan.complexity,
                "systems": r.systems_used,
                "confidence": r.confidence,
                "duration": r.total_duration,
                "timestamp": r.timestamp,
            }
            for r in reversed(self._history[-20:])
        ]

    def _emit(self, event_type: str, data: Dict):
        if self._progress_cb:
            try:
                self._progress_cb({"type": event_type, **data, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass

    def __del__(self):
        try:
            self._executor_pool.shutdown(wait=False)
        except Exception:
            pass
