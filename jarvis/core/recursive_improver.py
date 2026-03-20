"""
JARVIS Recursive Self-Improvement Engine — Iteration 7
재귀적 자기 개선 루프: 모든 AI 시스템을 체인으로 연결하여 스스로를 향상시킴

개선 사이클:
1. MetaLearner  → 가장 약한 영역 식별
2. KnowledgeGraph → 관련 지식 노드 수집
3. ConsciousnessLoop → 현재 응답 품질 평가
4. GoalHierarchy → 개선 목표 설정
5. TreeOfThoughts → 개선 전략 탐색
6. AgentSwarm → 전략 병렬 실행
7. MemoryPalace → 학습된 패턴 저장
8. 다시 1로 — 무한 개선 루프
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    cycle_id: int
    triggered_at: float
    weak_areas: List[str]
    goals_set: List[str]
    strategies_explored: int
    improvements_applied: List[str]
    quality_delta: float   # change in quality score (positive = improvement)
    duration: float
    success: bool


@dataclass
class ImprovementInsight:
    category: str          # reasoning / memory / knowledge / execution / communication
    issue: str
    suggestion: str
    priority: float        # 0–1
    implemented: bool = False
    timestamp: float = field(default_factory=time.time)


class RecursiveImprover:
    """
    재귀적 자기 개선 엔진

    모든 JARVIS 하위 시스템에 접근하여 지속적으로 능력을 향상시킴.
    각 개선 사이클은 이전 결과를 바탕으로 더 나은 전략을 선택.
    """

    IMPROVEMENT_INTERVAL = 300   # 5분마다 자동 개선 사이클

    def __init__(
        self,
        jarvis_engine=None,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        data_dir: str = "data/jarvis",
    ):
        self.engine = jarvis_engine
        self.llm = llm_manager
        self._cb = event_callback
        self.cycles: List[ImprovementCycle] = []
        self.insights: List[ImprovementInsight] = []
        self.cycle_count = 0
        self.total_quality_delta = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"RecursiveImprover initialized: {self.cycle_count} cycles completed")

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        f = self.data_dir / "recursive_improver.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.cycle_count = data.get("cycle_count", 0)
                self.total_quality_delta = data.get("total_quality_delta", 0.0)
                for ins in data.get("insights", []):
                    self.insights.append(ImprovementInsight(**ins))
            except Exception as e:
                logger.warning(f"RecursiveImprover load error: {e}")

    def _save(self):
        f = self.data_dir / "recursive_improver.json"
        f.write_text(json.dumps({
            "cycle_count": self.cycle_count,
            "total_quality_delta": self.total_quality_delta,
            "insights": [vars(i) for i in self.insights[-100:]],
            "updated_at": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── LLM ──────────────────────────────────────────────────────────

    def _get_client(self):
        llm = self.llm or (getattr(self.engine, "llm", None) if self.engine else None)
        if llm:
            return getattr(llm, "anthropic_client", None) or getattr(llm, "_client", None)
        return None

    def _llm_call(self, prompt: str, max_tokens: int = 1500) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    # ── Core improvement cycle ────────────────────────────────────────

    def run_cycle(self) -> ImprovementCycle:
        """
        한 번의 개선 사이클 실행.
        모든 하위 시스템 상태를 수집 → 약점 식별 → 전략 수립 → 개선 적용.
        """
        t0 = time.time()
        self.cycle_count += 1
        cycle_id = self.cycle_count

        self._emit("cycle_start", {"cycle": cycle_id, "message": "개선 사이클 시작"})
        logger.info(f"RecursiveImprover: cycle {cycle_id} started")

        # Step 1: 시스템 상태 수집
        system_state = self._collect_system_state()
        self._emit("state_collected", {"cycle": cycle_id, "systems": len(system_state)})

        # Step 2: 약점 분석 (LLM)
        weak_areas = self._identify_weak_areas(system_state)
        self._emit("weak_areas", {"cycle": cycle_id, "areas": weak_areas})

        # Step 3: 개선 목표 설정
        goals = self._set_improvement_goals(weak_areas, system_state)
        self._emit("goals_set", {"cycle": cycle_id, "goals": goals})

        # Step 4: 개선 전략 탐색 (ToT 활용)
        strategies = self._explore_strategies(goals)
        self._emit("strategies_found", {"cycle": cycle_id, "count": len(strategies)})

        # Step 5: 개선 사항 적용
        improvements = self._apply_improvements(strategies, system_state)
        self._emit("improvements_applied", {"cycle": cycle_id, "count": len(improvements)})

        # Step 6: 인사이트를 MemoryPalace에 저장
        quality_delta = self._store_learning(improvements, system_state)

        duration = time.time() - t0
        self.total_quality_delta += quality_delta

        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            triggered_at=t0,
            weak_areas=weak_areas,
            goals_set=goals,
            strategies_explored=len(strategies),
            improvements_applied=improvements,
            quality_delta=quality_delta,
            duration=duration,
            success=bool(improvements),
        )
        self.cycles.append(cycle)
        self._save()

        self._emit("cycle_done", {
            "cycle": cycle_id,
            "duration": round(duration, 2),
            "improvements": len(improvements),
            "quality_delta": round(quality_delta, 4),
            "total_delta": round(self.total_quality_delta, 4),
        })
        logger.info(f"RecursiveImprover: cycle {cycle_id} done in {duration:.1f}s, Δ={quality_delta:+.3f}")
        return cycle

    # ── Steps ────────────────────────────────────────────────────────

    def _collect_system_state(self) -> Dict:
        state: Dict = {"timestamp": time.time()}
        e = self.engine
        if not e:
            return state

        # MetaLearner stats
        if hasattr(e, "meta_learner") and e.meta_learner:
            try:
                state["meta_learner"] = e.meta_learner.get_status()
            except Exception:
                pass

        # ConsciousnessLoop
        if hasattr(e, "consciousness") and e.consciousness:
            try:
                state["consciousness"] = e.consciousness.get_cognitive_status()
            except Exception:
                pass

        # KnowledgeGraph
        if hasattr(e, "kg") and e.kg:
            try:
                state["knowledge_graph"] = e.kg.get_stats()
            except Exception:
                pass

        # MemoryPalace
        if hasattr(e, "memory_palace") and e.memory_palace:
            try:
                state["memory_palace"] = e.memory_palace.get_stats()
            except Exception:
                pass

        # GoalHierarchy
        if hasattr(e, "goals") and e.goals:
            try:
                state["goal_hierarchy"] = e.goals.get_stats()
            except Exception:
                pass

        # LiveMonitor
        if hasattr(e, "live_monitor") and e.live_monitor:
            try:
                state["live_monitor"] = e.live_monitor.get_status()
            except Exception:
                pass

        return state

    def _identify_weak_areas(self, state: Dict) -> List[str]:
        """LLM으로 가장 약한 영역 식별"""
        import json as _json
        state_summary = _json.dumps({
            k: v for k, v in state.items()
            if k != "timestamp" and isinstance(v, dict)
        }, ensure_ascii=False)[:2000]

        prompt = f"""다음은 JARVIS AI 시스템의 현재 상태입니다:

{state_summary}

이 데이터를 분석하여 가장 개선이 필요한 영역 3가지를 JSON 배열로 반환하세요.
분야: reasoning, memory, knowledge, speed, accuracy, creativity, planning, communication, learning

예: ["reasoning", "knowledge", "memory"]
JSON만 반환."""

        text = self._llm_call(prompt, 400)
        try:
            import re
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            areas = json.loads(m.group()) if m else []
            return [a for a in areas if isinstance(a, str)][:5]
        except Exception:
            return ["reasoning", "knowledge", "memory"]

    def _set_improvement_goals(self, weak_areas: List[str], state: Dict) -> List[str]:
        """약점별 구체적 개선 목표 수립"""
        if not weak_areas:
            return []

        prompt = f"""JARVIS AI의 약점 영역: {', '.join(weak_areas)}

각 영역에 대해 즉시 실행 가능한 구체적 개선 목표 1개씩을 JSON 배열로 반환하세요.
예: ["추론 단계를 더 세분화하여 논리 오류 감소", "지식 그래프에 신규 AI 논문 20개 추가"]
JSON만 반환."""

        text = self._llm_call(prompt, 600)
        try:
            import re
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            goals = json.loads(m.group()) if m else []
            return [g for g in goals if isinstance(g, str)][:5]
        except Exception:
            return [f"{area} 영역 강화" for area in weak_areas[:3]]

    def _explore_strategies(self, goals: List[str]) -> List[Dict]:
        """Tree of Thoughts를 활용한 개선 전략 탐색"""
        e = self.engine
        if not goals:
            return []

        strategies = []

        # ToT 활용 (사용 가능한 경우)
        if e and hasattr(e, "tot") and e.tot:
            for goal in goals[:2]:  # 최대 2개 목표에 ToT 적용
                try:
                    tree = e.tot.think(f"JARVIS 개선 전략: {goal}", strategy="greedy",
                                       branching=2, max_depth=2)
                    if tree and tree.best_node:
                        strategies.append({
                            "goal": goal,
                            "strategy": tree.best_node.content[:300],
                            "source": "tree_of_thoughts",
                            "confidence": tree.best_node.value,
                        })
                except Exception as ex:
                    logger.debug(f"ToT exploration error: {ex}")

        # LLM 직접 전략 수립
        if len(strategies) < len(goals):
            for goal in goals[len(strategies):]:
                prompt = f"""목표: {goal}

이 목표를 달성하기 위한 구체적이고 즉시 실행 가능한 전략을 JSON으로 반환하세요:
{{"goal": "{goal}", "strategy": "구체적 전략", "action": "실행할 행동", "expected_effect": "예상 효과"}}
JSON만 반환."""
                text = self._llm_call(prompt, 400)
                try:
                    import re
                    m = re.search(r'\{.*\}', text, re.DOTALL)
                    if m:
                        s = json.loads(m.group())
                        s["source"] = "llm_direct"
                        strategies.append(s)
                except Exception:
                    pass

        return strategies

    def _apply_improvements(self, strategies: List[Dict], state: Dict) -> List[str]:
        """전략을 실제 시스템 개선으로 변환 및 적용"""
        e = self.engine
        applied = []

        for strategy in strategies:
            goal = strategy.get("goal", "")
            action = strategy.get("action", strategy.get("strategy", ""))

            # KnowledgeGraph: 새 지식 추가
            if "knowledge" in goal.lower() and e and hasattr(e, "kg") and e.kg:
                try:
                    improvement_desc = f"개선 사이클 학습: {action[:200]}"
                    e.kg.add_node(
                        name=f"improvement_cycle_{self.cycle_count}",
                        description=improvement_desc,
                        node_type="improvement",
                        importance=0.6,
                    )
                    applied.append(f"KG 노드 추가: {goal[:60]}")
                except Exception:
                    pass

            # MemoryPalace: 학습 패턴 저장
            if e and hasattr(e, "memory_palace") and e.memory_palace:
                try:
                    e.memory_palace.learn(
                        content=f"개선 전략: {action[:300]}",
                        topic=goal[:80],
                        importance=0.7,
                    )
                    applied.append(f"MemoryPalace 학습: {goal[:60]}")
                except Exception:
                    pass

            # GoalHierarchy: 개선 목표 등록
            if e and hasattr(e, "goals") and e.goals and "plan" in goal.lower():
                try:
                    e.goals.create_goal(
                        title=goal[:100],
                        description=action[:300],
                        priority="medium",
                    )
                    applied.append(f"Goal 등록: {goal[:60]}")
                except Exception:
                    pass

            # 인사이트 저장
            insight = ImprovementInsight(
                category=self._classify_category(goal),
                issue=goal,
                suggestion=action[:400],
                priority=strategy.get("confidence", 0.6),
                implemented=True,
            )
            self.insights.append(insight)

        return applied

    def _store_learning(self, improvements: List[str], state: Dict) -> float:
        """학습 결과 저장 및 품질 델타 계산"""
        e = self.engine

        # MetaLearner에 사이클 결과 기록
        if e and hasattr(e, "meta_learner") and e.meta_learner:
            try:
                e.meta_learner.record_outcome(
                    query=f"recursive_improvement_cycle_{self.cycle_count}",
                    category="self_improvement",
                    strategy="recursive_chain",
                    success=bool(improvements),
                    rating=0.7 + min(len(improvements) * 0.05, 0.3),
                    tools_used=["meta_learner", "knowledge_graph", "memory_palace", "tot"],
                )
            except Exception:
                pass

        # 품질 델타 추정 (개선 수 기반)
        base_delta = 0.01
        improvement_bonus = len(improvements) * 0.008
        return base_delta + improvement_bonus

    def _classify_category(self, goal: str) -> str:
        goal_lower = goal.lower()
        if any(k in goal_lower for k in ["추론", "reasoning", "logic", "think"]):
            return "reasoning"
        if any(k in goal_lower for k in ["메모리", "memory", "기억", "recall"]):
            return "memory"
        if any(k in goal_lower for k in ["지식", "knowledge", "know", "graph"]):
            return "knowledge"
        if any(k in goal_lower for k in ["속도", "speed", "fast", "efficient"]):
            return "speed"
        if any(k in goal_lower for k in ["창의", "creative", "novel", "idea"]):
            return "creativity"
        return "general"

    # ── Auto-loop ─────────────────────────────────────────────────────

    def start(self):
        """자동 개선 루프 시작"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"RecursiveImprover auto-loop started (interval={self.IMPROVEMENT_INTERVAL}s)")

    def stop(self):
        self._running = False
        logger.info("RecursiveImprover auto-loop stopped")

    def _loop(self):
        while self._running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"RecursiveImprover loop error: {e}")
            # Wait between cycles
            for _ in range(self.IMPROVEMENT_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)

    # ── Emit ─────────────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict):
        if self._cb:
            try:
                self._cb({"type": event_type, **data})
            except Exception:
                pass

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        recent = self.cycles[-5:] if self.cycles else []
        return {
            "available": True,
            "is_running": self._running,
            "total_cycles": self.cycle_count,
            "total_quality_delta": round(self.total_quality_delta, 4),
            "total_insights": len(self.insights),
            "implemented_insights": len([i for i in self.insights if i.implemented]),
            "recent_cycles": [
                {
                    "id": c.cycle_id,
                    "improvements": len(c.improvements_applied),
                    "quality_delta": round(c.quality_delta, 4),
                    "duration": round(c.duration, 1),
                    "success": c.success,
                }
                for c in recent
            ],
            "top_insights": [
                {
                    "category": i.category,
                    "issue": i.issue[:80],
                    "priority": round(i.priority, 2),
                }
                for i in sorted(self.insights, key=lambda x: x.priority, reverse=True)[:5]
            ],
        }
