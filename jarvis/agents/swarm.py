"""
JARVIS 에이전트 스웜 — Iteration 5
병렬 에이전트 군집으로 복잡한 문제를 인간보다 빠르게 해결
- 동적 에이전트 생성 및 역할 할당
- 병렬 실행 (ThreadPoolExecutor)
- 결과 집계 및 모순 해결
- 전문화 에이전트 라우팅
- 스웜 인텔리전스 (집단 지성)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_PARALLEL = 6        # 동시 실행 에이전트 수
AGENT_TIMEOUT = 60      # 에이전트 타임아웃 (초)


@dataclass
class SwarmAgent:
    agent_id: str
    role: str
    system_prompt: str
    task: str
    result: str = ""
    success: bool = False
    duration: float = 0.0
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)


@dataclass
class SwarmResult:
    goal: str
    agents: List[SwarmAgent]
    synthesis: str
    confidence: float
    contradictions: List[str]
    consensus_points: List[str]
    recommended_action: str
    total_duration: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def success_rate(self) -> float:
        if not self.agents:
            return 0.0
        return sum(1 for a in self.agents if a.success) / len(self.agents)


class AgentSwarm:
    """
    JARVIS 에이전트 스웜
    복잡한 목표를 전문화된 에이전트 군집으로 병렬 처리
    """

    SPECIALIST_ROLES = {
        "analyst": {
            "system": "당신은 데이터 분석 전문가입니다. 수치, 통계, 패턴을 찾아 인사이트를 도출합니다.",
            "focus": "데이터 분석, 통계, 패턴 인식",
        },
        "researcher": {
            "system": "당신은 최고 연구자입니다. 최신 논문, 연구 결과, 전문 지식을 활용합니다.",
            "focus": "연구, 문헌 검토, 최신 기술",
        },
        "critic": {
            "system": "당신은 비판적 사고 전문가입니다. 모든 주장의 약점, 반증, 위험을 찾습니다.",
            "focus": "비판적 분석, 위험 평가, 반증",
        },
        "implementer": {
            "system": "당신은 실용적 구현 전문가입니다. 이론을 실제 작동하는 솔루션으로 변환합니다.",
            "focus": "구현, 실행, 최적화",
        },
        "strategist": {
            "system": "당신은 전략 컨설턴트입니다. 장기적 관점에서 최선의 전략을 수립합니다.",
            "focus": "전략, 계획, 로드맵",
        },
        "domain_expert": {
            "system": "당신은 관련 분야 최고 전문가입니다. 깊은 도메인 지식을 제공합니다.",
            "focus": "전문 지식, 업계 인사이트",
        },
    }

    AGENT_TASK_PROMPT = """목표: {goal}

당신의 역할: {role} — {focus}

다음 관점에서 분석하여 JSON으로 반환하세요:
{{
  "analysis": "상세 분석 (3-5 문단)",
  "key_findings": ["핵심 발견 1", "핵심 발견 2", "핵심 발견 3"],
  "recommendations": ["권고사항 1", "권고사항 2"],
  "confidence": 0.85,
  "caveats": ["주의사항 1", "주의사항 2"]
}}

JSON만 출력하세요."""

    SYNTHESIS_PROMPT = """다음 전문가들이 목표를 분석했습니다:
목표: {goal}

{agent_results}

모든 분석을 통합하여 JSON으로 반환:
{{
  "synthesis": "통합 결론 (상세)",
  "consensus_points": ["합의된 핵심 1", "합의된 핵심 2"],
  "contradictions": ["이견/모순 1", "이견/모순 2"],
  "recommended_action": "최종 권고 행동",
  "confidence": 0.82
}}"""

    def __init__(self, llm_manager, progress_callback: Optional[Callable] = None):
        self.llm = llm_manager
        self._progress_cb = progress_callback
        self._history: List[SwarmResult] = []
        logger.info("AgentSwarm initialized")

    def execute(
        self,
        goal: str,
        roles: Optional[List[str]] = None,
        max_agents: int = 4,
    ) -> SwarmResult:
        """스웜 실행 — 병렬 에이전트 군집으로 목표 해결"""
        start = time.time()
        selected_roles = roles or self._select_roles_for_goal(goal, max_agents)
        self._emit("swarm_start", {"goal": goal[:100], "agents": selected_roles})

        # 에이전트 객체 생성
        agents = []
        for role in selected_roles:
            cfg = self.SPECIALIST_ROLES.get(role, self.SPECIALIST_ROLES["analyst"])
            task = self.AGENT_TASK_PROMPT.format(
                goal=goal, role=role, focus=cfg["focus"]
            )
            agents.append(SwarmAgent(
                agent_id=f"{role}_{int(time.time())}",
                role=role,
                system_prompt=cfg["system"],
                task=task,
            ))

        # 병렬 실행
        completed_agents = self._run_parallel(agents)

        # 합성
        synthesis_data = self._synthesize(goal, completed_agents)

        result = SwarmResult(
            goal=goal,
            agents=completed_agents,
            synthesis=synthesis_data.get("synthesis", "합성 실패"),
            confidence=synthesis_data.get("confidence", 0.5),
            contradictions=synthesis_data.get("contradictions", []),
            consensus_points=synthesis_data.get("consensus_points", []),
            recommended_action=synthesis_data.get("recommended_action", ""),
            total_duration=round(time.time() - start, 2),
        )

        self._history.append(result)
        self._emit("swarm_done", {
            "goal": goal[:100],
            "agents": len(completed_agents),
            "success_rate": result.success_rate,
            "duration": result.total_duration,
        })
        return result

    def _run_parallel(self, agents: List[SwarmAgent]) -> List[SwarmAgent]:
        """ThreadPoolExecutor로 에이전트 병렬 실행"""
        results = []
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(agents))) as pool:
            future_to_agent = {
                pool.submit(self._run_single_agent, agent): agent
                for agent in agents
            }
            for future in as_completed(future_to_agent, timeout=AGENT_TIMEOUT * 2):
                agent = future_to_agent[future]
                try:
                    completed = future.result(timeout=AGENT_TIMEOUT)
                    results.append(completed)
                    self._emit("agent_done", {
                        "role": agent.role,
                        "success": completed.success,
                        "confidence": completed.confidence,
                    })
                except TimeoutError:
                    agent.success = False
                    agent.result = f"타임아웃 ({AGENT_TIMEOUT}s)"
                    results.append(agent)
                    logger.warning(f"Agent {agent.role} timed out")
                except Exception as e:
                    agent.success = False
                    agent.result = f"오류: {e}"
                    results.append(agent)
        return results

    def _run_single_agent(self, agent: SwarmAgent) -> SwarmAgent:
        """단일 에이전트 실행"""
        from jarvis.llm.manager import Message
        start = time.time()
        self._emit("agent_thinking", {"role": agent.role})

        try:
            messages = [Message(role="user", content=agent.task)]
            response = self.llm.chat(
                messages, system=agent.system_prompt, max_tokens=3000
            )
            import re, json as js
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = js.loads(match.group())
                agent.result = data.get("analysis", response.content[:1000])
                agent.confidence = float(data.get("confidence", 0.7))
                agent.success = True
            else:
                agent.result = response.content[:1000]
                agent.confidence = 0.6
                agent.success = True
        except Exception as e:
            agent.result = f"실패: {e}"
            agent.success = False
            logger.error(f"Agent {agent.role} error: {e}")
        finally:
            agent.duration = round(time.time() - start, 2)
        return agent

    def _select_roles_for_goal(self, goal: str, n: int) -> List[str]:
        """목표에 맞는 에이전트 역할 자동 선택"""
        g = goal.lower()
        scores = {}
        keyword_map = {
            "analyst": ["분석", "데이터", "통계", "패턴", "analyze"],
            "researcher": ["논문", "연구", "최신", "트렌드", "paper"],
            "critic": ["위험", "단점", "문제", "risk", "cons"],
            "implementer": ["구현", "코드", "개발", "만들어", "build"],
            "strategist": ["전략", "계획", "장기", "로드맵", "strategy"],
            "domain_expert": ["전문", "expert", "deep", "심층"],
        }
        for role, kws in keyword_map.items():
            scores[role] = sum(1 for kw in kws if kw in g)

        # 점수 기반 상위 n개 선택 + 최소 1개 critic 포함
        sorted_roles = sorted(scores, key=scores.get, reverse=True)
        selected = sorted_roles[:n]
        if "critic" not in selected and n >= 3:
            selected[-1] = "critic"
        return selected

    def _synthesize(self, goal: str, agents: List[SwarmAgent]) -> Dict:
        """모든 에이전트 결과 합성"""
        from jarvis.llm.manager import Message
        successful = [a for a in agents if a.success]
        if not successful:
            return {"synthesis": "모든 에이전트 실패", "confidence": 0.0}

        agent_results = "\n\n".join([
            f"**{a.role}** (신뢰도: {a.confidence:.0%}):\n{a.result[:600]}"
            for a in successful
        ])

        prompt = self.SYNTHESIS_PROMPT.format(goal=goal, agent_results=agent_results)
        try:
            messages = [Message(role="user", content=prompt)]
            resp = self.llm.chat(
                messages,
                system="수석 분석가로서 여러 전문가 의견을 통합합니다. JSON만 출력.",
                max_tokens=4096,
            )
            import re, json as js
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                return js.loads(match.group())
        except Exception as e:
            logger.error(f"Swarm synthesis error: {e}")

        # 폴백: 최고 신뢰도 에이전트 결과
        best = max(successful, key=lambda a: a.confidence)
        return {
            "synthesis": best.result,
            "confidence": best.confidence,
            "consensus_points": [],
            "contradictions": [],
            "recommended_action": "",
        }

    def _emit(self, event_type: str, data: Dict):
        if self._progress_cb:
            try:
                self._progress_cb({"type": event_type, **data})
            except Exception:
                pass

    def format_markdown(self, result: SwarmResult) -> str:
        lines = [
            f"# 스웜 인텔리전스 결과",
            f"**목표:** {result.goal}",
            f"**에이전트:** {len(result.agents)}개 | **성공률:** {result.success_rate:.0%} | **신뢰도:** {result.confidence:.0%} | **소요:** {result.total_duration}초",
            "",
            "## 에이전트별 분석",
        ]
        for a in result.agents:
            status = "✅" if a.success else "❌"
            lines.append(f"\n### {status} {a.role} ({a.confidence:.0%})")
            lines.append(a.result[:400])

        lines.extend(["", "## 통합 결론", result.synthesis])
        if result.consensus_points:
            lines.extend(["", "## 합의 사항"] + [f"- {p}" for p in result.consensus_points])
        if result.contradictions:
            lines.extend(["", "## 이견/모순"] + [f"- {c}" for c in result.contradictions])
        if result.recommended_action:
            lines.extend(["", f"## 최종 권고\n{result.recommended_action}"])
        return "\n".join(lines)

    def get_history(self) -> List[Dict]:
        return [
            {
                "goal": r.goal[:80],
                "agents": len(r.agents),
                "success_rate": r.success_rate,
                "confidence": r.confidence,
                "duration": r.total_duration,
                "timestamp": r.timestamp,
            }
            for r in reversed(self._history[-20:])
        ]
