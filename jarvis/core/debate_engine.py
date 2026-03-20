"""
JARVIS 멀티 에이전트 토론 엔진 — Iteration 4
하나의 질문에 대해 여러 관점의 에이전트가 내부 토론 후 최선의 답변을 합성
- 낙관론자 에이전트 / 비판론자 에이전트 / 현실론자 에이전트
- Devil's Advocate (반론자) 패턴
- 합의 도출 및 신뢰도 측정
- 불확실성 명시적 처리
- 인간 전문가 패널 시뮬레이션
"""

import json
import time
import logging
from typing import Dict, Generator, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentPerspective:
    agent_name: str
    role: str
    argument: str
    confidence: float
    key_points: List[str]
    concerns: List[str]


@dataclass
class DebateResult:
    question: str
    perspectives: List[AgentPerspective]
    synthesis: str
    consensus_level: float  # 0-1
    confidence: float
    dissenting_views: List[str]
    recommended_action: str
    duration: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DebateEngine:
    """
    JARVIS 내부 멀티 에이전트 토론 시스템
    단일 LLM 편향을 극복하고 다각도 분석으로 최선의 답변 도출
    """

    AGENTS = {
        "optimist": {
            "name": "낙관론자",
            "role": "기회와 가능성에 집중하는 혁신적 사고",
            "system": "당신은 기회와 가능성을 최대한 탐구하는 낙관론자입니다. 혁신적이고 창의적인 관점에서 분석하세요.",
        },
        "critic": {
            "name": "비판론자",
            "role": "위험, 한계, 반증에 집중하는 비판적 사고",
            "system": "당신은 위험, 한계, 반증을 철저히 분석하는 비판론자입니다. 모든 주장의 약점을 찾으세요.",
        },
        "pragmatist": {
            "name": "현실론자",
            "role": "실용성, 실행 가능성, 증거에 집중",
            "system": "당신은 실용성과 증거 기반 분석을 중시하는 현실론자입니다. 실제로 작동하는 것에 집중하세요.",
        },
        "expert": {
            "name": "도메인 전문가",
            "role": "해당 분야의 깊은 전문 지식 적용",
            "system": "당신은 관련 분야의 최고 전문가입니다. 전문 지식과 최신 연구를 기반으로 분석하세요.",
        },
        "devils_advocate": {
            "name": "악마의 변호인",
            "role": "기존 결론에 반론, 대안적 해석 제시",
            "system": "당신은 악마의 변호인입니다. 가장 그럴듯한 반론과 대안적 해석을 찾으세요.",
        },
    }

    DEBATE_PROMPT = """주제: {question}

당신의 역할: {role}

다음 형식으로 분석하세요:
{{
  "argument": "핵심 주장 (2-3 문단)",
  "key_points": ["핵심 포인트 1", "핵심 포인트 2", "핵심 포인트 3"],
  "concerns": ["우려 사항 1", "우려 사항 2"],
  "confidence": 0.85
}}

JSON만 출력하세요."""

    SYNTHESIS_PROMPT = """여러 전문가 에이전트가 다음 주제에 대해 토론했습니다:

주제: {question}

각 에이전트의 관점:
{perspectives}

이 토론을 바탕으로:
1. 모든 관점의 핵심을 통합한 최선의 답변을 작성하세요
2. 합의된 부분과 이견이 있는 부분을 구분하세요
3. 불확실성을 명확히 표현하세요
4. 구체적인 권고사항을 제시하세요

응답 형식:
{{
  "synthesis": "통합된 최선의 답변 (상세)",
  "consensus_points": ["합의 사항 1", "합의 사항 2"],
  "dissenting_views": ["이견 1", "이견 2"],
  "recommended_action": "구체적 권고 행동",
  "consensus_level": 0.75,
  "confidence": 0.85
}}"""

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self._history: List[DebateResult] = []

    def debate(
        self,
        question: str,
        agents: List[str] = None,
        fast_mode: bool = False,
    ) -> DebateResult:
        """
        멀티 에이전트 토론 실행
        fast_mode: True = 3개 에이전트만 (속도 우선)
        """
        start_time = time.time()
        selected_agents = agents or (
            ["optimist", "critic", "pragmatist"] if fast_mode
            else ["optimist", "critic", "pragmatist", "expert", "devils_advocate"]
        )

        perspectives = []
        for agent_id in selected_agents:
            agent_cfg = self.AGENTS.get(agent_id, self.AGENTS["pragmatist"])
            perspective = self._get_perspective(question, agent_id, agent_cfg)
            if perspective:
                perspectives.append(perspective)
                logger.info(f"[Debate] {agent_cfg['name']}: confidence={perspective.confidence:.2f}")

        # 합성
        synthesis_data = self._synthesize(question, perspectives)

        result = DebateResult(
            question=question,
            perspectives=perspectives,
            synthesis=synthesis_data.get("synthesis", "합성 실패"),
            consensus_level=synthesis_data.get("consensus_level", 0.5),
            confidence=synthesis_data.get("confidence", 0.7),
            dissenting_views=synthesis_data.get("dissenting_views", []),
            recommended_action=synthesis_data.get("recommended_action", ""),
            duration=round(time.time() - start_time, 2),
        )
        self._history.append(result)
        return result

    def debate_streaming(self, question: str, agents: List[str] = None) -> Generator[Dict, None, None]:
        """실시간 스트리밍 토론"""
        selected_agents = agents or ["optimist", "critic", "pragmatist", "expert"]
        yield {"type": "start", "question": question, "agents": selected_agents}

        perspectives = []
        for agent_id in selected_agents:
            agent_cfg = self.AGENTS.get(agent_id, self.AGENTS["pragmatist"])
            yield {"type": "agent_thinking", "agent": agent_cfg["name"], "role": agent_cfg["role"]}
            perspective = self._get_perspective(question, agent_id, agent_cfg)
            if perspective:
                perspectives.append(perspective)
                yield {
                    "type": "agent_done",
                    "agent": perspective.agent_name,
                    "argument": perspective.argument[:400],
                    "key_points": perspective.key_points,
                    "confidence": perspective.confidence,
                }

        yield {"type": "synthesizing"}
        synthesis_data = self._synthesize(question, perspectives)
        yield {
            "type": "done",
            "synthesis": synthesis_data.get("synthesis", ""),
            "consensus_level": synthesis_data.get("consensus_level", 0.5),
            "confidence": synthesis_data.get("confidence", 0.7),
            "recommended_action": synthesis_data.get("recommended_action", ""),
            "dissenting_views": synthesis_data.get("dissenting_views", []),
        }

    def _get_perspective(self, question: str, agent_id: str, agent_cfg: Dict) -> Optional[AgentPerspective]:
        """단일 에이전트 관점 획득"""
        from jarvis.llm.manager import Message
        prompt = self.DEBATE_PROMPT.format(question=question, role=agent_cfg["role"])
        messages = [Message(role="user", content=prompt)]
        try:
            response = self.llm.chat(messages, system=agent_cfg["system"], max_tokens=2048)
            import re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return AgentPerspective(
                    agent_name=agent_cfg["name"],
                    role=agent_cfg["role"],
                    argument=data.get("argument", ""),
                    confidence=float(data.get("confidence", 0.7)),
                    key_points=data.get("key_points", []),
                    concerns=data.get("concerns", []),
                )
        except Exception as e:
            logger.error(f"Agent {agent_id} error: {e}")
        return None

    def _synthesize(self, question: str, perspectives: List[AgentPerspective]) -> Dict:
        """모든 관점 합성"""
        from jarvis.llm.manager import Message
        if not perspectives:
            return {"synthesis": "에이전트 응답 없음", "consensus_level": 0, "confidence": 0}

        persp_text = "\n\n".join([
            f"**{p.agent_name}** (신뢰도: {p.confidence:.0%})\n"
            f"주장: {p.argument[:500]}\n"
            f"핵심: {', '.join(p.key_points[:3])}\n"
            f"우려: {', '.join(p.concerns[:2])}"
            for p in perspectives
        ])

        prompt = self.SYNTHESIS_PROMPT.format(question=question, perspectives=persp_text)
        messages = [Message(role="user", content=prompt)]

        try:
            system = """당신은 여러 전문가 의견을 통합하는 수석 분석가입니다.
편향 없이 모든 관점의 장점을 통합하여 가장 균형잡힌 답변을 제시하세요.
JSON만 출력하세요."""
            response = self.llm.chat(messages, system=system, max_tokens=4096)
            import re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Synthesis error: {e}")

        # 폴백: 가장 높은 신뢰도 관점 반환
        best = max(perspectives, key=lambda p: p.confidence)
        return {
            "synthesis": best.argument,
            "consensus_level": 0.5,
            "confidence": best.confidence,
            "dissenting_views": [],
            "recommended_action": "",
        }

    def quick_fact_check(self, claim: str) -> Dict:
        """주장의 사실 검증 (낙관/비판 2개 에이전트)"""
        result = self.debate(claim, agents=["expert", "devils_advocate"], fast_mode=True)
        return {
            "claim": claim,
            "verdict": "검증됨" if result.consensus_level > 0.7 else "의심됨" if result.consensus_level > 0.4 else "반증됨",
            "consensus": result.consensus_level,
            "analysis": result.synthesis[:500],
            "dissent": result.dissenting_views,
        }

    def format_debate_markdown(self, result: DebateResult) -> str:
        """토론 결과 마크다운"""
        lines = [
            f"# 멀티 에이전트 토론 결과",
            f"**주제:** {result.question}",
            f"**합의도:** {result.consensus_level:.0%} | **신뢰도:** {result.confidence:.0%} | **소요:** {result.duration}초",
            "",
            "## 에이전트별 관점",
        ]
        for p in result.perspectives:
            lines.extend([
                f"\n### {p.agent_name} ({p.confidence:.0%})",
                p.argument,
                f"**핵심 포인트:** {', '.join(p.key_points[:3])}",
                f"**우려사항:** {', '.join(p.concerns[:2])}" if p.concerns else "",
            ])

        lines.extend([
            "",
            "## 종합 결론",
            result.synthesis,
        ])
        if result.dissenting_views:
            lines.extend(["", "## 이견", *[f"- {v}" for v in result.dissenting_views]])
        if result.recommended_action:
            lines.extend(["", f"## 권고 행동\n{result.recommended_action}"])
        return "\n".join(lines)

    def get_history(self) -> List[Dict]:
        return [
            {
                "question": r.question[:80],
                "agents": len(r.perspectives),
                "consensus_level": r.consensus_level,
                "confidence": r.confidence,
                "duration": r.duration,
                "timestamp": r.timestamp,
            }
            for r in reversed(self._history[-20:])
        ]
