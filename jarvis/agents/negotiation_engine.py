"""
JARVIS 에이전트 협상 엔진 — Iteration 12
여러 AI 에이전트가 협상하고 합의하여 최적 결정을 도출한다

영감:
  - 게임 이론 (Nash Equilibrium, 파레토 최적)
  - 다중 에이전트 시스템 (MAS)
  - 협상 프로토콜 (Rubinstein Bargaining, Auction Theory)
  - 사회적 선택 이론 (Social Choice Theory)
  - 집단 지성 (Wisdom of Crowds)

핵심 개념:
  단일 LLM의 판단 → 여러 전문 에이전트의 협상 결과
  각 에이전트는 다른 관점/우선순위를 가짐
  협상을 통해 단일 에이전트보다 나은 결정 도달
  예: "최적 AI 아키텍처 선택" 시
    - 성능 에이전트: 정확도 최우선
    - 효율 에이전트: 비용/속도 최우선
    - 안전 에이전트: 견고성/안전 최우선
    → 협상 → 균형 잡힌 최적안
"""

import json
import time
import uuid
import logging
import threading
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    OPTIMIZER = "optimizer"             # 최적화 전문 (성능/비용)
    CRITIC = "critic"                   # 비판적 검토 (위험/단점)
    ADVOCATE = "advocate"               # 옹호 (장점/기회)
    PRAGMATIST = "pragmatist"           # 현실적 타협 (실용성)
    ETHICIST = "ethicist"               # 윤리적 판단
    INNOVATOR = "innovator"             # 창의적 대안 제시
    ANALYST = "analyst"                 # 데이터 기반 분석
    STRATEGIST = "strategist"           # 장기 전략


class NegotiationProtocol(Enum):
    DEBATE = "debate"                   # 토론 (찬반)
    AUCTION = "auction"                 # 경매 (최고 제안 선택)
    CONSENSUS = "consensus"             # 합의 (모두 동의)
    VOTING = "voting"                   # 투표 (다수결)
    MEDIATION = "mediation"             # 중재 (제3자 조율)


@dataclass
class NegotiationAgent:
    """협상 참여 에이전트"""
    agent_id: str
    name: str
    role: AgentRole
    priorities: List[str]               # 중시하는 가치 목록
    persona: str                        # 에이전트 성격/관점
    weight: float = 1.0                 # 발언 가중치

    def get_system_prompt(self) -> str:
        return (
            f"당신은 '{self.name}' 역할의 AI 협상 에이전트입니다.\n"
            f"역할: {self.role.value}\n"
            f"우선순위: {', '.join(self.priorities)}\n"
            f"관점: {self.persona}\n"
            f"이 관점에서 제안하고 협상하세요. 다른 에이전트와 다른 의견을 가질 수 있습니다."
        )


@dataclass
class Proposal:
    """협상 제안"""
    proposal_id: str
    agent_id: str
    content: str                        # 제안 내용
    rationale: str                      # 근거
    concessions: List[str]              # 양보 가능한 것들
    dealbreakers: List[str]             # 절대 양보 불가
    score: float = 0.0                  # 다른 에이전트들의 평가 점수
    round_number: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "proposal_id": self.proposal_id,
            "agent_id": self.agent_id,
            "content": self.content[:300],
            "rationale": self.rationale[:200],
            "concessions": self.concessions,
            "dealbreakers": self.dealbreakers,
            "score": self.score,
            "round": self.round_number,
        }


@dataclass
class NegotiationSession:
    """협상 세션"""
    session_id: str
    topic: str
    protocol: NegotiationProtocol
    agents: List[NegotiationAgent]
    proposals: List[Proposal] = field(default_factory=list)
    final_agreement: str = ""
    consensus_reached: bool = False
    rounds_completed: int = 0
    dissenting_views: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "protocol": self.protocol.value,
            "agent_count": self.agent_count,
            "rounds_completed": self.rounds_completed,
            "proposals_made": len(self.proposals),
            "consensus_reached": self.consensus_reached,
            "final_agreement": self.final_agreement[:500],
            "dissenting_views": self.dissenting_views,
        }


# ════════════════════════════════════════════════════════════════
# 협상 엔진
# ════════════════════════════════════════════════════════════════

class NegotiationEngine:
    """
    JARVIS 에이전트 협상 엔진
    여러 AI 에이전트가 협상을 통해 최적 결정을 도출한다
    """

    # 기본 에이전트 템플릿
    DEFAULT_AGENTS = [
        {
            "name": "성능 최적화 에이전트",
            "role": AgentRole.OPTIMIZER,
            "priorities": ["효율성", "속도", "정확성"],
            "persona": "최고의 성능과 효율성을 추구합니다. 벤치마크와 측정 가능한 결과를 중시합니다.",
        },
        {
            "name": "리스크 분석 에이전트",
            "role": AgentRole.CRITIC,
            "priorities": ["안전성", "신뢰성", "위험 최소화"],
            "persona": "잠재적 위험과 부작용을 분석합니다. 최악의 시나리오를 항상 고려합니다.",
        },
        {
            "name": "실용주의 에이전트",
            "role": AgentRole.PRAGMATIST,
            "priorities": ["실현 가능성", "비용 효율", "현실성"],
            "persona": "이상보다 현실을 중시합니다. 주어진 자원과 제약 내에서 최선을 찾습니다.",
        },
        {
            "name": "혁신 에이전트",
            "role": AgentRole.INNOVATOR,
            "priorities": ["창의성", "혁신", "장기적 가치"],
            "persona": "기존 방식에 얽매이지 않는 창의적 대안을 제시합니다. 미래 가능성을 봅니다.",
        },
        {
            "name": "윤리 에이전트",
            "role": AgentRole.ETHICIST,
            "priorities": ["공정성", "윤리성", "사회적 영향"],
            "persona": "결정이 모든 이해관계자에게 공정한지 판단합니다. 장기적 사회 영향을 고려합니다.",
        },
    ]

    def __init__(self, llm_manager, event_callback: Optional[Callable] = None):
        self.llm = llm_manager
        self._event_cb = event_callback
        self._sessions: Dict[str, NegotiationSession] = {}
        self._history: List[NegotiationSession] = []
        self._lock = threading.Lock()
        self._stats = {
            "sessions_run": 0,
            "consensus_reached": 0,
            "proposals_made": 0,
            "avg_rounds": 0.0,
        }
        logger.info("NegotiationEngine initialized — 5 default agents ready")

    # ── 핵심 API ─────────────────────────────────────────────────

    def negotiate(self, topic: str,
                  protocol: str = "consensus",
                  max_rounds: int = 3,
                  custom_agents: List[Dict] = None) -> NegotiationSession:
        """
        협상 실행
        topic: 협상할 주제/결정
        protocol: "consensus" | "debate" | "voting" | "mediation"
        max_rounds: 최대 협상 라운드
        """
        protocol_map = {
            "debate": NegotiationProtocol.DEBATE,
            "auction": NegotiationProtocol.AUCTION,
            "consensus": NegotiationProtocol.CONSENSUS,
            "voting": NegotiationProtocol.VOTING,
            "mediation": NegotiationProtocol.MEDIATION,
        }
        proto = protocol_map.get(protocol, NegotiationProtocol.CONSENSUS)

        # 에이전트 생성
        agent_configs = custom_agents or self.DEFAULT_AGENTS
        agents = self._create_agents(agent_configs)

        session = NegotiationSession(
            session_id=str(uuid.uuid4())[:10],
            topic=topic,
            protocol=proto,
            agents=agents,
        )

        with self._lock:
            self._sessions[session.session_id] = session

        self._emit("negotiation_started", {
            "session_id": session.session_id,
            "topic": topic,
            "protocol": protocol,
            "agents": [a.name for a in agents],
        })

        # 프로토콜별 실행
        handler = {
            NegotiationProtocol.DEBATE: self._run_debate,
            NegotiationProtocol.CONSENSUS: self._run_consensus,
            NegotiationProtocol.VOTING: self._run_voting,
            NegotiationProtocol.MEDIATION: self._run_mediation,
            NegotiationProtocol.AUCTION: self._run_auction,
        }.get(proto, self._run_consensus)

        handler(session, max_rounds)
        session.completed_at = time.time()

        with self._lock:
            self._sessions.pop(session.session_id, None)
            self._history.append(session)
            self._stats["sessions_run"] += 1
            if session.consensus_reached:
                self._stats["consensus_reached"] += 1
            self._stats["proposals_made"] += len(session.proposals)
            total_rounds = sum(s.rounds_completed for s in self._history)
            self._stats["avg_rounds"] = total_rounds / len(self._history) if self._history else 0

        self._emit("negotiation_completed", {
            "session_id": session.session_id,
            "consensus": session.consensus_reached,
            "rounds": session.rounds_completed,
            "agreement": session.final_agreement[:200],
        })
        logger.info(f"Negotiation '{topic[:40]}': {'합의' if session.consensus_reached else '미합의'} "
                    f"({session.rounds_completed} rounds, {len(session.proposals)} proposals)")
        return session

    # ── 협상 프로토콜 구현 ────────────────────────────────────────

    def _run_debate(self, session: NegotiationSession, max_rounds: int):
        """토론 형식: 각 에이전트가 입장을 제시하고 상대방을 비판"""
        for round_num in range(1, max_rounds + 1):
            session.rounds_completed = round_num
            round_proposals = []

            for agent in session.agents:
                proposal = self._get_agent_proposal(agent, session, round_num)
                session.proposals.append(proposal)
                round_proposals.append(proposal)
                with self._lock:
                    pass

            self._emit("round_completed", {
                "session_id": session.session_id,
                "round": round_num,
                "proposals": [p.to_dict() for p in round_proposals],
            })

        # 토론 결론 — 심판자 역할로 최종 판결
        session.final_agreement = self._synthesize_debate(session)
        session.consensus_reached = True  # 토론은 항상 결론 도출

    def _run_consensus(self, session: NegotiationSession, max_rounds: int):
        """합의 형식: 모든 에이전트가 동의하는 안 탐색"""
        for round_num in range(1, max_rounds + 1):
            session.rounds_completed = round_num

            # 라운드 1: 초기 제안
            if round_num == 1:
                for agent in session.agents:
                    proposal = self._get_agent_proposal(agent, session, round_num)
                    session.proposals.append(proposal)

            # 라운드 2+: 이전 제안들을 보고 수정
            else:
                prev_proposals = [p for p in session.proposals if p.round_number == round_num - 1]
                best_prev = self._select_best_proposals(prev_proposals, session)

                for agent in session.agents:
                    proposal = self._get_revised_proposal(agent, session, round_num, best_prev)
                    session.proposals.append(proposal)

            # 합의 확인
            latest = [p for p in session.proposals if p.round_number == round_num]
            if self._check_consensus(latest, session.topic):
                session.consensus_reached = True
                break

        session.final_agreement = self._synthesize_consensus(session)

    def _run_voting(self, session: NegotiationSession, max_rounds: int):
        """투표 형식: 각 에이전트가 제안 후 투표"""
        # 1단계: 각 에이전트 제안
        for agent in session.agents:
            proposal = self._get_agent_proposal(agent, session, 1)
            session.proposals.append(proposal)

        # 2단계: 모든 에이전트가 모든 제안에 투표
        votes: Dict[str, float] = {}
        for proposal in session.proposals:
            total_vote = 0.0
            for voter in session.agents:
                if voter.agent_id != proposal.agent_id:
                    vote = self._get_vote(voter, proposal, session.topic)
                    total_vote += vote * voter.weight
            votes[proposal.proposal_id] = total_vote
            proposal.score = total_vote

        # 최다 득표 제안 선택
        if votes:
            best_id = max(votes, key=votes.get)
            best_proposal = next((p for p in session.proposals if p.proposal_id == best_id), None)
            if best_proposal:
                session.final_agreement = (
                    f"[투표 결과] {best_proposal.content}\n"
                    f"(총점: {best_proposal.score:.1f})\n"
                    f"근거: {best_proposal.rationale}"
                )
                session.consensus_reached = True

        session.rounds_completed = 2

    def _run_mediation(self, session: NegotiationSession, max_rounds: int):
        """중재 형식: 중재자가 각 에이전트 의견을 조율"""
        # 1단계: 각 에이전트 입장 수집
        for agent in session.agents:
            proposal = self._get_agent_proposal(agent, session, 1)
            session.proposals.append(proposal)

        # 2단계: 중재자(LLM)가 조율
        proposals_text = "\n\n".join([
            f"[{a.name}]:\n제안: {p.content}\n근거: {p.rationale}\n양보: {', '.join(p.concessions)}"
            for a in session.agents
            for p in session.proposals if p.agent_id == a.agent_id
        ])

        mediation_prompt = f"""당신은 공정한 중재자입니다.
다음 에이전트들의 입장을 조율하여 모든 측이 받아들일 수 있는 합의안을 도출하세요.

주제: {session.topic}

각 에이전트의 입장:
{proposals_text}

중재 결과:
1. 공통 관심사 (모두가 원하는 것)
2. 양보 가능한 것들
3. 최종 합의안 (구체적이고 실행 가능)
4. 소수 의견 (합의에 동의하지 않는 관점)

한국어로 작성하세요."""

        try:
            mediation = self.llm.generate(mediation_prompt, max_tokens=600)
            if isinstance(mediation, dict):
                mediation = mediation.get("content", "")
            session.final_agreement = mediation
            session.consensus_reached = True
        except Exception as e:
            session.final_agreement = f"중재 실패: {e}"

        session.rounds_completed = 1

    def _run_auction(self, session: NegotiationSession, max_rounds: int):
        """경매 형식: 각 에이전트가 제안에 '입찰' (가중치 베팅)"""
        for agent in session.agents:
            proposal = self._get_agent_proposal(agent, session, 1)
            session.proposals.append(proposal)

        # 각 에이전트가 최선의 제안에 가중치 투자
        bid_totals: Dict[str, float] = {}
        for proposal in session.proposals:
            bid_totals[proposal.proposal_id] = 0.0

        for bidder in session.agents:
            # 각 제안의 가치를 평가
            for proposal in session.proposals:
                if proposal.agent_id == bidder.agent_id:
                    continue
                value = self._get_vote(bidder, proposal, session.topic)
                bid_totals[proposal.proposal_id] += value * bidder.weight

        if bid_totals:
            winner_id = max(bid_totals, key=bid_totals.get)
            winner_proposal = next((p for p in session.proposals if p.proposal_id == winner_id), None)
            if winner_proposal:
                session.final_agreement = (
                    f"[경매 낙찰 제안] 총 입찰: {bid_totals[winner_id]:.1f}\n\n"
                    f"{winner_proposal.content}\n\n"
                    f"선택 이유: {winner_proposal.rationale}"
                )
                session.consensus_reached = True

        session.rounds_completed = 1

    # ── 에이전트 행동 ────────────────────────────────────────────

    def _get_agent_proposal(self, agent: NegotiationAgent,
                             session: NegotiationSession,
                             round_num: int) -> Proposal:
        """에이전트가 제안 생성"""
        prev_proposals_text = ""
        if session.proposals:
            prev = session.proposals[-3:]
            prev_proposals_text = "\n".join([
                f"- {p.content[:100]}"
                for p in prev
            ])

        prompt = f"""{agent.get_system_prompt()}

협상 주제: {session.topic}
라운드: {round_num}

{"이전 제안들:\n" + prev_proposals_text if prev_proposals_text else ""}

당신의 입장에서 최선의 제안을 작성하세요.
JSON으로만 응답:
{{
  "content": "제안 내용 (구체적이고 실행 가능)",
  "rationale": "왜 이 제안이 최선인지 근거",
  "concessions": ["양보 가능한 것1", "양보 가능한 것2"],
  "dealbreakers": ["절대 양보 불가한 것"]
}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=400)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            data = self._parse_json(response)
        except Exception:
            data = {
                "content": f"{agent.role.value} 관점의 최적 접근",
                "rationale": f"{', '.join(agent.priorities)} 기반",
                "concessions": [],
                "dealbreakers": [],
            }

        return Proposal(
            proposal_id=str(uuid.uuid4())[:8],
            agent_id=agent.agent_id,
            content=data.get("content", ""),
            rationale=data.get("rationale", ""),
            concessions=data.get("concessions", []),
            dealbreakers=data.get("dealbreakers", []),
            round_number=round_num,
        )

    def _get_revised_proposal(self, agent: NegotiationAgent,
                               session: NegotiationSession,
                               round_num: int,
                               best_proposals: List[Proposal]) -> Proposal:
        """이전 제안을 보고 수정된 제안 생성"""
        best_text = "\n".join([f"- {p.content[:100]}" for p in best_proposals[:3]])

        prompt = f"""{agent.get_system_prompt()}

주제: {session.topic}
라운드 {round_num} — 합의를 위해 제안을 조정하세요.

현재 유력한 제안들:
{best_text}

당신의 핵심 가치를 지키면서 합의에 가까워지도록 제안을 수정하세요.
JSON으로만 응답:
{{
  "content": "수정된 제안",
  "rationale": "조정 이유",
  "concessions": ["이번 라운드에 새로 양보하는 것"],
  "dealbreakers": ["여전히 양보 불가"]
}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=300)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            data = self._parse_json(response)
        except Exception:
            data = {"content": "이전 제안 유지", "rationale": "", "concessions": [], "dealbreakers": []}

        return Proposal(
            proposal_id=str(uuid.uuid4())[:8],
            agent_id=agent.agent_id,
            content=data.get("content", ""),
            rationale=data.get("rationale", ""),
            concessions=data.get("concessions", []),
            dealbreakers=data.get("dealbreakers", []),
            round_number=round_num,
        )

    def _get_vote(self, voter: NegotiationAgent, proposal: Proposal, topic: str) -> float:
        """에이전트가 제안에 점수 부여 (0~10)"""
        prompt = f"""{voter.get_system_prompt()}

주제: {topic}
다음 제안을 당신의 우선순위 관점에서 평가하세요 (0~10점):

제안: {proposal.content}

숫자만 응답 (예: 7.5):"""

        try:
            response = self.llm.generate(prompt, max_tokens=10)
            if isinstance(response, dict):
                response = response.get("content", "5")
            score = float(re.search(r"\d+\.?\d*", str(response)).group())
            return min(10.0, max(0.0, score))
        except Exception:
            return 5.0

    # ── 합성 유틸리티 ────────────────────────────────────────────

    def _check_consensus(self, proposals: List[Proposal], topic: str) -> bool:
        """제안들이 합의에 충분히 수렴했는지 확인"""
        if len(proposals) < 2:
            return False
        contents = [p.content for p in proposals]
        check_prompt = f"""다음 제안들이 합의에 충분히 수렴했나요? (예/아니오만 응답)

주제: {topic}
제안들:
{chr(10).join(contents[:4])}"""
        try:
            response = self.llm.generate(check_prompt, max_tokens=5)
            if isinstance(response, dict):
                response = response.get("content", "아니오")
            return "예" in str(response) or "yes" in str(response).lower()
        except Exception:
            return False

    def _select_best_proposals(self, proposals: List[Proposal],
                                session: NegotiationSession) -> List[Proposal]:
        """가장 높은 지지를 받는 제안들 선택"""
        if not proposals:
            return []
        return sorted(proposals, key=lambda p: p.score, reverse=True)[:3]

    def _synthesize_debate(self, session: NegotiationSession) -> str:
        """토론 결론 종합"""
        all_positions = "\n".join([
            f"[{a.name}]: {next((p.content for p in session.proposals if p.agent_id == a.agent_id), '입장 없음')[:150]}"
            for a in session.agents
        ])
        prompt = f"""다음 토론의 핵심 논점과 최적 결론을 도출하세요.

주제: {session.topic}

각 에이전트 입장:
{all_positions}

결론:
1. 핵심 합의점
2. 주요 쟁점
3. 최종 권장 결정 및 근거
4. 소수 의견

한국어로 간결하게 작성하세요."""
        try:
            result = self.llm.generate(prompt, max_tokens=500)
            return result.get("content", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"토론 종합 실패: {e}"

    def _synthesize_consensus(self, session: NegotiationSession) -> str:
        """합의 결과 최종 정리"""
        latest_round = session.rounds_completed
        final_proposals = [p for p in session.proposals if p.round_number == latest_round]
        proposals_text = "\n".join([
            f"[{p.agent_id[:6]}]: {p.content[:150]}"
            for p in final_proposals
        ])
        prompt = f"""다음 협상의 최종 합의안을 정리하세요.

주제: {session.topic}
최종 라운드 제안:
{proposals_text}

공통점을 찾아 명확한 합의안을 작성하세요. (3-5문장)"""
        try:
            result = self.llm.generate(prompt, max_tokens=400)
            return result.get("content", "") if isinstance(result, dict) else str(result)
        except Exception:
            return "합의 내용을 정리하는 중 오류 발생"

    # ── 에이전트 팩토리 ──────────────────────────────────────────

    def _create_agents(self, configs: List[Dict]) -> List[NegotiationAgent]:
        agents = []
        for config in configs:
            agent = NegotiationAgent(
                agent_id=str(uuid.uuid4())[:8],
                name=config.get("name", "에이전트"),
                role=config.get("role", AgentRole.ANALYST),
                priorities=config.get("priorities", ["균형"]),
                persona=config.get("persona", "중립적 관점"),
                weight=config.get("weight", 1.0),
            )
            agents.append(agent)
        return agents

    def _parse_json(self, text: str) -> Dict:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _emit(self, event_type: str, data: Dict):
        if self._event_cb:
            try:
                self._event_cb({
                    "type": f"negotiation_{event_type}",
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    # ── 통계 ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                "active_sessions": len(self._sessions),
                "history_count": len(self._history),
            }

    def get_history(self, n: int = 10) -> List[Dict]:
        return [s.to_dict() for s in self._history[-n:]]
