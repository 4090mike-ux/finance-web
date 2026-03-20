"""
JARVIS 현실 시뮬레이터 — Iteration 9
멀티버스 시나리오 시뮬레이션 (What-If 엔진)

핵심 아키텍처:
  - Monte Carlo Tree Search (MCTS) 기반 다단계 계획
  - 세계 모델 (WorldState): 실체, 상태, 관계 추적
  - 분기 시뮬레이션: 현재 상태 + 행동 → 미래 상태 예측
  - 확률 트리: 각 분기의 확률과 결과 점수화
  - 최대 5단계 선행 시뮬레이션
  - 병렬 분기 탐색 (threading)
  - LLM 기반 세계 모델 생성 및 평가
"""

import json
import math
import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """세계 내 실체 (사람, 사물, 개념 등)"""
    name: str
    entity_type: str = "object"    # person, object, concept, system
    properties: Dict = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "relationships": self.relationships,
        }


@dataclass
class WorldState:
    """특정 시점의 세계 상태 스냅샷"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    description: str = ""

    def state_hash(self) -> str:
        """상태 해시 (캐시 키)"""
        content = json.dumps(self.facts[:5], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "facts": self.facts[:10],
            "constraints": self.constraints[:5],
            "step": self.step,
            "description": self.description,
        }


@dataclass
class SimulationBranch:
    """하나의 시뮬레이션 분기 (What-If 경로)"""
    branch_id: str
    action: str                      # 취한 행동
    initial_state: WorldState = field(default_factory=WorldState)
    final_state: WorldState = field(default_factory=WorldState)
    probability: float = 0.5        # 이 분기의 발생 확률
    confidence: float = 0.5         # LLM 평가 신뢰도
    outcome_score: float = 0.0      # 결과 품질 (-1 ~ +1)
    reasoning: str = ""             # 추론 근거
    sub_branches: List = field(default_factory=list)  # 하위 분기
    depth: int = 0
    elapsed_time: float = 0.0

    def __post_init__(self):
        if not self.branch_id:
            self.branch_id = hashlib.md5(
                f"{self.action}{time.time()}".encode()
            ).hexdigest()[:8]

    def to_dict(self) -> Dict:
        return {
            "branch_id": self.branch_id,
            "action": self.action,
            "probability": round(self.probability, 3),
            "confidence": round(self.confidence, 3),
            "outcome_score": round(self.outcome_score, 3),
            "reasoning": self.reasoning[:500],
            "depth": self.depth,
            "initial_description": self.initial_state.description[:200],
            "final_description": self.final_state.description[:300],
            "sub_branch_count": len(self.sub_branches),
            "elapsed_time": round(self.elapsed_time, 2),
        }


# ════════════════════════════════════════════════════════════════
# MCTS 노드
# ════════════════════════════════════════════════════════════════

class MCTSNode:
    """Monte Carlo Tree Search 노드"""

    def __init__(self, state: WorldState, action: str = "", parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.is_expanded: bool = False

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """UCB1 점수 계산"""
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value / self.visits) + exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def best_child(self) -> Optional['MCTSNode']:
        """가장 높은 UCB1 점수의 자식 노드"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb1())

    def most_visited_child(self) -> Optional['MCTSNode']:
        """가장 많이 방문된 자식 (최종 행동 선택용)"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


# ════════════════════════════════════════════════════════════════
# 메인 현실 시뮬레이터
# ════════════════════════════════════════════════════════════════

class RealitySimulator:
    """
    JARVIS 현실 시뮬레이터 — Iteration 9

    What-If 엔진:
    - 질문 → 세계 모델 생성 → MCTS 분기 탐색 → 결과 예측
    - 최대 5단계 선행, 최대 10개 분기
    - 각 분기에 확률 + 신뢰도 + 결과 점수 부여
    - 캐시로 중복 LLM 호출 방지
    """

    MAX_DEPTH = 5
    MAX_BRANCHES = 5
    MCTS_ITERATIONS = 50
    CACHE_TTL = 300.0  # 5분 캐시

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        max_depth: int = None,
    ):
        self.llm = llm_manager
        self.event_callback = event_callback
        self.max_depth = max_depth or self.MAX_DEPTH
        self._lock = threading.RLock()

        # 세계 상태 캐시 {hash: (WorldState, timestamp)}
        self._state_cache: Dict[str, Tuple[WorldState, float]] = {}

        # 통계
        self._total_simulations = 0
        self._total_branches_explored = 0

        logger.info("RealitySimulator initialized — MCTS multiverse ready")

    def simulate(
        self,
        question: str,
        context: str = "",
        depth: int = 3,
    ) -> List[SimulationBranch]:
        """
        메인 시뮬레이션 함수

        질문과 컨텍스트를 받아 다중 미래 분기를 시뮬레이션하고 반환

        Returns:
            list of SimulationBranch, 확률 내림차순 정렬
        """
        depth = min(depth, self.MAX_DEPTH)
        start_time = time.time()

        self._emit_event("simulation_start", {
            "question": question[:200],
            "depth": depth,
        })

        try:
            # 1. 세계 모델 생성
            world_state = self.create_world_model(question, context)

            # 2. 가능한 행동(시나리오) 생성
            actions = self._generate_actions(question, world_state)
            if not actions:
                actions = ["현재 상태 유지", "점진적 변화", "급격한 전환"]

            # 3. 각 행동별 분기 시뮬레이션 (병렬)
            branches: List[SimulationBranch] = []
            branch_lock = threading.Lock()
            threads = []

            for i, action in enumerate(actions[:self.MAX_BRANCHES]):
                def sim_branch(a=action, idx=i):
                    branch = self._simulate_branch(world_state, a, depth=min(depth, 3))
                    with branch_lock:
                        branches.append(branch)

                t = threading.Thread(target=sim_branch, daemon=True)
                threads.append(t)
                t.start()

            # 모든 분기 완료 대기 (최대 30초)
            for t in threads:
                t.join(timeout=30.0)

            # 4. MCTS로 최적 행동 보강
            if self.llm and world_state.facts:
                try:
                    best_action = self.mcts_search(world_state, iterations=self.MCTS_ITERATIONS)
                    if best_action and best_action not in actions:
                        mcts_branch = self._simulate_branch(world_state, best_action, depth=2)
                        mcts_branch.confidence *= 1.2  # MCTS 선택 부스트
                        branches.append(mcts_branch)
                except Exception as e:
                    logger.debug(f"MCTS failed: {e}")

            # 5. 정규화 및 정렬
            branches = self._normalize_probabilities(branches)
            branches.sort(key=lambda b: b.probability * b.confidence, reverse=True)

            elapsed = time.time() - start_time
            self._total_simulations += 1
            self._total_branches_explored += len(branches)

            self._emit_event("simulation_done", {
                "branches": len(branches),
                "elapsed": round(elapsed, 2),
                "best_outcome": branches[0].action if branches else "none",
            })

            logger.info(f"RealitySimulator: {len(branches)} branches in {elapsed:.1f}s")
            return branches

        except Exception as e:
            logger.warning(f"RealitySimulator.simulate failed: {e}")
            return self._fallback_simulate(question, depth)

    def create_world_model(self, question: str, context: str = "") -> WorldState:
        """
        질문과 컨텍스트로부터 세계 모델 생성

        LLM이 있으면 상세 모델, 없으면 키워드 기반 간단 모델
        """
        cache_key = hashlib.md5(f"{question}{context}".encode()).hexdigest()[:12]

        # 캐시 확인
        with self._lock:
            if cache_key in self._state_cache:
                cached_state, cached_time = self._state_cache[cache_key]
                if time.time() - cached_time < self.CACHE_TTL:
                    return cached_state

        world_state = WorldState(description=f"초기 상태: {question[:100]}")

        if self.llm:
            try:
                world_state = self._llm_create_world_model(question, context)
            except Exception as e:
                logger.debug(f"LLM world model creation failed: {e}")
                world_state = self._keyword_world_model(question)
        else:
            world_state = self._keyword_world_model(question)

        # 캐시 저장
        with self._lock:
            self._state_cache[cache_key] = (world_state, time.time())

        return world_state

    def _llm_create_world_model(self, question: str, context: str) -> WorldState:
        """LLM 기반 세계 모델 생성"""
        prompt = f"""다음 질문에 대한 현재 세계 상태를 JSON으로 모델링하세요.

질문: "{question}"
컨텍스트: "{context[:300] if context else '없음'}"

JSON 형식:
{{
  "entities": [
    {{"name": "실체명", "type": "person/object/concept/system", "properties": {{"key": "value"}}}}
  ],
  "facts": ["현재 상태 팩트 1", "팩트 2", "팩트 3"],
  "constraints": ["제약 조건 1", "제약 조건 2"],
  "description": "현재 상태 요약"
}}

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=600, temperature=0.4)

        if response:
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                state = WorldState(
                    description=data.get("description", ""),
                    facts=data.get("facts", [])[:10],
                    constraints=data.get("constraints", [])[:5],
                )
                for e_data in data.get("entities", [])[:8]:
                    entity = Entity(
                        name=e_data.get("name", "unknown"),
                        entity_type=e_data.get("type", "object"),
                        properties=e_data.get("properties", {}),
                    )
                    state.entities[entity.name] = entity
                return state

        return self._keyword_world_model(question)

    def _keyword_world_model(self, question: str) -> WorldState:
        """키워드 기반 간단 세계 모델 (폴백)"""
        words = question.split()[:20]
        facts = [f"현재 상황: {question[:150]}"]
        entities = {}

        # 간단한 엔티티 추출 (대문자 단어)
        for word in words:
            if len(word) > 3 and word[0].isupper():
                entities[word] = Entity(name=word, entity_type="concept")

        return WorldState(
            entities=entities,
            facts=facts,
            description=f"기본 모델: {question[:100]}",
        )

    def _generate_actions(self, question: str, world_state: WorldState) -> List[str]:
        """가능한 행동/시나리오 목록 생성"""
        if self.llm:
            try:
                return self._llm_generate_actions(question, world_state)
            except Exception:
                pass

        # 기본 시나리오
        return [
            "현재 상태 유지 (변화 없음)",
            "긍정적 개입",
            "부정적 개입",
            "점진적 변화",
            "급격한 변화",
        ]

    def _llm_generate_actions(self, question: str, world_state: WorldState) -> List[str]:
        """LLM 기반 행동 생성"""
        facts_str = '\n'.join(world_state.facts[:5])
        prompt = f"""다음 상황에서 가능한 5가지 시나리오/행동을 간결하게 나열하세요.

질문: "{question[:200]}"
현재 상태:
{facts_str}

JSON 배열로 응답:
["시나리오1", "시나리오2", "시나리오3", "시나리오4", "시나리오5"]

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=300, temperature=0.6)
        if response:
            import re
            match = re.search(r'\[[\s\S]*?\]', response)
            if match:
                actions = json.loads(match.group())
                return [str(a) for a in actions[:5]]

        return []

    def _simulate_branch(
        self,
        initial_state: WorldState,
        action: str,
        depth: int = 2,
    ) -> SimulationBranch:
        """단일 분기 시뮬레이션"""
        start_time = time.time()
        branch_id = hashlib.md5(f"{action}{initial_state.state_hash()}{time.time()}".encode()).hexdigest()[:8]

        branch = SimulationBranch(
            branch_id=branch_id,
            action=action,
            initial_state=initial_state,
            depth=depth,
        )

        try:
            final_state = self.branch_expand(initial_state, action)
            branch.final_state = final_state
            branch.confidence = self.evaluate_branch(branch)
            branch.probability = self._estimate_probability(action, initial_state)
            branch.outcome_score = self._score_outcome(final_state)
            branch.reasoning = self._get_reasoning(action, final_state)

        except Exception as e:
            branch.reasoning = f"시뮬레이션 오류: {str(e)[:100]}"
            branch.probability = 0.1
            branch.confidence = 0.2

        branch.elapsed_time = time.time() - start_time
        return branch

    def branch_expand(self, state: WorldState, action: str) -> WorldState:
        """
        현재 상태 + 행동 → 미래 상태 예측

        LLM으로 결과 상태 생성
        """
        if self.llm:
            try:
                return self._llm_branch_expand(state, action)
            except Exception as e:
                logger.debug(f"LLM branch expand failed: {e}")

        # 폴백: 단순 상태 전환
        new_state = WorldState(
            entities=dict(state.entities),
            facts=[f"[{action} 결과] " + fact for fact in state.facts[:3]],
            constraints=state.constraints,
            step=state.step + 1,
            description=f"{action} → 결과 상태",
        )
        return new_state

    def _llm_branch_expand(self, state: WorldState, action: str) -> WorldState:
        """LLM 기반 분기 확장"""
        facts_str = '\n'.join(state.facts[:5])
        prompt = f"""행동을 취했을 때의 결과 상태를 예측하세요.

현재 상태:
{facts_str or state.description}

행동: {action}

JSON 형식으로 결과 상태:
{{
  "facts": ["결과 팩트 1", "결과 팩트 2", "결과 팩트 3"],
  "description": "결과 상태 요약 (2-3문장)"
}}

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=400, temperature=0.5)
        if response:
            import re
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                return WorldState(
                    entities=dict(state.entities),
                    facts=data.get("facts", [])[:8],
                    constraints=state.constraints,
                    step=state.step + 1,
                    description=data.get("description", f"{action} 결과"),
                )

        return WorldState(
            description=f"{action} → 결과 (LLM 파싱 실패)",
            step=state.step + 1,
        )

    def evaluate_branch(self, branch: SimulationBranch) -> float:
        """분기의 신뢰도 평가 (0.0 ~ 1.0)"""
        if self.llm:
            try:
                return self._llm_evaluate_branch(branch)
            except Exception:
                pass

        # 폴백: 휴리스틱 평가
        score = 0.5
        if branch.final_state.facts:
            score += 0.1 * min(len(branch.final_state.facts), 3)
        if branch.final_state.description:
            score += 0.1
        return min(score, 1.0)

    def _llm_evaluate_branch(self, branch: SimulationBranch) -> float:
        """LLM 기반 분기 신뢰도 평가"""
        prompt = f"""다음 시나리오의 신뢰도를 0.0~1.0으로 평가하세요.

행동: {branch.action}
초기 상태: {branch.initial_state.description[:100]}
예측 결과: {branch.final_state.description[:200]}

JSON: {{"confidence": 0.0~1.0, "reason": "이유"}}
JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
        if response:
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return float(data.get("confidence", 0.5))

        return 0.5

    def _estimate_probability(self, action: str, state: WorldState) -> float:
        """행동의 발생 확률 추정 (0.0 ~ 1.0)"""
        # 키워드 기반 간단 추정
        action_lower = action.lower()
        if "유지" in action_lower or "status quo" in action_lower:
            return 0.4  # 현상 유지는 비교적 높은 확률
        if "급격" in action_lower or "갑자기" in action_lower:
            return 0.1  # 급격한 변화는 낮은 확률
        if "점진" in action_lower or "gradually" in action_lower:
            return 0.3
        return 0.2  # 기본 확률

    def _score_outcome(self, state: WorldState) -> float:
        """결과 상태의 품질 점수 (-1 ~ +1)"""
        description = state.description.lower()
        positive_signals = ["성공", "개선", "향상", "좋", "긍정", "success", "improve", "better"]
        negative_signals = ["실패", "악화", "문제", "위험", "나쁨", "fail", "worse", "problem"]

        score = 0.0
        for sig in positive_signals:
            if sig in description:
                score += 0.2
        for sig in negative_signals:
            if sig in description:
                score -= 0.2

        return max(-1.0, min(1.0, score))

    def _get_reasoning(self, action: str, final_state: WorldState) -> str:
        """행동에 대한 추론 설명"""
        return f"'{action}' 시나리오: {final_state.description[:300]}"

    def _normalize_probabilities(self, branches: List[SimulationBranch]) -> List[SimulationBranch]:
        """분기 확률 정규화"""
        total = sum(b.probability for b in branches)
        if total > 0:
            for b in branches:
                b.probability = b.probability / total
        return branches

    def get_best_outcome(self, branches: List[SimulationBranch] = None) -> Optional[SimulationBranch]:
        """가장 좋은 결과 분기 반환"""
        if not branches:
            return None
        return max(branches, key=lambda b: b.probability * b.confidence * (b.outcome_score + 1))

    def mcts_search(self, root_state: WorldState, iterations: int = 50) -> str:
        """
        Monte Carlo Tree Search로 최적 행동 탐색

        Returns: 최적 행동 문자열
        """
        root = MCTSNode(state=root_state, action="root")

        for _ in range(min(iterations, 30)):  # 성능을 위해 제한
            # 1. 선택 (Selection)
            node = self._mcts_select(root)

            # 2. 확장 (Expansion)
            if not node.is_expanded and node.visits > 0:
                self._mcts_expand(node)
                node.is_expanded = True

            # 3. 시뮬레이션 (Rollout) — 단순화
            reward = self._mcts_rollout(node)

            # 4. 역전파 (Backpropagation)
            self._mcts_backpropagate(node, reward)

        # 가장 많이 방문된 자식의 행동 반환
        best = root.most_visited_child()
        return best.action if best else "direct"

    def _mcts_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1로 선택"""
        while node.children:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            node = node.best_child() or node
        return node

    def _mcts_expand(self, node: MCTSNode):
        """노드 확장 — 자식 행동 생성"""
        import random
        actions = ["현상 유지", "점진적 개선", "급격한 변화", "역방향 접근", "외부 도움"]
        for action in actions[:3]:
            child = MCTSNode(state=node.state, action=action, parent=node)
            node.children.append(child)

    def _mcts_rollout(self, node: MCTSNode) -> float:
        """무작위 롤아웃으로 보상 추정"""
        import random
        return random.uniform(-0.5, 1.0)  # 빠른 추정

    def _mcts_backpropagate(self, node: MCTSNode, reward: float):
        """역전파"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _fallback_simulate(self, question: str, depth: int) -> List[SimulationBranch]:
        """LLM 없이 기본 시뮬레이션"""
        scenarios = [
            ("현상 유지", 0.4, 0.7, 0.0),
            ("긍정적 변화", 0.3, 0.6, 0.6),
            ("부정적 변화", 0.2, 0.6, -0.5),
            ("예측 불가 결과", 0.1, 0.3, 0.0),
        ]

        branches = []
        initial_state = WorldState(description=f"초기 상태: {question[:100]}")

        for action, prob, conf, score in scenarios:
            branch = SimulationBranch(
                branch_id=hashlib.md5(action.encode()).hexdigest()[:8],
                action=action,
                initial_state=initial_state,
                final_state=WorldState(description=f"{action} 결과"),
                probability=prob,
                confidence=conf,
                outcome_score=score,
                reasoning=f"기본 시나리오: {action}",
            )
            branches.append(branch)

        return branches

    def _emit_event(self, event_type: str, data: Dict):
        """이벤트 콜백 호출"""
        if self.event_callback:
            try:
                self.event_callback({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def get_stats(self) -> Dict:
        """통계 반환"""
        return {
            "total_simulations": self._total_simulations,
            "total_branches_explored": self._total_branches_explored,
            "cache_size": len(self._state_cache),
            "max_depth": self.max_depth,
            "mcts_iterations": self.MCTS_ITERATIONS,
        }


# 순환 임포트 방지용
import random
