"""
JARVIS Tree of Thoughts (ToT) 추론 엔진 — Iteration 5
인간의 직선적 사고를 뛰어넘는 다차원 추론 시스템

알고리즘:
- 각 단계에서 K개의 후보 사고 생성 (너비)
- 가치 함수로 각 사고 평가
- Beam Search로 최적 경로 유지
- 최종: 전체 추론 트리에서 최선의 답변 추출

인간 대비 우위:
- 동시에 수십 개의 사고 경로 탐색
- 각 경로의 품질을 객관적으로 평가
- 막다른 경로는 자동으로 가지치기
- 최종 답변의 신뢰도 수치화
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """단일 사고 노드"""
    id: str
    content: str          # 사고 내용
    score: float          # 가치 점수 (0-1)
    depth: int            # 트리 깊이
    parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    is_solution: bool = False
    reasoning: str = ""   # 이 사고를 선택한 이유


@dataclass
class ThoughtTree:
    """사고 트리 전체"""
    problem: str
    root_id: str
    nodes: Dict[str, ThoughtNode]
    best_path: List[str]        # 최적 경로의 노드 ID 순서
    final_answer: str
    confidence: float
    total_thoughts: int
    pruned_branches: int
    duration: float
    strategy: str               # bfs / dfs / beam
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_path_contents(self) -> List[str]:
        return [self.nodes[nid].content for nid in self.best_path if nid in self.nodes]


class TreeOfThoughts:
    """
    JARVIS Tree of Thoughts 추론 엔진
    복잡한 문제를 다차원 사고 트리로 탐색하여 최선의 답변 도출
    """

    # 각 단계에서 생성할 후보 사고 수
    DEFAULT_BRANCHING = 3
    # 유지할 최선 경로 수 (빔 너비)
    DEFAULT_BEAM_WIDTH = 2
    # 최대 트리 깊이
    DEFAULT_MAX_DEPTH = 4

    THOUGHT_GEN_PROMPT = """문제: {problem}

현재까지의 추론:
{previous_thoughts}

다음 단계의 추론을 {k}개 제시하세요. 각각 다른 관점/접근법이어야 합니다.

JSON 배열로 반환:
[
  {{
    "thought": "추론 내용 (구체적이고 논리적으로)",
    "approach": "이 접근법의 특징",
    "leads_to": "이 추론이 이어질 방향"
  }}
]

JSON만 출력하세요."""

    EVALUATE_PROMPT = """문제: {problem}

추론 경로:
{thought_path}

이 추론 경로를 다음 기준으로 평가하세요:
1. 논리적 일관성 (0-10)
2. 문제 해결 가능성 (0-10)
3. 근거의 충분성 (0-10)
4. 완성도 (0-10)

JSON으로 반환:
{{
  "logic_score": 8,
  "solvability": 7,
  "evidence_score": 6,
  "completeness": 8,
  "total_score": 7.25,
  "verdict": "이 경로의 장단점 한 줄 평가"
}}

JSON만 출력하세요."""

    SOLUTION_PROMPT = """문제: {problem}

최적 추론 경로 (신뢰도 {confidence:.0%}):
{thought_path}

이 추론 경로를 바탕으로 최종 답변을 작성하세요.
답변은 완전하고 상세해야 하며, 추론 과정에서 발견한 모든 핵심 인사이트를 포함해야 합니다.

JSON으로 반환:
{{
  "answer": "최종 답변 (상세)",
  "key_insights": ["핵심 인사이트 1", "인사이트 2", "인사이트 3"],
  "caveats": ["주의사항 1", "주의사항 2"],
  "confidence": 0.85
}}

JSON만 출력하세요."""

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self._history: List[ThoughtTree] = []
        self._node_counter = 0
        logger.info("TreeOfThoughts engine initialized")

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter}"

    # ── 메인 추론 ──────────────────────────────────────────────────────────

    def think(
        self,
        problem: str,
        strategy: str = "beam",
        branching: int = DEFAULT_BRANCHING,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> ThoughtTree:
        """
        Tree of Thoughts 추론 실행

        strategy:
          - "beam": 빔 서치 (권장 — 균형)
          - "bfs": 너비 우선 (철저)
          - "greedy": 탐욕적 (빠름)
        """
        start_time = time.time()
        nodes: Dict[str, ThoughtNode] = {}
        pruned = 0

        logger.info(f"[ToT] Starting {strategy} search — problem: {problem[:80]}")

        # 루트 노드
        root_id = self._new_node_id()
        root = ThoughtNode(
            id=root_id,
            content=f"문제 이해: {problem}",
            score=0.7,
            depth=0,
            parent_id=None,
        )
        nodes[root_id] = root

        if strategy == "beam":
            best_path, final_score = self._beam_search(
                problem, nodes, root_id, branching, beam_width, max_depth
            )
        elif strategy == "bfs":
            best_path, final_score = self._bfs_search(
                problem, nodes, root_id, branching, max_depth
            )
        else:  # greedy
            best_path, final_score = self._greedy_search(
                problem, nodes, root_id, branching, max_depth
            )

        pruned = len(nodes) - len(best_path)
        thought_path_text = self._path_to_text(nodes, best_path)
        solution = self._generate_solution(problem, thought_path_text, final_score)

        tree = ThoughtTree(
            problem=problem,
            root_id=root_id,
            nodes=nodes,
            best_path=best_path,
            final_answer=solution.get("answer", "추론 실패"),
            confidence=float(solution.get("confidence", final_score)),
            total_thoughts=len(nodes),
            pruned_branches=max(0, pruned),
            duration=round(time.time() - start_time, 2),
            strategy=strategy,
        )
        self._history.append(tree)
        logger.info(f"[ToT] Done — {len(nodes)} thoughts, confidence={tree.confidence:.2f}")
        return tree

    def think_streaming(self, problem: str, branching: int = 3) -> Generator[Dict, None, None]:
        """스트리밍 ToT — UI 실시간 표시용"""
        yield {"type": "start", "problem": problem}
        nodes: Dict[str, ThoughtNode] = {}

        root_id = self._new_node_id()
        root = ThoughtNode(id=root_id, content=f"문제: {problem}", score=0.7, depth=0, parent_id=None)
        nodes[root_id] = root
        yield {"type": "root", "content": root.content}

        # Greedy 탐색 (스트리밍용)
        current_path = [root_id]
        for depth in range(1, 4):
            yield {"type": "depth", "depth": depth}
            parent_id = current_path[-1]
            parent = nodes[parent_id]
            prev_text = self._path_to_text(nodes, current_path)

            candidates = self._generate_thoughts(problem, prev_text, k=branching)
            if not candidates:
                break

            best_candidate = None
            best_score = 0.0
            for i, cand in enumerate(candidates):
                nid = self._new_node_id()
                score = self._evaluate_thought(problem, prev_text + "\n" + cand["thought"])
                node = ThoughtNode(id=nid, content=cand["thought"], score=score, depth=depth, parent_id=parent_id)
                nodes[nid] = node
                parent.children.append(nid)
                yield {"type": "thought", "id": nid, "content": cand["thought"][:200], "score": score, "depth": depth}
                if score > best_score:
                    best_score = score
                    best_candidate = nid

            if best_candidate:
                current_path.append(best_candidate)

        thought_path_text = self._path_to_text(nodes, current_path)
        solution = self._generate_solution(problem, thought_path_text, best_score)
        yield {
            "type": "done",
            "answer": solution.get("answer", ""),
            "key_insights": solution.get("key_insights", []),
            "confidence": solution.get("confidence", best_score),
            "total_thoughts": len(nodes),
            "path_length": len(current_path),
        }

    # ── 탐색 전략 ──────────────────────────────────────────────────────────

    def _beam_search(
        self,
        problem: str,
        nodes: Dict[str, ThoughtNode],
        root_id: str,
        branching: int,
        beam_width: int,
        max_depth: int,
    ) -> Tuple[List[str], float]:
        """빔 서치 — 상위 beam_width개 경로 유지"""
        # 각 빔: (score, path)
        beams: List[Tuple[float, List[str]]] = [(0.7, [root_id])]

        for depth in range(1, max_depth + 1):
            next_beams: List[Tuple[float, List[str]]] = []

            for beam_score, path in beams:
                parent_id = path[-1]
                parent = nodes[parent_id]
                prev_text = self._path_to_text(nodes, path)

                candidates = self._generate_thoughts(problem, prev_text, k=branching)
                for cand in candidates:
                    nid = self._new_node_id()
                    score = self._evaluate_thought(problem, prev_text + "\n" + cand["thought"])
                    node = ThoughtNode(
                        id=nid,
                        content=cand["thought"],
                        score=score,
                        depth=depth,
                        parent_id=parent_id,
                        reasoning=cand.get("approach", ""),
                    )
                    nodes[nid] = node
                    parent.children.append(nid)
                    new_path = path + [nid]
                    # 누적 점수 (평균)
                    path_score = (beam_score * (depth - 1) + score) / depth
                    next_beams.append((path_score, new_path))

            if not next_beams:
                break

            # 상위 beam_width개만 유지
            next_beams.sort(key=lambda x: -x[0])
            beams = next_beams[:beam_width]
            logger.debug(f"[ToT Beam] Depth {depth}: {len(beams)} beams, best={beams[0][0]:.3f}")

        if not beams:
            return [root_id], 0.5

        best_score, best_path = beams[0]
        return best_path, best_score

    def _bfs_search(
        self,
        problem: str,
        nodes: Dict[str, ThoughtNode],
        root_id: str,
        branching: int,
        max_depth: int,
    ) -> Tuple[List[str], float]:
        """너비 우선 탐색"""
        from collections import deque
        queue = deque([(root_id, [root_id], 0.7)])
        best_path = [root_id]
        best_score = 0.7

        visited_at_depth: Dict[int, List[Tuple[float, str]]] = {}

        while queue:
            current_id, path, score = queue.popleft()
            depth = len(path)

            if depth > max_depth:
                continue

            if score > best_score:
                best_score = score
                best_path = path

            prev_text = self._path_to_text(nodes, path)
            candidates = self._generate_thoughts(problem, prev_text, k=branching)

            depth_nodes = []
            for cand in candidates:
                nid = self._new_node_id()
                c_score = self._evaluate_thought(problem, prev_text + "\n" + cand["thought"])
                node = ThoughtNode(id=nid, content=cand["thought"], score=c_score, depth=depth, parent_id=current_id)
                nodes[nid] = node
                nodes[current_id].children.append(nid)
                depth_nodes.append((c_score, nid, path + [nid]))

            # 가지치기: 점수 낮은 것 제거 (BFS도 제한)
            depth_nodes.sort(key=lambda x: -x[0])
            for c_score, nid, new_path in depth_nodes[:2]:  # BFS는 상위 2개만
                queue.append((nid, new_path, c_score))

        return best_path, best_score

    def _greedy_search(
        self,
        problem: str,
        nodes: Dict[str, ThoughtNode],
        root_id: str,
        branching: int,
        max_depth: int,
    ) -> Tuple[List[str], float]:
        """탐욕적 탐색 — 매 단계 최고 점수 선택"""
        current_path = [root_id]
        current_score = 0.7

        for depth in range(1, max_depth + 1):
            parent_id = current_path[-1]
            prev_text = self._path_to_text(nodes, current_path)

            candidates = self._generate_thoughts(problem, prev_text, k=branching)
            if not candidates:
                break

            best_node_id = None
            best_node_score = 0.0
            for cand in candidates:
                nid = self._new_node_id()
                score = self._evaluate_thought(problem, prev_text + "\n" + cand["thought"])
                node = ThoughtNode(id=nid, content=cand["thought"], score=score, depth=depth, parent_id=parent_id)
                nodes[nid] = node
                nodes[parent_id].children.append(nid)
                if score > best_node_score:
                    best_node_score = score
                    best_node_id = nid

            if best_node_id and best_node_score > 0.3:
                current_path.append(best_node_id)
                current_score = (current_score + best_node_score) / 2

        return current_path, current_score

    # ── LLM 호출 ───────────────────────────────────────────────────────────

    def _generate_thoughts(self, problem: str, prev_thoughts: str, k: int) -> List[Dict]:
        """다음 사고 후보 생성"""
        from jarvis.llm.manager import Message
        prompt = self.THOUGHT_GEN_PROMPT.format(
            problem=problem, previous_thoughts=prev_thoughts or "없음", k=k
        )
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=2048)
            import re
            match = re.search(r'\[.*\]', resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.debug(f"[ToT] Generate thoughts error: {e}")
        return []

    def _evaluate_thought(self, problem: str, thought_path: str) -> float:
        """사고 경로 평가 — 0.0~1.0"""
        from jarvis.llm.manager import Message
        prompt = self.EVALUATE_PROMPT.format(problem=problem, thought_path=thought_path[-1500:])
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=512)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                total = float(data.get("total_score", 5.0))
                return min(1.0, max(0.0, total / 10.0))
        except Exception as e:
            logger.debug(f"[ToT] Evaluate error: {e}")
        return 0.5

    def _generate_solution(self, problem: str, thought_path: str, confidence: float) -> Dict:
        """최적 경로에서 최종 답변 생성"""
        from jarvis.llm.manager import Message
        prompt = self.SOLUTION_PROMPT.format(
            problem=problem, thought_path=thought_path, confidence=confidence
        )
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=4096)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"[ToT] Solution generation error: {e}")
        return {"answer": thought_path, "confidence": confidence, "key_insights": []}

    # ── 유틸리티 ───────────────────────────────────────────────────────────

    def _path_to_text(self, nodes: Dict[str, ThoughtNode], path: List[str]) -> str:
        parts = []
        for i, nid in enumerate(path):
            if nid in nodes:
                parts.append(f"[단계 {i}] {nodes[nid].content}")
        return "\n".join(parts)

    def format_tree_markdown(self, tree: ThoughtTree) -> str:
        lines = [
            "# Tree of Thoughts 추론 결과",
            f"**문제:** {tree.problem}",
            f"**전략:** {tree.strategy} | **총 사고:** {tree.total_thoughts}개 | "
            f"**신뢰도:** {tree.confidence:.0%} | **소요:** {tree.duration}초",
            "",
            "## 최적 추론 경로",
        ]
        for i, nid in enumerate(tree.best_path):
            node = tree.nodes.get(nid)
            if node:
                lines.append(f"\n**[단계 {i}]** (점수: {node.score:.2f})\n{node.content}")

        lines.extend(["", "## 최종 답변", tree.final_answer])
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        if not self._history:
            return {"total_runs": 0}
        avg_confidence = sum(t.confidence for t in self._history) / len(self._history)
        avg_thoughts = sum(t.total_thoughts for t in self._history) / len(self._history)
        return {
            "total_runs": len(self._history),
            "avg_confidence": round(avg_confidence, 3),
            "avg_thoughts_per_run": round(avg_thoughts, 1),
            "strategies_used": list(set(t.strategy for t in self._history)),
            "last_run": self._history[-1].timestamp if self._history else None,
        }

    def get_history(self) -> List[Dict]:
        return [
            {
                "problem": t.problem[:80],
                "strategy": t.strategy,
                "confidence": t.confidence,
                "total_thoughts": t.total_thoughts,
                "duration": t.duration,
                "timestamp": t.timestamp,
            }
            for t in reversed(self._history[-20:])
        ]
