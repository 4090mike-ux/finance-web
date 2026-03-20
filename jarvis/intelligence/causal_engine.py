"""
JARVIS Causal Reasoning Engine — Iteration 7
Pearl의 인과 계층 구조를 기반으로 한 인과 추론 엔진

Level 1 (Association):      P(Y|X)           — 관찰/상관
Level 2 (Intervention):     P(Y|do(X))       — 개입/실험
Level 3 (Counterfactual):   P(Y_x|X',Y')     — 반사실/상상
"""

import json
import time
import logging
import re
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CausalNode:
    id: str
    name: str
    description: str
    node_type: str = "variable"   # variable / event / state / action
    domain: str = ""


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relationship: str = "causes"   # causes / enables / inhibits / correlates
    strength: float = 0.7          # 0–1
    mechanism: str = ""
    reversible: bool = True


@dataclass
class CounterfactualResult:
    antecedent: str
    consequent: str
    answer: str
    confidence: float
    reasoning_chain: List[str]
    counterfactual_world: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterventionPlan:
    goal: str
    target_variable: str
    required_interventions: List[Dict]
    expected_outcome: str
    side_effects: List[str]
    feasibility: float = 0.5


class CausalEngine:
    """
    인과 추론 엔진
    - 인과 그래프 자동 구축
    - 반사실적 추론 (What-if 분석)
    - 개입 계획 (do-calculus)
    - 인과 경로 탐색
    """

    def __init__(self, llm_manager=None, data_dir: str = "data/jarvis"):
        self.llm = llm_manager
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.counterfactuals: List[CounterfactualResult] = []
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"CausalEngine initialized: {len(self.nodes)} nodes, {len(self.edges)} edges")

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        f = self.data_dir / "causal_graph.json"
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                for n in data.get("nodes", []):
                    nd = CausalNode(**n)
                    self.nodes[nd.id] = nd
                for e in data.get("edges", []):
                    self.edges.append(CausalEdge(**e))
            except Exception as ex:
                logger.warning(f"CausalEngine load error: {ex}")

    def _save(self):
        f = self.data_dir / "causal_graph.json"
        f.write_text(json.dumps({
            "nodes": [vars(n) for n in self.nodes.values()],
            "edges": [vars(e) for e in self.edges],
            "updated_at": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Graph construction ────────────────────────────────────────────

    def _get_client(self):
        if self.llm:
            return getattr(self.llm, "anthropic_client", None) or getattr(self.llm, "_client", None)
        return None

    def _llm_call(self, prompt: str, max_tokens: int = 1500) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            import anthropic
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def add_node(self, name: str, description: str = "",
                 node_type: str = "variable", domain: str = "") -> CausalNode:
        nid = hashlib.md5(name.lower().encode()).hexdigest()[:8]
        if nid not in self.nodes:
            self.nodes[nid] = CausalNode(id=nid, name=name,
                                          description=description or name,
                                          node_type=node_type, domain=domain)
        return self.nodes[nid]

    def add_causal_link(self, cause: str, effect: str,
                        mechanism: str = "", strength: float = 0.7,
                        relationship: str = "causes") -> CausalEdge:
        cn = self.add_node(cause, cause)
        en = self.add_node(effect, effect)
        # Deduplicate
        existing = next((e for e in self.edges
                         if e.source_id == cn.id and e.target_id == en.id), None)
        if existing:
            existing.strength = max(existing.strength, strength)
            return existing
        edge = CausalEdge(source_id=cn.id, target_id=en.id,
                          relationship=relationship, strength=strength,
                          mechanism=mechanism)
        self.edges.append(edge)
        self._save()
        return edge

    def extract_causal_relations(self, text: str, domain: str = "") -> List[Dict]:
        """텍스트에서 인과관계 자동 추출 (LLM)"""
        prompt = f"""다음 텍스트에서 인과관계를 추출하여 JSON 배열로 반환하세요.

텍스트: {text[:3000]}
도메인: {domain or "일반"}

형식 (JSON 배열만 반환):
[
  {{
    "cause": "원인",
    "effect": "결과",
    "mechanism": "인과 메커니즘",
    "strength": 0.8,
    "relationship": "causes"
  }}
]

주의: 순수 상관관계가 아닌 실제 인과관계만 포함. JSON만 반환."""

        text_out = self._llm_call(prompt, 2000)
        try:
            m = re.search(r'\[.*\]', text_out, re.DOTALL)
            relations = json.loads(m.group()) if m else []
            for r in relations:
                self.add_causal_link(r["cause"], r["effect"],
                                     r.get("mechanism", ""),
                                     float(r.get("strength", 0.7)),
                                     r.get("relationship", "causes"))
            return relations
        except Exception:
            return []

    # ── Level 3: Counterfactual ───────────────────────────────────────

    def counterfactual(self, antecedent: str, consequent: str) -> CounterfactualResult:
        """
        반사실적 추론: '만약 X가 달랐다면 Y는 어떻게 됐을까?'
        Pearl's Level 3: P(Y_x | X=x', Y=y')
        """
        graph_ctx = "\n".join(
            f"- {self.nodes.get(e.source_id, CausalNode('','?','','','')).name}"
            f" →[{e.relationship}:{e.strength:.1f}]→ "
            f"{self.nodes.get(e.target_id, CausalNode('','?','','','')).name}: {e.mechanism}"
            for e in self.edges[:30]
        )

        prompt = f"""반사실적 추론(Counterfactual Reasoning)을 수행하세요. Pearl's Level 3 적용.

가정 (Antecedent): {antecedent}
질문 (Consequent): {consequent}

알려진 인과 그래프:
{graph_ctx or "(인과 데이터 없음)"}

JSON으로 응답:
{{
  "answer": "반사실적 결과",
  "confidence": 0.8,
  "reasoning_chain": ["단계1", "단계2", "단계3"],
  "counterfactual_world": "반사실 세계의 상태 설명",
  "key_differences": ["현실과 다른 점1", "점2"]
}}"""

        text_out = self._llm_call(prompt, 1500)
        try:
            m = re.search(r'\{.*\}', text_out, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

        result = CounterfactualResult(
            antecedent=antecedent, consequent=consequent,
            answer=data.get("answer", "분석 불가"),
            confidence=float(data.get("confidence", 0.0)),
            reasoning_chain=data.get("reasoning_chain", []),
            counterfactual_world=data.get("counterfactual_world", ""),
        )
        self.counterfactuals.append(result)
        return result

    # ── Level 2: Intervention ─────────────────────────────────────────

    def plan_intervention(self, goal: str) -> InterventionPlan:
        """
        개입 계획: 목표 달성을 위해 무엇을 변경해야 하는가?
        P(Y | do(X)) — 관찰이 아닌 실제 개입
        """
        graph_ctx = "\n".join(
            f"- {self.nodes.get(e.source_id, CausalNode('','?','','','')).name}"
            f" → {self.nodes.get(e.target_id, CausalNode('','?','','','')).name}"
            f": {e.relationship} (강도 {e.strength:.1f})"
            for e in self.edges[:20]
        )

        prompt = f"""do-calculus를 적용하여 목표 달성을 위한 최적 개입 계획을 수립하세요.

목표: {goal}

알려진 인과 구조:
{graph_ctx or "(인과 데이터 없음)"}

JSON으로 응답:
{{
  "target_variable": "개입할 핵심 변수",
  "required_interventions": [
    {{"action": "행동", "mechanism": "메커니즘", "priority": 1, "effort": "low|medium|high"}}
  ],
  "expected_outcome": "예상 결과",
  "side_effects": ["부작용1", "부작용2"],
  "feasibility": 0.8
}}"""

        text_out = self._llm_call(prompt, 2000)
        try:
            m = re.search(r'\{.*\}', text_out, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

        return InterventionPlan(
            goal=goal,
            target_variable=data.get("target_variable", ""),
            required_interventions=data.get("required_interventions", []),
            expected_outcome=data.get("expected_outcome", ""),
            side_effects=data.get("side_effects", []),
            feasibility=float(data.get("feasibility", 0.5)),
        )

    # ── Level 1: Association / Path finding ──────────────────────────

    def find_causal_path(self, from_name: str, to_name: str) -> List[List[str]]:
        """두 개념 사이의 인과 경로 탐색 (BFS)"""
        from_ids = [n.id for n in self.nodes.values()
                    if from_name.lower() in n.name.lower()]
        to_ids = [n.id for n in self.nodes.values()
                  if to_name.lower() in n.name.lower()]
        if not from_ids or not to_ids:
            return []

        paths: List[List[str]] = []
        queue = [[from_ids[0]]]
        visited: set = set()

        while queue and len(paths) < 5:
            path = queue.pop(0)
            node_id = path[-1]
            if node_id in to_ids:
                paths.append([
                    self.nodes.get(nid, CausalNode(nid, nid, "", "", "")).name
                    for nid in path
                ])
                continue
            if node_id in visited or len(path) > 7:
                continue
            visited.add(node_id)
            for e in self.edges:
                if e.source_id == node_id and e.target_id not in path:
                    queue.append(path + [e.target_id])
        return paths

    def get_root_causes(self, effect_name: str) -> List[Dict]:
        """어떤 현상의 근본 원인들을 찾음"""
        effect_ids = {n.id for n in self.nodes.values()
                      if effect_name.lower() in n.name.lower()}
        causes = []
        for edge in self.edges:
            if edge.target_id in effect_ids:
                src = self.nodes.get(edge.source_id)
                if src:
                    causes.append({
                        "cause": src.name,
                        "strength": edge.strength,
                        "mechanism": edge.mechanism,
                        "relationship": edge.relationship,
                    })
        return sorted(causes, key=lambda x: x["strength"], reverse=True)

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "available": True,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "counterfactuals_run": len(self.counterfactuals),
            "domains": list({n.domain for n in self.nodes.values() if n.domain}),
        }
