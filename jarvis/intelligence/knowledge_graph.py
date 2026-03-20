"""
JARVIS 지식 그래프 — Iteration 5
개념들 사이의 관계를 그래프로 표현하여 인간보다 깊은 연결 추론 수행

핵심 기능:
- 개념(노드)과 관계(엣지)로 지식 표현
- 경로 탐색으로 간접 관계 발견
- 의미적 클러스터링으로 지식 영역 분류
- LLM 보조 관계 추출 (텍스트 → 그래프)
- 그래프 추론: "A는 B와 관련있고 B는 C와 관련있으면 A는 C와?"
- PageRank 기반 중요도 계산
- 실시간 그래프 업데이트 및 모순 감지
"""

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

KG_DB = Path("data/jarvis/knowledge_graph.json")


@dataclass
class KGNode:
    """지식 그래프 노드 (개념/엔티티)"""
    id: str
    name: str
    type: str          # concept / entity / event / fact / person / technology
    description: str = ""
    properties: Dict = field(default_factory=dict)
    importance: float = 0.5    # PageRank 점수
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""           # 지식 출처


@dataclass
class KGEdge:
    """지식 그래프 엣지 (관계)"""
    id: str
    source: str        # 소스 노드 ID
    target: str        # 대상 노드 ID
    relation: str      # 관계 유형 (is_a / has / causes / enables / contradicts / related_to 등)
    weight: float = 1.0       # 관계 강도
    evidence: str = ""        # 관계 근거
    bidirectional: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphPath:
    """그래프 경로"""
    nodes: List[str]
    edges: List[str]
    total_weight: float
    description: str


class KnowledgeGraph:
    """
    JARVIS 지식 그래프 엔진
    개념 관계 네트워크 기반의 심층 추론 시스템
    """

    EXTRACT_PROMPT = """텍스트에서 개념들과 그 관계를 추출하세요:

텍스트: {text}

JSON으로 반환:
{{
  "entities": [
    {{"name": "개념명", "type": "concept|entity|technology|person", "description": "설명"}}
  ],
  "relations": [
    {{
      "source": "소스 개념",
      "target": "대상 개념",
      "relation": "관계 유형",
      "evidence": "근거 문장",
      "bidirectional": false
    }}
  ]
}}

관계 유형: is_a(상위 개념), has(포함), causes(원인), enables(가능하게 함),
contradicts(모순), related_to(관련), used_in(사용됨), part_of(일부)

JSON만 출력하세요."""

    REASON_PROMPT = """지식 그래프를 기반으로 질문에 답하세요:

질문: {question}

관련 개념들:
{nodes}

관련 관계들:
{edges}

그래프 경로들:
{paths}

이 정보를 바탕으로 깊이 있는 추론을 수행하고 답변하세요.
직접 연결되지 않은 개념들 사이의 간접 관계도 설명하세요."""

    def __init__(self, llm_manager=None):
        self.llm = llm_manager
        self._nodes: Dict[str, KGNode] = {}
        self._edges: Dict[str, KGEdge] = {}
        # 인접 리스트
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._rev_adj: Dict[str, Set[str]] = defaultdict(set)
        self._edge_counter = 0
        self._load_graph()
        logger.info(f"KnowledgeGraph initialized: {len(self._nodes)} nodes, {len(self._edges)} edges")

    # ── 노드/엣지 추가 ─────────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        node_type: str = "concept",
        description: str = "",
        properties: Dict = None,
        source: str = "",
    ) -> KGNode:
        """노드 추가 (중복 시 업데이트)"""
        node_id = self._name_to_id(name)
        if node_id in self._nodes:
            # 기존 노드 업데이트
            existing = self._nodes[node_id]
            if description and not existing.description:
                existing.description = description
            if properties:
                existing.properties.update(properties)
            return existing

        node = KGNode(
            id=node_id,
            name=name,
            type=node_type,
            description=description,
            properties=properties or {},
            source=source,
        )
        self._nodes[node_id] = node
        self._adj[node_id] = set()
        self._rev_adj[node_id] = set()
        return node

    def add_edge(
        self,
        source_name: str,
        target_name: str,
        relation: str,
        weight: float = 1.0,
        evidence: str = "",
        bidirectional: bool = False,
    ) -> KGEdge:
        """엣지 추가"""
        src_node = self.add_node(source_name)
        tgt_node = self.add_node(target_name)

        edge_id = f"e{self._edge_counter}"
        self._edge_counter += 1

        edge = KGEdge(
            id=edge_id,
            source=src_node.id,
            target=tgt_node.id,
            relation=relation,
            weight=weight,
            evidence=evidence,
            bidirectional=bidirectional,
        )
        self._edges[edge_id] = edge
        self._adj[src_node.id].add(tgt_node.id)
        self._rev_adj[tgt_node.id].add(src_node.id)

        if bidirectional:
            self._adj[tgt_node.id].add(src_node.id)
            self._rev_adj[src_node.id].add(tgt_node.id)

        return edge

    # ── 텍스트에서 자동 추출 ───────────────────────────────────────────────

    def extract_from_text(self, text: str, source: str = "") -> Dict:
        """텍스트에서 개념과 관계 자동 추출 및 그래프 추가"""
        if not self.llm:
            return {"error": "LLM not available"}

        from jarvis.llm.manager import Message
        prompt = self.EXTRACT_PROMPT.format(text=text[:3000])
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=2048)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if not match:
                return {"error": "No JSON found"}

            data = json.loads(match.group())
            nodes_added = []
            edges_added = []

            for entity_data in data.get("entities", []):
                name = entity_data.get("name", "")
                if name:
                    node = self.add_node(
                        name=name,
                        node_type=entity_data.get("type", "concept"),
                        description=entity_data.get("description", ""),
                        source=source,
                    )
                    nodes_added.append(node.name)

            for rel_data in data.get("relations", []):
                src = rel_data.get("source", "")
                tgt = rel_data.get("target", "")
                rel = rel_data.get("relation", "related_to")
                if src and tgt:
                    edge = self.add_edge(
                        source_name=src,
                        target_name=tgt,
                        relation=rel,
                        evidence=rel_data.get("evidence", ""),
                        bidirectional=rel_data.get("bidirectional", False),
                    )
                    edges_added.append(f"{src} —[{rel}]→ {tgt}")

            self._save_graph()
            logger.info(f"[KG] Extracted: {len(nodes_added)} nodes, {len(edges_added)} edges")
            return {
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
            }

        except Exception as e:
            logger.error(f"[KG] Extract error: {e}")
            return {"error": str(e)}

    # ── 그래프 탐색 ────────────────────────────────────────────────────────

    def find_path(self, source_name: str, target_name: str, max_hops: int = 5) -> Optional[GraphPath]:
        """두 개념 간 최단 경로 탐색 (BFS)"""
        src_id = self._name_to_id(source_name)
        tgt_id = self._name_to_id(target_name)

        if src_id not in self._nodes or tgt_id not in self._nodes:
            return None

        if src_id == tgt_id:
            return GraphPath(nodes=[src_id], edges=[], total_weight=0.0, description="같은 개념")

        # BFS
        queue = deque([(src_id, [src_id], [])])
        visited = {src_id}

        while queue:
            current, path, edge_path = queue.popleft()
            if len(path) > max_hops:
                continue

            for neighbor in self._adj.get(current, set()):
                if neighbor == tgt_id:
                    final_path = path + [neighbor]
                    edge_labels = self._get_edge_labels(final_path)
                    return GraphPath(
                        nodes=final_path,
                        edges=edge_path,
                        total_weight=len(final_path) - 1,
                        description=self._path_to_description(final_path, edge_labels),
                    )
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], edge_path))

        return None

    def get_neighbors(self, name: str, depth: int = 1) -> Dict:
        """개념의 이웃 노드들 조회"""
        node_id = self._name_to_id(name)
        if node_id not in self._nodes:
            return {"error": f"Node '{name}' not found"}

        result = {"center": name, "neighbors": [], "relations": []}
        visited = {node_id}
        current_level = {node_id}

        for d in range(depth):
            next_level = set()
            for nid in current_level:
                for neighbor_id in self._adj.get(nid, set()):
                    if neighbor_id not in visited and neighbor_id in self._nodes:
                        neighbor = self._nodes[neighbor_id]
                        result["neighbors"].append({
                            "name": neighbor.name,
                            "type": neighbor.type,
                            "depth": d + 1,
                            "importance": neighbor.importance,
                        })
                        next_level.add(neighbor_id)
                        visited.add(neighbor_id)

                        # 관계 정보
                        edge = self._find_edge(nid, neighbor_id)
                        if edge:
                            result["relations"].append({
                                "from": self._nodes[nid].name,
                                "to": neighbor.name,
                                "relation": edge.relation,
                                "weight": edge.weight,
                            })
            current_level = next_level

        return result

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """키워드 기반 개념 검색"""
        query_lower = query.lower()
        scored = []
        for node in self._nodes.values():
            score = 0.0
            if query_lower in node.name.lower():
                score += 3.0
            if query_lower in node.description.lower():
                score += 1.0
            for word in query_lower.split():
                if word in node.name.lower():
                    score += 1.0
                if word in node.description.lower():
                    score += 0.5
            score += node.importance * 0.5
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: -x[0])
        return [
            {
                "name": n.name,
                "type": n.type,
                "description": n.description,
                "importance": n.importance,
                "degree": len(self._adj.get(n.id, set())),
                "relevance": round(s, 2),
            }
            for s, n in scored[:top_k]
        ]

    def get_clusters(self, min_size: int = 2) -> List[Dict]:
        """연결 컴포넌트로 지식 클러스터 탐색"""
        visited = set()
        clusters = []

        for node_id in self._nodes:
            if node_id not in visited:
                cluster = self._bfs_cluster(node_id, visited)
                if len(cluster) >= min_size:
                    clusters.append({
                        "size": len(cluster),
                        "nodes": [self._nodes[nid].name for nid in cluster[:10]],
                        "central_node": self._most_connected(cluster),
                    })

        clusters.sort(key=lambda c: -c["size"])
        return clusters

    # ── PageRank 중요도 계산 ────────────────────────────────────────────────

    def compute_pagerank(self, iterations: int = 20, damping: float = 0.85):
        """PageRank로 노드 중요도 계산"""
        n = len(self._nodes)
        if n == 0:
            return

        node_ids = list(self._nodes.keys())
        rank = {nid: 1.0 / n for nid in node_ids}

        for _ in range(iterations):
            new_rank = {}
            for nid in node_ids:
                incoming = self._rev_adj.get(nid, set())
                r = (1 - damping) / n
                for src_id in incoming:
                    out_degree = len(self._adj.get(src_id, set()))
                    if out_degree > 0:
                        r += damping * rank[src_id] / out_degree
                new_rank[nid] = r
            rank = new_rank

        for nid, r in rank.items():
            if nid in self._nodes:
                self._nodes[nid].importance = round(r * n, 4)  # 정규화

        logger.info(f"[KG] PageRank computed for {n} nodes")

    # ── 그래프 추론 ────────────────────────────────────────────────────────

    def reason(self, question: str) -> Dict:
        """질문에 대해 그래프 기반 추론"""
        if not self.llm:
            return {"error": "LLM not available"}

        # 관련 개념 검색
        relevant_nodes = self.semantic_search(question, top_k=8)
        if not relevant_nodes:
            return {"error": "관련 개념 없음", "answer": "지식 그래프에 관련 정보가 없습니다."}

        # 관련 관계 수집
        node_names = [n["name"] for n in relevant_nodes]
        edges_info = self._get_edges_for_nodes(node_names[:5])

        # 경로 탐색 (상위 2개 노드 간)
        paths_info = []
        if len(node_names) >= 2:
            path = self.find_path(node_names[0], node_names[1], max_hops=4)
            if path:
                paths_info.append(path.description)

        from jarvis.llm.manager import Message
        prompt = self.REASON_PROMPT.format(
            question=question,
            nodes="\n".join([f"- {n['name']} ({n['type']}): {n['description']}" for n in relevant_nodes]),
            edges="\n".join(edges_info[:10]),
            paths="\n".join(paths_info) or "직접 경로 없음",
        )
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=2048)
            return {
                "question": question,
                "answer": resp.content,
                "relevant_concepts": node_names,
                "graph_paths": paths_info,
            }
        except Exception as e:
            return {"error": str(e)}

    def detect_contradictions(self) -> List[Dict]:
        """그래프 내 모순 관계 탐지"""
        contradictions = []
        for edge_id, edge in self._edges.items():
            if edge.relation == "contradicts":
                src = self._nodes.get(edge.source)
                tgt = self._nodes.get(edge.target)
                if src and tgt:
                    # 같은 노드가 서로 모순되는지 확인
                    contradictions.append({
                        "node1": src.name,
                        "node2": tgt.name,
                        "evidence": edge.evidence,
                    })
        return contradictions

    # ── 통계 ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        type_counts: Dict[str, int] = {}
        relation_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
        for edge in self._edges.values():
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1

        avg_degree = (
            sum(len(adj) for adj in self._adj.values()) / max(len(self._nodes), 1)
        )
        top_nodes = sorted(self._nodes.values(), key=lambda n: -n.importance)[:5]

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": type_counts,
            "relation_types": relation_counts,
            "avg_degree": round(avg_degree, 2),
            "top_nodes": [{"name": n.name, "importance": n.importance} for n in top_nodes],
        }

    def get_all_nodes(self, limit: int = 100) -> List[Dict]:
        nodes = sorted(self._nodes.values(), key=lambda n: -n.importance)[:limit]
        return [
            {
                "id": n.id,
                "name": n.name,
                "type": n.type,
                "description": n.description[:100],
                "importance": n.importance,
                "degree": len(self._adj.get(n.id, set())),
            }
            for n in nodes
        ]

    def get_all_edges(self, limit: int = 200) -> List[Dict]:
        edges = list(self._edges.values())[:limit]
        return [
            {
                "source": self._nodes[e.source].name if e.source in self._nodes else e.source,
                "target": self._nodes[e.target].name if e.target in self._nodes else e.target,
                "relation": e.relation,
                "weight": e.weight,
            }
            for e in edges
        ]

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    def _name_to_id(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")[:50]

    def _find_edge(self, src_id: str, tgt_id: str) -> Optional[KGEdge]:
        for edge in self._edges.values():
            if edge.source == src_id and edge.target == tgt_id:
                return edge
        return None

    def _get_edge_labels(self, path: List[str]) -> List[str]:
        labels = []
        for i in range(len(path) - 1):
            edge = self._find_edge(path[i], path[i + 1])
            labels.append(edge.relation if edge else "→")
        return labels

    def _path_to_description(self, path: List[str], edge_labels: List[str]) -> str:
        parts = []
        for i, nid in enumerate(path):
            name = self._nodes[nid].name if nid in self._nodes else nid
            parts.append(name)
            if i < len(edge_labels):
                parts.append(f"—[{edge_labels[i]}]→")
        return " ".join(parts)

    def _get_edges_for_nodes(self, node_names: List[str]) -> List[str]:
        node_ids = {self._name_to_id(n) for n in node_names}
        result = []
        for edge in self._edges.values():
            if edge.source in node_ids or edge.target in node_ids:
                src_name = self._nodes[edge.source].name if edge.source in self._nodes else edge.source
                tgt_name = self._nodes[edge.target].name if edge.target in self._nodes else edge.target
                result.append(f"{src_name} —[{edge.relation}]→ {tgt_name}")
        return result

    def _bfs_cluster(self, start_id: str, visited: set) -> List[str]:
        cluster = []
        queue = deque([start_id])
        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            cluster.append(nid)
            for neighbor in self._adj.get(nid, set()) | self._rev_adj.get(nid, set()):
                if neighbor not in visited and neighbor in self._nodes:
                    queue.append(neighbor)
        return cluster

    def _most_connected(self, node_ids: List[str]) -> str:
        if not node_ids:
            return ""
        best = max(node_ids, key=lambda nid: len(self._adj.get(nid, set())))
        return self._nodes[best].name if best in self._nodes else ""

    # ── 영속성 ─────────────────────────────────────────────────────────────

    def _save_graph(self):
        try:
            KG_DB.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "nodes": {
                    nid: {
                        "id": n.id, "name": n.name, "type": n.type,
                        "description": n.description, "properties": n.properties,
                        "importance": n.importance, "source": n.source,
                        "created_at": n.created_at,
                    }
                    for nid, n in self._nodes.items()
                },
                "edges": {
                    eid: {
                        "id": e.id, "source": e.source, "target": e.target,
                        "relation": e.relation, "weight": e.weight,
                        "evidence": e.evidence, "bidirectional": e.bidirectional,
                        "created_at": e.created_at,
                    }
                    for eid, e in self._edges.items()
                },
                "edge_counter": self._edge_counter,
            }
            KG_DB.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[KG] Save error: {e}")

    def _load_graph(self):
        try:
            if KG_DB.exists():
                data = json.loads(KG_DB.read_text(encoding="utf-8"))
                self._edge_counter = data.get("edge_counter", 0)
                for nid, nd in data.get("nodes", {}).items():
                    self._nodes[nid] = KGNode(
                        id=nd["id"], name=nd["name"], type=nd.get("type", "concept"),
                        description=nd.get("description", ""), properties=nd.get("properties", {}),
                        importance=nd.get("importance", 0.5), source=nd.get("source", ""),
                        created_at=nd.get("created_at", datetime.now().isoformat()),
                    )
                for eid, ed in data.get("edges", {}).items():
                    self._edges[eid] = KGEdge(
                        id=ed["id"], source=ed["source"], target=ed["target"],
                        relation=ed.get("relation", "related_to"), weight=ed.get("weight", 1.0),
                        evidence=ed.get("evidence", ""), bidirectional=ed.get("bidirectional", False),
                        created_at=ed.get("created_at", datetime.now().isoformat()),
                    )
                    self._adj[ed["source"]].add(ed["target"])
                    self._rev_adj[ed["target"]].add(ed["source"])
                    if ed.get("bidirectional"):
                        self._adj[ed["target"]].add(ed["source"])
                logger.info(f"[KG] Loaded: {len(self._nodes)} nodes, {len(self._edges)} edges")
        except Exception as e:
            logger.debug(f"[KG] Load error: {e}")
