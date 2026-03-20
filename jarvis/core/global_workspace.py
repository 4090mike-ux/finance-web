"""
JARVIS Global Workspace Theory — Iteration 8
Baars (1988)의 전역 작업공간 이론 구현

핵심 원리:
  1. Competition — 모든 모듈이 "방송" 권한 경쟁 (salience 기반)
  2. Broadcast   — 우승 모듈이 전체 시스템에 정보 전파
  3. Integration — 다른 모듈들이 방송을 받아 처리 및 응답
  4. Working Memory — 방송된 내용의 단기 유지 (7±2 용량, Miller)

이 구조가 인간 의식과 유사한 통합적 정보 처리를 가능하게 함.
"""

import time
import threading
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceItem:
    id: str
    content: Any
    source_module: str
    salience: float             # 0–1; 높을수록 방송 우선
    tags: List[str] = field(default_factory=list)
    broadcast_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class BroadcastEvent:
    winner_module: str
    content_preview: str
    salience: float
    responders: List[str]
    timestamp: float = field(default_factory=time.time)


class GlobalWorkspace:
    """
    전역 작업공간 (Global Workspace)

    모든 JARVIS 서브시스템은 이 버스를 통해 정보를 공유하고
    경쟁적으로 방송 권한을 획득한다.
    """

    CAPACITY          = 14    # 동시 유지 항목 수 (Miller × 2)
    BROADCAST_INTERVAL = 15   # seconds between auto-broadcasts
    HARVEST_INTERVAL   = 45   # seconds between subsystem harvests

    def __init__(
        self,
        jarvis_engine=None,
        event_callback: Optional[Callable] = None,
    ):
        self.engine   = jarvis_engine
        self._cb      = event_callback
        self._lock    = threading.Lock()

        self.workspace: List[WorkspaceItem]   = []
        self.history:   List[BroadcastEvent]  = []
        self.registry:  Dict[str, Dict]       = {}

        self._running = False
        self._threads: List[threading.Thread] = []

        self._register_defaults()
        logger.info("GlobalWorkspace initialized — Baars cognitive architecture")

    # ── Module Registry ───────────────────────────────────────────────

    def _register_defaults(self):
        modules = [
            ("consciousness_loop",   "의식 루프",      0.90),
            ("executive",            "총사령관",        0.95),
            ("tree_of_thoughts",     "사고의 나무",     0.85),
            ("agent_swarm",          "에이전트 스웜",   0.75),
            ("knowledge_graph",      "지식 그래프",     0.80),
            ("memory_palace",        "기억의 궁전",     0.80),
            ("meta_learner",         "메타 학습기",     0.70),
            ("live_monitor",         "실시간 모니터",   0.65),
            ("causal_engine",        "인과 추론",       0.80),
            ("hypothesis_engine",    "가설 엔진",       0.78),
            ("temporal_engine",      "시간 추론",       0.72),
            ("recursive_improver",   "재귀 개선기",     0.68),
            ("goal_hierarchy",       "목표 계층",       0.75),
            ("web_agent",            "웹 에이전트",     0.62),
        ]
        for mid, name, base in modules:
            self.register_module(mid, name, base)

    def register_module(self, module_id: str, name: str,
                        base_salience: float = 0.5):
        self.registry[module_id] = {
            "name": name,
            "base_salience": base_salience,
            "broadcasts_won": 0,
            "items_contributed": 0,
            "last_active": time.time(),
        }

    # ── Contribution & Competition ────────────────────────────────────

    def contribute(self, module_id: str, content: Any,
                   salience_boost: float = 0.0,
                   tags: List[str] = None) -> WorkspaceItem:
        """모듈이 작업공간에 항목 기여"""
        if module_id not in self.registry:
            self.register_module(module_id, module_id)

        base     = self.registry[module_id]["base_salience"]
        salience = min(1.0, base + salience_boost)

        item = WorkspaceItem(
            id=f"{module_id}_{int(time.time()*1000) % 1_000_000}",
            content=content,
            source_module=module_id,
            salience=salience,
            tags=tags or [],
        )

        with self._lock:
            self.workspace.append(item)
            self.registry[module_id]["items_contributed"] += 1
            self.registry[module_id]["last_active"] = time.time()
            # Prune to capacity
            if len(self.workspace) > self.CAPACITY * 3:
                self.workspace = sorted(
                    self.workspace,
                    key=lambda x: x.salience * (1 / (1 + (time.time() - x.timestamp) / 120)),
                    reverse=True,
                )[:self.CAPACITY * 2]
        return item

    def compete_and_broadcast(self) -> Optional[BroadcastEvent]:
        """경쟁 → 방송: 가장 현저한(salient) 항목이 전 시스템에 전파"""
        with self._lock:
            if not self.workspace:
                return None
            now = time.time()
            # Salience decays with age
            ranked = sorted(
                self.workspace,
                key=lambda x: x.salience * (1 / (1 + (now - x.timestamp) / 120)),
                reverse=True,
            )
            winner = ranked[0]
            winner.broadcast_count += 1
            # Remove from workspace
            self.workspace = [x for x in self.workspace if x.id != winner.id]

        # Broadcast to subsystems
        responders = self._propagate(winner)

        evt = BroadcastEvent(
            winner_module=winner.source_module,
            content_preview=str(winner.content)[:200],
            salience=winner.salience,
            responders=responders,
        )
        self.history.append(evt)
        if len(self.history) > 200:
            self.history = self.history[-200:]

        if winner.source_module in self.registry:
            self.registry[winner.source_module]["broadcasts_won"] += 1

        self._emit("broadcast", {
            "module": winner.source_module,
            "salience": round(winner.salience, 3),
            "responders": len(responders),
            "preview": str(winner.content)[:100],
        })
        return evt

    def _propagate(self, winner: WorkspaceItem) -> List[str]:
        """당선 항목을 모든 서브시스템에 전파"""
        e = self.engine
        responders: List[str] = []
        if not e:
            return responders

        content   = winner.content
        src_mod   = winner.source_module
        sal       = winner.salience
        content_s = str(content)[:500]

        # MemoryPalace: 중요 항목 기억
        if src_mod != "memory_palace" and sal > 0.7:
            mp = getattr(e, "memory_palace", None)
            if mp:
                try:
                    mp.remember(
                        content=content_s,
                        memory_type="working",
                        importance=sal,
                        tags=winner.tags + ["gw_broadcast"],
                    )
                    responders.append("memory_palace")
                except Exception:
                    pass

        # KnowledgeGraph: 새 지식 노드
        if src_mod != "knowledge_graph" and sal > 0.8:
            kg = getattr(e, "kg", None)
            if kg and isinstance(content, str):
                try:
                    kg.add_node(
                        name=f"gw_{winner.id[:6]}",
                        description=content_s[:300],
                        node_type="broadcast",
                        importance=sal,
                    )
                    responders.append("knowledge_graph")
                except Exception:
                    pass

        # HypothesisEngine: 방송된 내용을 관찰로 등록
        if src_mod not in ("hypothesis_engine", "memory_palace"):
            he = getattr(e, "hypothesis_engine", None)
            if he and sal > 0.6:
                try:
                    he.observe(content_s[:300], source=f"gw_{src_mod}", reliability=sal)
                    responders.append("hypothesis_engine")
                except Exception:
                    pass

        # TemporalEngine: 이벤트로 등록
        if src_mod != "temporal_engine" and sal > 0.65:
            te = getattr(e, "temporal_engine", None)
            if te:
                try:
                    te.add_event(
                        description=content_s[:200],
                        domain=winner.tags[0] if winner.tags else "workspace",
                        certainty=sal,
                        source=f"gw_{src_mod}",
                    )
                    responders.append("temporal_engine")
                except Exception:
                    pass

        # MetaLearner
        if src_mod != "meta_learner" and sal > 0.75:
            ml = getattr(e, "meta_learner", None)
            if ml:
                try:
                    ml.record_outcome(
                        query=f"gw_broadcast_{src_mod}",
                        category="integration",
                        strategy="global_workspace",
                        success=True,
                        rating=sal,
                        tools_used=[src_mod, "global_workspace"],
                    )
                    responders.append("meta_learner")
                except Exception:
                    pass

        return responders

    # ── Auto-loop ─────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True

        def broadcast_loop():
            while self._running:
                try:
                    if self.workspace:
                        self.compete_and_broadcast()
                except Exception as ex:
                    logger.error(f"GW broadcast loop: {ex}")
                time.sleep(self.BROADCAST_INTERVAL)

        def harvest_loop():
            while self._running:
                try:
                    self._harvest()
                except Exception as ex:
                    logger.error(f"GW harvest loop: {ex}")
                time.sleep(self.HARVEST_INTERVAL)

        t1 = threading.Thread(target=broadcast_loop, daemon=True, name="gw-broadcast")
        t2 = threading.Thread(target=harvest_loop,   daemon=True, name="gw-harvest")
        t1.start(); t2.start()
        self._threads = [t1, t2]
        logger.info("GlobalWorkspace loops started")

    def stop(self):
        self._running = False

    def _harvest(self):
        """서브시스템에서 중요 정보 자동 수집"""
        e = self.engine
        if not e:
            return

        # ConsciousnessLoop quality alert
        con = getattr(e, "consciousness", None)
        if con:
            try:
                s = con.get_cognitive_status()
                avg_q = s.get("stats", {}).get("average_quality", 1.0)
                if avg_q < 0.55:
                    self.contribute(
                        "consciousness_loop",
                        f"⚠️ 응답 품질 저하 감지: {avg_q:.2f}. 즉각 개선 필요.",
                        salience_boost=0.35,
                        tags=["quality_alert"],
                    )
            except Exception:
                pass

        # LiveMonitor new items
        lm = getattr(e, "live_monitor", None)
        if lm:
            try:
                st = lm.get_status()
                n  = st.get("feed_items", 0)
                if n > 0:
                    self.contribute(
                        "live_monitor",
                        f"실시간 피드 업데이트: {n}개 항목",
                        salience_boost=0.05,
                        tags=["feed", "live"],
                    )
            except Exception:
                pass

        # RecursiveImprover cycle result
        ri = getattr(e, "recursive_improver", None)
        if ri:
            try:
                st = ri.get_status()
                cycles = st.get("total_cycles", 0)
                delta  = st.get("total_quality_delta", 0.0)
                if cycles > 0:
                    boost = 0.15 if delta > 0 else 0.0
                    self.contribute(
                        "recursive_improver",
                        f"재귀 개선 {cycles}사이클 완료 · Δ품질={delta:+.3f}",
                        salience_boost=boost,
                        tags=["improvement"],
                    )
            except Exception:
                pass

        # HypothesisEngine: new supported hypothesis
        he = getattr(e, "hypothesis_engine", None)
        if he:
            try:
                st = he.get_status()
                sup = st.get("supported_count", 0)
                if sup > 0:
                    self.contribute(
                        "hypothesis_engine",
                        f"{sup}개 가설이 검증됨 (지지율 >75%)",
                        salience_boost=0.10,
                        tags=["hypothesis", "knowledge"],
                    )
            except Exception:
                pass

    # ── Introspection ─────────────────────────────────────────────────

    def get_workspace_state(self) -> List[Dict]:
        with self._lock:
            now = time.time()
            return [
                {
                    "id":      item.id,
                    "module":  item.source_module,
                    "salience": round(item.salience, 3),
                    "preview": str(item.content)[:120],
                    "tags":    item.tags,
                    "age_s":   round(now - item.timestamp, 1),
                }
                for item in sorted(
                    self.workspace,
                    key=lambda x: x.salience,
                    reverse=True,
                )
            ]

    def get_recent_broadcasts(self, limit: int = 15) -> List[Dict]:
        return [
            {
                "module":    b.winner_module,
                "salience":  round(b.salience, 3),
                "responders": b.responders,
                "preview":   b.content_preview,
                "timestamp": b.timestamp,
            }
            for b in sorted(
                self.history, key=lambda x: x.timestamp, reverse=True
            )[:limit]
        ]

    def _emit(self, event_type: str, data: Dict):
        if self._cb:
            try:
                self._cb({"type": event_type, **data})
            except Exception:
                pass

    def get_status(self) -> Dict:
        return {
            "available":         True,
            "is_running":        self._running,
            "workspace_items":   len(self.workspace),
            "registered_modules": len(self.registry),
            "total_broadcasts":  len(self.history),
            "module_stats": {
                mid: {
                    "broadcasts_won":   m["broadcasts_won"],
                    "items_contributed": m["items_contributed"],
                }
                for mid, m in self.registry.items()
            },
        }
