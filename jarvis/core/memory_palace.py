"""
JARVIS 기억의 궁전 (Memory Palace) — Iteration 6
인간의 기억 체계를 모델링한 계층적 장기 기억 시스템

기억 유형:
- 에피소드 기억 (Episodic): 구체적 사건, 대화, 경험 — "언제 무슨 일이 있었다"
- 의미 기억 (Semantic): 일반 지식, 사실, 개념 — "세상이 어떻게 작동하는가"
- 절차 기억 (Procedural): 방법, 기술, 레시피 — "어떻게 하는가"
- 작업 기억 (Working): 현재 대화 컨텍스트 — "지금 무엇을 하고 있는가"
- 감정 기억 (Emotional): 사용자 선호, 피드백 — "무엇을 좋아하는가"

특징:
- 중요도 기반 자동 망각 (Forgetting Curve 모사)
- 연관 기억 검색 (Associative Retrieval)
- 기억 강화 (Rehearsal): 자주 접근할수록 강화
- 수면 통합 (Consolidation): 주기적으로 단기 → 장기 변환
"""

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

PALACE_DB = Path("data/jarvis/memory_palace.json")


class MemoryType:
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    EMOTIONAL = "emotional"


@dataclass
class Memory:
    """단일 기억 단위"""
    id: str
    type: str                    # MemoryType.*
    content: str
    summary: str = ""            # 짧은 요약
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5      # 0-1
    strength: float = 1.0        # 기억 강도 (접근할수록 증가)
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    context: Dict = field(default_factory=dict)  # 추가 메타데이터

    def decay(self, hours_since_access: float) -> float:
        """에빙하우스 망각 곡선으로 현재 기억 강도 계산"""
        # R = e^(-t/S) where S = stability
        stability = self.importance * 10 * (1 + self.access_count * 0.5)
        return self.strength * math.exp(-hours_since_access / stability)

    def current_strength(self) -> float:
        last = datetime.fromisoformat(self.last_accessed)
        hours = (datetime.now() - last).total_seconds() / 3600
        return round(self.decay(hours), 4)

    def reinforce(self, amount: float = 0.1):
        """기억 강화 (접근 시 호출)"""
        self.strength = min(1.0, self.strength + amount)
        self.access_count += 1
        self.last_accessed = datetime.now().isoformat()


class MemoryPalace:
    """
    JARVIS 기억의 궁전
    인간보다 정확하고 체계적인 장기 기억 시스템
    """

    # 각 유형별 최대 기억 수
    MAX_MEMORIES = {
        MemoryType.EPISODIC: 2000,
        MemoryType.SEMANTIC: 5000,
        MemoryType.PROCEDURAL: 1000,
        MemoryType.WORKING: 50,
        MemoryType.EMOTIONAL: 500,
    }

    # 망각 임계값 (이 이하이면 망각 대상)
    FORGET_THRESHOLD = 0.05

    def __init__(self, llm_manager=None):
        self.llm = llm_manager
        self._memories: Dict[str, Memory] = {}
        self._type_index: Dict[str, List[str]] = defaultdict(list)  # type → [id]
        self._tag_index: Dict[str, List[str]] = defaultdict(list)   # tag → [id]
        self._working_context: List[str] = []  # 현재 세션 기억 ID
        self._load_palace()
        logger.info(f"MemoryPalace initialized: {len(self._memories)} memories")

    # ── 기억 저장 ──────────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = MemoryType.SEMANTIC,
        summary: str = "",
        tags: List[str] = None,
        importance: float = 0.5,
        context: Dict = None,
    ) -> Memory:
        """새 기억 저장"""
        # 중복 확인 (유사한 내용이 있으면 강화)
        existing = self._find_similar(content, memory_type)
        if existing:
            existing.reinforce(0.2)
            self._save_palace()
            return existing

        mem_id = str(uuid4())[:8]
        summary = summary or content[:100]

        memory = Memory(
            id=mem_id,
            type=memory_type,
            content=content,
            summary=summary,
            tags=tags or self._auto_tag(content),
            importance=importance,
            context=context or {},
        )

        self._memories[mem_id] = memory
        self._type_index[memory_type].append(mem_id)
        for tag in memory.tags:
            self._tag_index[tag].append(mem_id)

        # 작업 기억에 추가
        if memory_type == MemoryType.WORKING:
            self._working_context.append(mem_id)
            if len(self._working_context) > self.MAX_MEMORIES[MemoryType.WORKING]:
                self._working_context.pop(0)

        # 용량 초과 시 망각
        self._enforce_capacity(memory_type)
        self._save_palace()
        return memory

    def remember_episode(self, event: str, context: str = "", importance: float = 0.5) -> Memory:
        """에피소드 기억 저장 (대화, 사건)"""
        return self.remember(
            content=f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {event}",
            memory_type=MemoryType.EPISODIC,
            summary=event[:80],
            importance=importance,
            context={"context": context},
        )

    def learn(self, fact: str, importance: float = 0.6) -> Memory:
        """의미 기억 저장 (지식/사실)"""
        return self.remember(content=fact, memory_type=MemoryType.SEMANTIC, importance=importance)

    def learn_procedure(self, skill_name: str, steps: List[str]) -> Memory:
        """절차 기억 저장 (방법/기술)"""
        content = f"기술: {skill_name}\n단계:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        return self.remember(content=content, memory_type=MemoryType.PROCEDURAL, importance=0.7)

    def note_preference(self, preference: str, sentiment: float = 1.0) -> Memory:
        """감정/선호 기억 저장"""
        return self.remember(
            content=preference,
            memory_type=MemoryType.EMOTIONAL,
            importance=0.8,
            context={"sentiment": sentiment},
        )

    # ── 기억 검색 ──────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_strength: float = 0.1,
    ) -> List[Memory]:
        """연관 기억 검색"""
        query_lower = query.lower()
        scored: List[Tuple[float, Memory]] = []

        candidates = (
            [self._memories[mid] for mid in self._type_index.get(memory_type, []) if mid in self._memories]
            if memory_type
            else list(self._memories.values())
        )

        for mem in candidates:
            strength = mem.current_strength()
            if strength < min_strength:
                continue

            # 텍스트 관련도
            relevance = 0.0
            for word in query_lower.split():
                if len(word) > 2:
                    if word in mem.content.lower():
                        relevance += 1.0
                    if word in mem.summary.lower():
                        relevance += 0.5
                    if any(word in tag for tag in mem.tags):
                        relevance += 0.3

            if relevance > 0:
                final_score = relevance * strength * (1 + mem.importance)
                scored.append((final_score, mem))

        scored.sort(key=lambda x: -x[0])
        result = [mem for _, mem in scored[:top_k]]

        # 접근한 기억 강화
        for mem in result:
            mem.reinforce(0.05)

        return result

    def recall_recent(self, n: int = 10, memory_type: Optional[str] = None) -> List[Memory]:
        """최근 기억 조회"""
        mems = list(self._memories.values())
        if memory_type:
            mems = [m for m in mems if m.type == memory_type]
        mems.sort(key=lambda m: m.last_accessed, reverse=True)
        return mems[:n]

    def get_working_context(self) -> str:
        """현재 세션 컨텍스트 (작업 기억)"""
        working = [
            self._memories[mid] for mid in self._working_context
            if mid in self._memories
        ][-10:]
        if not working:
            return ""
        return "\n".join(f"[{m.type}] {m.summary}" for m in working)

    def get_user_preferences(self) -> List[Dict]:
        """사용자 선호/피드백 기억 조회"""
        prefs = [
            m for m in self._memories.values()
            if m.type == MemoryType.EMOTIONAL
        ]
        prefs.sort(key=lambda m: (-m.importance, -m.strength))
        return [
            {"preference": m.content[:100], "sentiment": m.context.get("sentiment", 1.0),
             "strength": m.current_strength()}
            for m in prefs[:20]
        ]

    # ── 기억 통합 (수면 효과) ─────────────────────────────────────────────

    def consolidate(self) -> Dict:
        """기억 통합 — 단기 → 장기 변환, 불필요한 기억 망각"""
        forgotten = 0
        strengthened = 0
        consolidated = 0

        # 망각 처리 (강도 낮은 비중요 기억)
        to_forget = []
        for mem_id, mem in self._memories.items():
            if mem.type == MemoryType.WORKING:
                continue  # 작업 기억은 건드리지 않음
            strength = mem.current_strength()
            if strength < self.FORGET_THRESHOLD and mem.importance < 0.6:
                to_forget.append(mem_id)

        for mem_id in to_forget:
            self._forget(mem_id)
            forgotten += 1

        # 자주 접근된 에피소드 → 의미 기억으로 통합
        episodic_strong = [
            m for m in self._memories.values()
            if m.type == MemoryType.EPISODIC and m.access_count >= 3 and m.strength > 0.7
        ]
        for mem in episodic_strong[:5]:
            # 중요한 에피소드는 의미 기억으로 승격
            self.remember(
                content=f"(통합됨) {mem.summary}",
                memory_type=MemoryType.SEMANTIC,
                importance=mem.importance,
            )
            mem.importance = max(0.3, mem.importance - 0.1)  # 원본 중요도 낮춤
            consolidated += 1

        self._save_palace()
        logger.info(f"[MemoryPalace] Consolidation: forgot={forgotten}, consolidated={consolidated}")
        return {
            "forgotten": forgotten,
            "consolidated": consolidated,
            "total_memories": len(self._memories),
        }

    # ── 자동 태깅 ──────────────────────────────────────────────────────────

    def _auto_tag(self, content: str) -> List[str]:
        """컨텐츠 기반 자동 태그 생성"""
        content_lower = content.lower()
        tags = []
        tag_patterns = {
            "AI": ["ai", "llm", "gpt", "claude", "neural", "model"],
            "코딩": ["python", "code", "function", "class", "bug"],
            "연구": ["paper", "arxiv", "research", "논문"],
            "사용자": ["사용자", "user", "선호", "피드백"],
            "시스템": ["cpu", "memory", "disk", "server", "process"],
        }
        for tag, keywords in tag_patterns.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)
        return tags[:3]

    def _find_similar(self, content: str, memory_type: str, threshold: float = 0.8) -> Optional[Memory]:
        """매우 유사한 기억 찾기 (중복 방지)"""
        content_words = set(content.lower().split())
        for mid in self._type_index.get(memory_type, []):
            mem = self._memories.get(mid)
            if not mem:
                continue
            mem_words = set(mem.content.lower().split())
            if not content_words or not mem_words:
                continue
            intersection = content_words & mem_words
            union = content_words | mem_words
            jaccard = len(intersection) / len(union)
            if jaccard > threshold:
                return mem
        return None

    def _forget(self, mem_id: str):
        """기억 삭제"""
        mem = self._memories.pop(mem_id, None)
        if not mem:
            return
        # 인덱스에서 제거
        type_list = self._type_index.get(mem.type, [])
        if mem_id in type_list:
            type_list.remove(mem_id)
        for tag in mem.tags:
            tag_list = self._tag_index.get(tag, [])
            if mem_id in tag_list:
                tag_list.remove(mem_id)

    def _enforce_capacity(self, memory_type: str):
        """용량 초과 시 가장 약한 기억 제거"""
        max_cap = self.MAX_MEMORIES.get(memory_type, 1000)
        type_ids = self._type_index.get(memory_type, [])
        if len(type_ids) <= max_cap:
            return

        # 강도 낮은 순으로 제거
        scored = [(self._memories[mid].current_strength() * self._memories[mid].importance, mid)
                  for mid in type_ids if mid in self._memories]
        scored.sort(key=lambda x: x[0])
        to_remove = len(type_ids) - max_cap
        for _, mid in scored[:to_remove]:
            self._forget(mid)

    # ── 통계 ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        total = len(self._memories)
        by_type = {t: len(ids) for t, ids in self._type_index.items()}
        avg_strength = (
            sum(m.current_strength() for m in self._memories.values()) / max(total, 1)
        )
        return {
            "total_memories": total,
            "by_type": by_type,
            "avg_strength": round(avg_strength, 3),
            "working_context_size": len(self._working_context),
            "unique_tags": len(self._tag_index),
        }

    def get_all_memories(self, memory_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        mems = list(self._memories.values())
        if memory_type:
            mems = [m for m in mems if m.type == memory_type]
        mems.sort(key=lambda m: (-m.importance, -m.current_strength()))
        return [
            {
                "id": m.id,
                "type": m.type,
                "summary": m.summary,
                "content": m.content[:200],
                "importance": m.importance,
                "strength": m.current_strength(),
                "access_count": m.access_count,
                "tags": m.tags,
                "created_at": m.created_at,
                "last_accessed": m.last_accessed,
            }
            for m in mems[:limit]
        ]

    # ── 영속성 ─────────────────────────────────────────────────────────────

    def _save_palace(self):
        try:
            PALACE_DB.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "memories": {
                    mid: {
                        "id": m.id, "type": m.type, "content": m.content,
                        "summary": m.summary, "tags": m.tags,
                        "importance": m.importance, "strength": m.strength,
                        "access_count": m.access_count, "created_at": m.created_at,
                        "last_accessed": m.last_accessed, "context": m.context,
                    }
                    for mid, m in self._memories.items()
                },
                "working_context": self._working_context,
            }
            PALACE_DB.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[MemoryPalace] Save error: {e}")

    def _load_palace(self):
        try:
            if PALACE_DB.exists():
                data = json.loads(PALACE_DB.read_text(encoding="utf-8"))
                for mid, md in data.get("memories", {}).items():
                    mem = Memory(
                        id=md["id"], type=md["type"], content=md["content"],
                        summary=md.get("summary", ""), tags=md.get("tags", []),
                        importance=md.get("importance", 0.5),
                        strength=md.get("strength", 1.0),
                        access_count=md.get("access_count", 0),
                        created_at=md.get("created_at", datetime.now().isoformat()),
                        last_accessed=md.get("last_accessed", datetime.now().isoformat()),
                        context=md.get("context", {}),
                    )
                    self._memories[mid] = mem
                    self._type_index[mem.type].append(mid)
                    for tag in mem.tags:
                        self._tag_index[tag].append(mid)
                self._working_context = data.get("working_context", [])
                logger.info(f"[MemoryPalace] Loaded {len(self._memories)} memories")
        except Exception as e:
            logger.debug(f"[MemoryPalace] Load error: {e}")
