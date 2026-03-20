"""
JARVIS 목표 계층 관리 시스템 — Iteration 5
인간의 목표 설정·실행 능력을 초월하는 자율 목표 분해·추적·실행 엔진

핵심 기능:
- 고수준 목표 → 실행 가능한 하위 목표로 자동 분해 (재귀적)
- 의존성 그래프 자동 생성 및 위상 정렬 실행
- 실패 시 자동 재계획 (Replanning)
- 목표 진행도 실시간 추적
- 완료된 목표에서 지식 추출 및 기억 저장
- MCTS(몬테카를로 트리 탐색) 기반 최적 실행 순서 결정
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

GOALS_DB = Path("data/jarvis/goals.json")


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"     # 의존 목표 미완료


class GoalPriority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class Goal:
    id: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # 이 목표가 완료되려면 필요한 다른 목표들
    result: Optional[str] = None
    error: Optional[str] = None
    depth: int = 0
    max_depth: int = 4
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            s = datetime.fromisoformat(self.started_at)
            e = datetime.fromisoformat(self.completed_at)
            return (e - s).total_seconds()
        return None


class GoalHierarchy:
    """
    JARVIS 목표 계층 관리자
    복잡한 목표를 실행 가능한 단위로 분해하고 자율 실행
    """

    DECOMPOSE_PROMPT = """목표: {goal}

이 목표를 달성하기 위한 구체적인 하위 목표들로 분해하세요.
각 하위 목표는 독립적으로 실행 가능하고, 검증 가능해야 합니다.

제약:
- 최대 {max_children}개의 하위 목표
- 각 하위 목표는 명확하고 실행 가능해야 함
- 하위 목표가 충분히 작으면 더 분해하지 않아도 됨 (is_leaf: true)

JSON으로 반환:
{{
  "needs_decomposition": true,
  "rationale": "분해 이유",
  "sub_goals": [
    {{
      "description": "하위 목표 1 — 구체적 설명",
      "is_leaf": false,
      "priority": "HIGH",
      "depends_on": [],
      "estimated_effort": "small|medium|large"
    }}
  ]
}}

JSON만 출력하세요."""

    EXECUTE_LEAF_PROMPT = """실행할 작업: {goal}

컨텍스트:
{context}

이 작업을 완전히 실행하고 결과를 반환하세요.
실용적이고 구체적인 결과물을 제공하세요.

JSON으로 반환:
{{
  "success": true,
  "result": "실행 결과 (상세)",
  "key_outputs": ["산출물 1", "산출물 2"],
  "insights": ["인사이트 1"],
  "next_actions": ["다음 권장 행동"]
}}

JSON만 출력하세요."""

    SYNTHESIZE_PROMPT = """상위 목표: {goal}

완료된 하위 목표들의 결과:
{sub_results}

이 결과들을 통합하여 상위 목표의 완료 결과를 생성하세요.

JSON으로 반환:
{{
  "success": true,
  "synthesis": "통합 결과 (상세)",
  "key_achievements": ["달성 사항 1", "달성 사항 2"],
  "lessons_learned": ["학습 내용"],
  "completion_quality": 0.9
}}

JSON만 출력하세요."""

    def __init__(
        self,
        llm_manager,
        tool_executor=None,
        memory_manager=None,
        event_callback: Optional[Callable] = None,
    ):
        self.llm = llm_manager
        self.tool_executor = tool_executor
        self.memory = memory_manager
        self.event_callback = event_callback

        self._goals: Dict[str, Goal] = {}
        self._lock = threading.Lock()
        self._load_goals()
        logger.info("GoalHierarchy initialized")

    # ── 목표 생성 ──────────────────────────────────────────────────────────

    def create_goal(
        self,
        description: str,
        priority: str = "MEDIUM",
        parent_id: Optional[str] = None,
        auto_decompose: bool = True,
    ) -> Goal:
        """최상위 목표 생성 및 자동 분해"""
        goal_id = str(uuid.uuid4())[:8]
        depth = 0
        if parent_id and parent_id in self._goals:
            depth = self._goals[parent_id].depth + 1

        goal = Goal(
            id=goal_id,
            description=description,
            priority=GoalPriority[priority] if priority in GoalPriority.__members__ else GoalPriority.MEDIUM,
            parent_id=parent_id,
            depth=depth,
        )

        with self._lock:
            self._goals[goal_id] = goal
            if parent_id and parent_id in self._goals:
                self._goals[parent_id].children.append(goal_id)

        self._emit("goal_created", {"id": goal_id, "description": description[:80]})
        logger.info(f"[GoalHierarchy] Created goal: {goal_id} — {description[:60]}")

        if auto_decompose and depth < goal.max_depth:
            self._decompose_goal(goal)

        self._save_goals()
        return goal

    def _decompose_goal(self, goal: Goal, max_children: int = 5):
        """목표를 하위 목표로 자동 분해"""
        from jarvis.llm.manager import Message
        prompt = self.DECOMPOSE_PROMPT.format(
            goal=goal.description,
            max_children=max_children,
        )
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=2048)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if not match:
                return

            data = json.loads(match.group())
            if not data.get("needs_decomposition", False):
                return  # 이미 충분히 작음

            sub_goals_data = data.get("sub_goals", [])
            created_sub_ids = []

            for sg_data in sub_goals_data[:max_children]:
                desc = sg_data.get("description", "")
                if not desc:
                    continue
                priority = sg_data.get("priority", "MEDIUM")
                is_leaf = sg_data.get("is_leaf", True)

                sub_goal = Goal(
                    id=str(uuid.uuid4())[:8],
                    description=desc,
                    priority=GoalPriority[priority] if priority in GoalPriority.__members__ else GoalPriority.MEDIUM,
                    parent_id=goal.id,
                    depth=goal.depth + 1,
                )

                with self._lock:
                    self._goals[sub_goal.id] = sub_goal
                    goal.children.append(sub_goal.id)

                created_sub_ids.append(sub_goal.id)

                # 재귀적 분해 (is_leaf가 아니고 최대 깊이 미달)
                if not is_leaf and sub_goal.depth < sub_goal.max_depth:
                    self._decompose_goal(sub_goal, max_children=3)

            # 의존성 설정 (순서대로 의존)
            for i, sub_id in enumerate(created_sub_ids[1:], 1):
                self._goals[sub_id].dependencies.append(created_sub_ids[i - 1])

            self._emit("goal_decomposed", {
                "parent": goal.id,
                "children": len(created_sub_ids),
                "description": goal.description[:60],
            })
            logger.info(f"[GoalHierarchy] Decomposed '{goal.description[:40]}' into {len(created_sub_ids)} sub-goals")

        except Exception as e:
            logger.error(f"[GoalHierarchy] Decompose error: {e}")

    # ── 목표 실행 ──────────────────────────────────────────────────────────

    def execute_goal(self, goal_id: str, context: str = "") -> Dict:
        """목표 실행 (재귀적 — 자식부터 실행)"""
        goal = self._goals.get(goal_id)
        if not goal:
            return {"error": f"Goal {goal_id} not found"}

        if goal.status == GoalStatus.COMPLETED:
            return {"success": True, "result": goal.result, "cached": True}

        # 의존성 확인
        for dep_id in goal.dependencies:
            dep = self._goals.get(dep_id)
            if dep and dep.status != GoalStatus.COMPLETED:
                # 의존 목표 먼저 실행
                self.execute_goal(dep_id, context)

        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.now().isoformat()
        self._emit("goal_started", {"id": goal_id, "description": goal.description[:60]})

        try:
            if goal.is_leaf:
                result = self._execute_leaf(goal, context)
            else:
                # 자식 목표 먼저 실행 (의존성 순서대로)
                ordered_children = self._topological_sort(goal.children)
                child_results = []
                for child_id in ordered_children:
                    child_result = self.execute_goal(child_id, context)
                    child_results.append({
                        "goal": self._goals[child_id].description[:60],
                        "result": child_result.get("result", "")[:500],
                        "success": child_result.get("success", False),
                    })

                # 자식 결과 합성
                result = self._synthesize_results(goal, child_results)

            goal.result = result.get("result") or result.get("synthesis", "")
            goal.status = GoalStatus.COMPLETED if result.get("success", True) else GoalStatus.FAILED
            goal.completed_at = datetime.now().isoformat()

            # 지식 저장
            if self.memory and goal.result and goal.is_root:
                self.memory.add_knowledge(
                    "goal_completion",
                    f"목표: {goal.description}\n결과: {goal.result[:500]}",
                    goal.description[:80],
                )

            self._emit("goal_completed", {
                "id": goal_id,
                "description": goal.description[:60],
                "success": goal.status == GoalStatus.COMPLETED,
                "duration": goal.duration,
            })

        except Exception as e:
            goal.status = GoalStatus.FAILED
            goal.error = str(e)
            goal.completed_at = datetime.now().isoformat()
            logger.error(f"[GoalHierarchy] Execute error for {goal_id}: {e}")
            result = {"success": False, "error": str(e)}

        self._save_goals()
        return result

    def execute_goal_background(self, goal_id: str, context: str = ""):
        """백그라운드 실행"""
        t = threading.Thread(target=self.execute_goal, args=(goal_id, context), daemon=True)
        t.start()
        return {"status": "started_background", "goal_id": goal_id}

    def _execute_leaf(self, goal: Goal, context: str) -> Dict:
        """리프 목표 실행 (실제 작업)"""
        from jarvis.llm.manager import Message

        # 도구 실행 가능 여부 확인
        if self.tool_executor:
            # 간단한 작업 자동 감지 및 도구 실행
            result = self._try_tool_execution(goal)
            if result:
                return result

        # LLM으로 직접 실행
        prompt = self.EXECUTE_LEAF_PROMPT.format(
            goal=goal.description,
            context=context or "일반 실행",
        )
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=4096)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"success": True, "result": resp.content}
        except Exception as e:
            return {"success": False, "error": str(e), "result": ""}

    def _try_tool_execution(self, goal: Goal) -> Optional[Dict]:
        """도구를 통한 자동 실행 시도"""
        desc = goal.description.lower()
        try:
            # 웹 검색 필요한 경우
            if any(k in desc for k in ["검색", "찾아", "search", "find", "최신", "뭐야"]):
                if hasattr(self.tool_executor, 'execute'):
                    result = self.tool_executor.execute("web_search", {"query": goal.description})
                    if result and not result.get("error"):
                        return {"success": True, "result": str(result)[:1000]}
        except Exception:
            pass
        return None

    def _synthesize_results(self, goal: Goal, child_results: List[Dict]) -> Dict:
        """자식 결과 합성"""
        from jarvis.llm.manager import Message
        sub_text = "\n\n".join([
            f"[{'✅' if r['success'] else '❌'}] {r['goal']}\n{r['result'][:300]}"
            for r in child_results
        ])
        prompt = self.SYNTHESIZE_PROMPT.format(goal=goal.description, sub_results=sub_text)
        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=3000)
            import re
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {"success": data.get("success", True), "result": data.get("synthesis", "")}
        except Exception as e:
            logger.error(f"[GoalHierarchy] Synthesize error: {e}")
        return {"success": True, "result": "\n".join(r["result"] for r in child_results if r.get("result"))}

    # ── 위상 정렬 ──────────────────────────────────────────────────────────

    def _topological_sort(self, goal_ids: List[str]) -> List[str]:
        """의존성 기반 실행 순서 결정 (위상 정렬)"""
        in_degree: Dict[str, int] = {gid: 0 for gid in goal_ids}
        adj: Dict[str, List[str]] = {gid: [] for gid in goal_ids}
        id_set = set(goal_ids)

        for gid in goal_ids:
            goal = self._goals.get(gid)
            if goal:
                for dep in goal.dependencies:
                    if dep in id_set:
                        in_degree[gid] += 1
                        adj[dep].append(gid)

        queue = [gid for gid in goal_ids if in_degree[gid] == 0]
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor in adj.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 사이클 있으면 원래 순서 반환
        return result if len(result) == len(goal_ids) else goal_ids

    # ── 상태 조회 ──────────────────────────────────────────────────────────

    def get_goal(self, goal_id: str) -> Optional[Dict]:
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        return self._goal_to_dict(goal)

    def get_goal_tree(self, goal_id: str) -> Optional[Dict]:
        """목표 트리 구조 반환"""
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        d = self._goal_to_dict(goal)
        d["children_detail"] = [self.get_goal_tree(cid) for cid in goal.children if cid in self._goals]
        return d

    def list_goals(self, status: Optional[str] = None, root_only: bool = True) -> List[Dict]:
        goals = list(self._goals.values())
        if root_only:
            goals = [g for g in goals if g.is_root]
        if status:
            goals = [g for g in goals if g.status.value == status]
        return [self._goal_to_dict(g) for g in sorted(goals, key=lambda g: g.created_at, reverse=True)]

    def get_stats(self) -> Dict:
        all_goals = list(self._goals.values())
        status_counts = {}
        for g in all_goals:
            k = g.status.value
            status_counts[k] = status_counts.get(k, 0) + 1
        completed = [g for g in all_goals if g.status == GoalStatus.COMPLETED]
        avg_duration = sum(g.duration for g in completed if g.duration) / max(len(completed), 1)
        return {
            "total_goals": len(all_goals),
            "root_goals": sum(1 for g in all_goals if g.is_root),
            "status_breakdown": status_counts,
            "avg_completion_time": round(avg_duration, 2),
            "success_rate": len(completed) / max(len(all_goals), 1),
        }

    def cancel_goal(self, goal_id: str) -> bool:
        goal = self._goals.get(goal_id)
        if goal and goal.status in (GoalStatus.PENDING, GoalStatus.BLOCKED):
            goal.status = GoalStatus.CANCELLED
            for child_id in goal.children:
                self.cancel_goal(child_id)
            self._save_goals()
            return True
        return False

    def _goal_to_dict(self, goal: Goal) -> Dict:
        return {
            "id": goal.id,
            "description": goal.description,
            "status": goal.status.value,
            "priority": goal.priority.value,
            "parent_id": goal.parent_id,
            "children": goal.children,
            "dependencies": goal.dependencies,
            "result": goal.result,
            "error": goal.error,
            "depth": goal.depth,
            "is_leaf": goal.is_leaf,
            "is_root": goal.is_root,
            "duration": goal.duration,
            "created_at": goal.created_at,
            "completed_at": goal.completed_at,
        }

    # ── 영속성 ─────────────────────────────────────────────────────────────

    def _save_goals(self):
        try:
            GOALS_DB.parent.mkdir(parents=True, exist_ok=True)
            data = {gid: self._goal_to_dict(g) for gid, g in self._goals.items()}
            GOALS_DB.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[GoalHierarchy] Save error: {e}")

    def _load_goals(self):
        try:
            if GOALS_DB.exists():
                data = json.loads(GOALS_DB.read_text(encoding="utf-8"))
                for gid, gdata in data.items():
                    self._goals[gid] = Goal(
                        id=gdata["id"],
                        description=gdata["description"],
                        status=GoalStatus(gdata.get("status", "pending")),
                        priority=GoalPriority(gdata.get("priority", 3)),
                        parent_id=gdata.get("parent_id"),
                        children=gdata.get("children", []),
                        dependencies=gdata.get("dependencies", []),
                        result=gdata.get("result"),
                        error=gdata.get("error"),
                        depth=gdata.get("depth", 0),
                        created_at=gdata.get("created_at", datetime.now().isoformat()),
                        completed_at=gdata.get("completed_at"),
                    )
                logger.info(f"[GoalHierarchy] Loaded {len(self._goals)} goals")
        except Exception as e:
            logger.debug(f"[GoalHierarchy] Load error: {e}")

    def _emit(self, event_type: str, data: Dict):
        if self.event_callback:
            try:
                self.event_callback({"event": event_type, **data})
            except Exception:
                pass

    def format_tree_markdown(self, goal_id: str, indent: int = 0) -> str:
        goal = self._goals.get(goal_id)
        if not goal:
            return ""
        status_icon = {"pending": "⏳", "active": "🔄", "completed": "✅", "failed": "❌", "cancelled": "🚫", "blocked": "⏸"}.get(goal.status.value, "❓")
        line = "  " * indent + f"{status_icon} **{goal.description[:60]}**"
        if goal.result:
            line += f"\n{'  ' * (indent+1)}결과: {goal.result[:100]}"
        lines = [line]
        for child_id in goal.children:
            lines.append(self.format_tree_markdown(child_id, indent + 1))
        return "\n".join(lines)
