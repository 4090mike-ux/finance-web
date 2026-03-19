"""
JARVIS 고급 오케스트레이터 — Iteration 2
복잡한 장기 태스크를 자율적으로 계획하고 실행하는 시스템
- 태스크 자동 분해 (Task Decomposition)
- 서브에이전트 병렬 실행
- 실행 계획 생성 및 추적
- 실패 복구 및 대안 경로
- 결과 통합 및 보고
"""

import json
import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubTask:
    id: str
    name: str
    description: str
    agent_type: str
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    started_at: float = 0
    completed_at: float = 0
    retries: int = 0


@dataclass
class OrchestratorTask:
    id: str
    goal: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    plan: str = ""
    final_result: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    progress: int = 0  # 0-100


class Orchestrator:
    """
    JARVIS 자율 오케스트레이터
    복잡한 목표를 분해하고 에이전트들을 조율하여 달성
    """

    MAX_RETRIES = 2
    MAX_SUBTASKS = 8

    def __init__(self, llm_manager, agent_manager, tool_executor, memory_manager):
        self.llm = llm_manager
        self.agents = agent_manager
        self.tools = tool_executor
        self.memory = memory_manager
        self._tasks: Dict[str, OrchestratorTask] = {}
        self._running_tasks: Dict[str, threading.Thread] = {}
        self._progress_callbacks: List[Callable] = []
        logger.info("Orchestrator initialized")

    def on_progress(self, callback: Callable):
        """진행 상황 콜백 등록"""
        self._progress_callbacks.append(callback)

    def _emit(self, task_id: str, event: str, data: Dict):
        for cb in self._progress_callbacks:
            try:
                cb(task_id, event, data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    # ── 메인 진입점 ──────────────────────────────────────────────────────────
    def execute_goal(self, goal: str, background: bool = False) -> Dict:
        """
        복잡한 목표 실행
        1. 계획 수립 (LLM)
        2. 서브태스크 분해
        3. 순서대로 (의존성 고려) 실행
        4. 결과 통합
        """
        task_id = f"orch_{int(time.time() * 1000)}"
        task = OrchestratorTask(id=task_id, goal=goal)
        self._tasks[task_id] = task

        if background:
            thread = threading.Thread(target=self._run_task, args=(task,), daemon=True, name=f"JARVIS-Orch-{task_id}")
            self._running_tasks[task_id] = thread
            thread.start()
            return {"task_id": task_id, "status": "started", "goal": goal}
        else:
            return self._run_task(task)

    def _run_task(self, task: OrchestratorTask) -> Dict:
        try:
            # 1단계: 계획
            task.status = TaskStatus.PLANNING
            self._emit(task.id, "planning", {"goal": task.goal})
            plan = self._plan(task.goal)
            task.plan = plan.get("summary", "")
            task.subtasks = plan.get("subtasks", [])

            if not task.subtasks:
                # 단순 태스크 — 직접 실행
                task.status = TaskStatus.RUNNING
                result = self._execute_simple(task.goal)
                task.final_result = result
                task.status = TaskStatus.DONE
                task.progress = 100
                return {"task_id": task.id, "status": "done", "result": result}

            # 2단계: 서브태스크 실행 (의존성 순서)
            task.status = TaskStatus.RUNNING
            completed_results = {}
            total = len(task.subtasks)

            for i, subtask in enumerate(task.subtasks):
                # 의존성 확인
                for dep_id in subtask.depends_on:
                    dep = next((s for s in task.subtasks if s.id == dep_id), None)
                    if dep and dep.status != TaskStatus.DONE:
                        logger.warning(f"Dependency {dep_id} not done, skipping {subtask.id}")
                        subtask.status = TaskStatus.FAILED
                        subtask.error = f"Dependency {dep_id} failed"
                        continue

                self._emit(task.id, "subtask_start", {"name": subtask.name, "index": i, "total": total})
                result = self._run_subtask(subtask, completed_results)
                completed_results[subtask.id] = result
                task.progress = int((i + 1) / total * 90)
                self._emit(task.id, "subtask_done", {"name": subtask.name, "result_preview": str(result)[:200]})

            # 3단계: 결과 통합
            task.progress = 95
            self._emit(task.id, "integrating", {})
            final = self._integrate_results(task.goal, completed_results)
            task.final_result = final
            task.status = TaskStatus.DONE
            task.progress = 100
            task.completed_at = datetime.now().isoformat()
            self._emit(task.id, "done", {"result": final[:500]})

            # 기억에 저장
            self.memory.add_knowledge(
                "orchestrator_result",
                f"Goal: {task.goal}\nResult: {final[:1000]}",
                task.goal,
            )
            return {"task_id": task.id, "status": "done", "result": final, "subtasks": len(task.subtasks)}

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.final_result = f"실행 실패: {e}"
            logger.error(f"Orchestrator task {task.id} failed: {e}")
            self._emit(task.id, "error", {"error": str(e)})
            return {"task_id": task.id, "status": "failed", "error": str(e)}

    # ── 계획 수립 ────────────────────────────────────────────────────────────
    def _plan(self, goal: str) -> Dict:
        """LLM으로 서브태스크 계획 수립"""
        from jarvis.llm.manager import Message

        prompt = f"""목표를 달성하기 위한 실행 계획을 수립하세요.

목표: {goal}

사용 가능한 에이전트:
- research: 웹 검색, 정보 수집, 논문 검색
- code: 코드 작성, 분석, 실행
- system: 파일 관리, 시스템 제어
- planner: 계획 수립, 분석
- summary: 결과 요약, 보고서 작성

규칙:
1. 최대 {self.MAX_SUBTASKS}개 서브태스크
2. 단순한 목표는 1-2개 서브태스크만
3. 의존성을 명시 (depends_on)
4. JSON 형식으로만 반환

{{
  "summary": "계획 요약",
  "complexity": "simple|medium|complex",
  "subtasks": [
    {{
      "id": "t1",
      "name": "서브태스크 이름",
      "description": "상세 설명",
      "agent_type": "research",
      "depends_on": []
    }}
  ]
}}"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="당신은 태스크를 JSON으로 분해하는 플래너입니다. JSON만 출력하세요.", max_tokens=2048)

        try:
            text = response.content.strip()
            # JSON 추출
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(text)

            # SubTask 객체 생성
            subtasks = []
            for i, st in enumerate(data.get("subtasks", [])):
                subtasks.append(SubTask(
                    id=st.get("id", f"t{i+1}"),
                    name=st.get("name", f"태스크 {i+1}"),
                    description=st.get("description", ""),
                    agent_type=st.get("agent_type", "general"),
                    depends_on=st.get("depends_on", []),
                ))

            # 단순 태스크면 서브태스크 없음
            if data.get("complexity") == "simple" or not subtasks:
                return {"summary": data.get("summary", goal), "subtasks": []}

            return {"summary": data.get("summary", goal), "subtasks": subtasks}

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Plan parsing failed: {e}")
            return {"summary": goal, "subtasks": []}

    # ── 서브태스크 실행 ──────────────────────────────────────────────────────
    def _run_subtask(self, subtask: SubTask, context: Dict) -> str:
        """단일 서브태스크 실행 (재시도 포함)"""
        subtask.status = TaskStatus.RUNNING
        subtask.started_at = time.time()

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                # 컨텍스트를 설명에 추가
                task_desc = subtask.description
                if context:
                    prev_results = "\n".join([f"- {k}: {str(v)[:200]}" for k, v in context.items()])
                    task_desc += f"\n\n이전 결과:\n{prev_results}"

                result = self._dispatch_agent(subtask.agent_type, task_desc)
                subtask.status = TaskStatus.DONE
                subtask.result = result
                subtask.completed_at = time.time()
                return result

            except Exception as e:
                subtask.retries = attempt + 1
                logger.warning(f"Subtask {subtask.id} attempt {attempt+1} failed: {e}")
                if attempt >= self.MAX_RETRIES:
                    subtask.status = TaskStatus.FAILED
                    subtask.error = str(e)
                    return f"[실패: {e}]"
                time.sleep(1)

        return "[최대 재시도 초과]"

    def _dispatch_agent(self, agent_type: str, task: str) -> str:
        """에이전트 타입에 따라 실행"""
        from jarvis.agents.agent_manager import AgentType
        try:
            at = AgentType(agent_type)
            result = self.agents.run_agent(task, agent_type=at)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except (ValueError, Exception):
            # 일반 에이전트로 폴백
            try:
                at = AgentType("general")
                return self.agents.run_agent(task, agent_type=at)
            except Exception as e:
                return f"에이전트 실행 실패: {e}"

    def _execute_simple(self, goal: str) -> str:
        """단순 목표 직접 실행"""
        from jarvis.agents.agent_manager import AgentType
        try:
            return self.agents.run_agent(goal, agent_type=AgentType.GENERAL)
        except Exception as e:
            return f"실행 결과: {e}"

    def _integrate_results(self, goal: str, results: Dict) -> str:
        """서브태스크 결과 통합"""
        from jarvis.llm.manager import Message

        results_text = "\n\n".join([
            f"## {k}\n{str(v)[:800]}"
            for k, v in results.items()
        ])

        prompt = f"""다음 서브태스크 결과들을 통합하여 원래 목표에 대한 최종 답변을 작성하세요.

원래 목표: {goal}

서브태스크 결과:
{results_text}

최종 통합 보고서 (한국어, 마크다운 형식):"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="당신은 정보를 통합하고 명확한 보고서를 작성하는 전문가입니다.", max_tokens=4096)
        return response.content

    # ── 태스크 관리 ──────────────────────────────────────────────────────────
    def get_task(self, task_id: str) -> Optional[Dict]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "id": task.id,
            "goal": task.goal,
            "status": task.status.value,
            "progress": task.progress,
            "plan": task.plan,
            "subtasks": [
                {
                    "id": st.id,
                    "name": st.name,
                    "status": st.status.value,
                    "agent": st.agent_type,
                    "result_preview": str(st.result)[:200] if st.result else "",
                }
                for st in task.subtasks
            ],
            "final_result": task.final_result,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }

    def list_tasks(self) -> List[Dict]:
        return [
            {"id": t.id, "goal": t.goal[:80], "status": t.status.value, "progress": t.progress}
            for t in sorted(self._tasks.values(), key=lambda x: x.created_at, reverse=True)
        ]

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.CANCELLED
            return True
        return False

    def get_stats(self) -> Dict:
        total = len(self._tasks)
        by_status = {}
        for t in self._tasks.values():
            s = t.status.value
            by_status[s] = by_status.get(s, 0) + 1
        return {"total_tasks": total, "by_status": by_status}
