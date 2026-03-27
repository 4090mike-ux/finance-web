"""
JARVIS 장기 자율 실행기 — Iteration 12
복잡한 다단계 목표를 자율적으로 계획하고 끝까지 실행한다

영감:
  - BabyAGI / AutoGPT (자율 태스크 루프)
  - Voyager (Minecraft 자율 에이전트)
  - DEPENDENCIES (체인 오브 씽킹 + 실행)
  - ReAct (Reason + Act 루프)
  - Hierarchical Task Networks (HTN)

핵심 개념:
  인간이 "X를 해줘"라고 말하면 JARVIS는 혼자서:
  1. X를 위한 하위 태스크 트리를 생성한다
  2. 각 태스크를 실제 도구로 실행한다
  3. 결과를 검증하고 실패 시 재계획한다
  4. 최종 목표 달성까지 자율 반복한다
  사람은 최종 결과만 받는다
"""

import json
import time
import uuid
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"       # 순서대로
    PARALLEL = "parallel"           # 병렬
    ADAPTIVE = "adaptive"           # 결과에 따라 적응


@dataclass
class SubTask:
    """하위 태스크 단위"""
    task_id: str
    title: str
    description: str
    tool_hint: str                  # 어떤 도구/모듈 사용 추천
    dependencies: List[str]         # 선행 task_id 목록
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    started_at: float = 0.0
    finished_at: float = 0.0
    priority: int = 5               # 1(높음) ~ 10(낮음)

    @property
    def duration(self) -> float:
        if self.finished_at and self.started_at:
            return self.finished_at - self.started_at
        return 0.0

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "tool_hint": self.tool_hint,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result[:200] if self.result else None,
            "error": self.error,
            "attempts": self.attempts,
            "duration": round(self.duration, 2),
            "priority": self.priority,
        }


@dataclass
class ExecutionPlan:
    """전체 실행 계획"""
    plan_id: str
    goal: str
    strategy: ExecutionStrategy
    tasks: List[SubTask] = field(default_factory=list)
    context: Dict = field(default_factory=dict)   # 실행 중 공유 컨텍스트
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    final_summary: str = ""
    success: bool = False

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED))

    @property
    def progress_pct(self) -> float:
        if not self.tasks:
            return 0.0
        return self.completed_tasks / self.total_tasks * 100

    def get_ready_tasks(self) -> List[SubTask]:
        """실행 준비된 태스크 (의존성 완료 + PENDING 상태)"""
        completed_ids = {
            t.task_id for t in self.tasks
            if t.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED)
        }
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "strategy": self.strategy.value,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "progress_pct": round(self.progress_pct, 1),
            "tasks": [t.to_dict() for t in self.tasks],
            "final_summary": self.final_summary,
            "success": self.success,
        }


# ════════════════════════════════════════════════════════════════
# 장기 자율 실행기
# ════════════════════════════════════════════════════════════════

class LongHorizonExecutor:
    """
    JARVIS 장기 자율 실행기
    복잡한 목표를 하위 태스크로 분해하고 자율 실행한다
    최대 50개 하위 태스크, 최대 실행 시간 10분
    """

    MAX_TASKS = 50
    MAX_EXECUTION_TIME = 600       # 10분
    MAX_CONCURRENT_TASKS = 3       # 병렬 최대

    def __init__(self, llm_manager, jarvis_engine=None,
                 event_callback: Optional[Callable] = None):
        self.llm = llm_manager
        self.jarvis = jarvis_engine   # 메인 엔진 참조 (도구 사용용)
        self._event_cb = event_callback
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._history: List[ExecutionPlan] = []
        self._lock = threading.Lock()
        self._stats = {
            "plans_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_goals_achieved": 0,
        }
        logger.info("LongHorizonExecutor initialized — HTN planner ready")

    # ── 핵심 API ─────────────────────────────────────────────────

    def execute_goal(self, goal: str, context: Dict = None,
                     strategy: str = "adaptive",
                     blocking: bool = True) -> ExecutionPlan:
        """
        목표 → 계획 생성 → 자율 실행
        blocking=True면 완료까지 대기 (최대 MAX_EXECUTION_TIME초)
        blocking=False면 백그라운드에서 실행
        """
        plan = self._create_plan(goal, context or {}, strategy)

        with self._lock:
            self._active_plans[plan.plan_id] = plan

        self._emit("plan_created", {
            "plan_id": plan.plan_id,
            "goal": goal,
            "task_count": plan.total_tasks,
        })

        if blocking:
            self._execute_plan(plan)
            return plan
        else:
            thread = threading.Thread(
                target=self._execute_plan,
                args=(plan,),
                daemon=True,
            )
            thread.start()
            return plan

    def get_plan_status(self, plan_id: str) -> Optional[Dict]:
        plan = self._active_plans.get(plan_id)
        if plan:
            return plan.to_dict()
        for p in self._history:
            if p.plan_id == plan_id:
                return p.to_dict()
        return None

    def abort_plan(self, plan_id: str) -> bool:
        plan = self._active_plans.get(plan_id)
        if not plan:
            return False
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
        plan.finished_at = time.time()
        plan.final_summary = "사용자 요청으로 중단됨"
        return True

    # ── 계획 생성 ────────────────────────────────────────────────

    def _create_plan(self, goal: str, context: Dict,
                     strategy_str: str) -> ExecutionPlan:
        """LLM으로 실행 계획 생성"""
        strategy_map = {
            "sequential": ExecutionStrategy.SEQUENTIAL,
            "parallel": ExecutionStrategy.PARALLEL,
            "adaptive": ExecutionStrategy.ADAPTIVE,
        }
        strategy = strategy_map.get(strategy_str, ExecutionStrategy.ADAPTIVE)

        plan_prompt = f"""당신은 AI 태스크 플래너입니다. 다음 목표를 달성하기 위한 구체적인 실행 계획을 수립하세요.

목표: {goal}
컨텍스트: {json.dumps(context, ensure_ascii=False)}

가용 도구:
- web_search: 인터넷 검색
- github_search: GitHub 코드/레포 검색
- arxiv_search: 학술 논문 검색
- code_execute: Python 코드 실행
- file_read: 파일 읽기
- file_write: 파일 쓰기
- llm_generate: LLM으로 텍스트 생성/분석
- memory_search: 장기 메모리 검색
- data_analysis: 데이터 분석 (pandas)
- image_analyze: 이미지 분석 (Vision)

규칙:
1. 목표를 최대 10개의 구체적 하위 태스크로 분해하세요
2. 각 태스크는 단일하고 검증 가능해야 합니다
3. 의존성을 명확히 지정하세요 (이전 task_id 참조)
4. 첫 태스크는 dependencies가 빈 배열이어야 합니다

JSON으로만 응답:
{{
  "tasks": [
    {{
      "task_id": "t1",
      "title": "간단한 제목",
      "description": "무엇을 어떻게 할지 구체적 설명",
      "tool_hint": "web_search",
      "dependencies": [],
      "priority": 1
    }},
    {{
      "task_id": "t2",
      "title": "제목2",
      "description": "설명2",
      "tool_hint": "llm_generate",
      "dependencies": ["t1"],
      "priority": 2
    }}
  ]
}}"""

        try:
            response = self.llm.generate(plan_prompt, max_tokens=2000)
            if isinstance(response, dict):
                response = response.get("content", "")

            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                raise ValueError("JSON 파싱 실패")

            tasks_raw = plan_data.get("tasks", [])
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}, using fallback")
            tasks_raw = [
                {"task_id": "t1", "title": "정보 수집",
                 "description": f"'{goal}' 관련 정보 수집", "tool_hint": "web_search",
                 "dependencies": [], "priority": 1},
                {"task_id": "t2", "title": "분석 및 처리",
                 "description": "수집된 정보 분석", "tool_hint": "llm_generate",
                 "dependencies": ["t1"], "priority": 2},
                {"task_id": "t3", "title": "결과 정리",
                 "description": "최종 결과 정리 및 요약", "tool_hint": "llm_generate",
                 "dependencies": ["t2"], "priority": 3},
            ]

        tasks = []
        for raw in tasks_raw[:self.MAX_TASKS]:
            task = SubTask(
                task_id=raw.get("task_id", str(uuid.uuid4())[:6]),
                title=raw.get("title", "태스크"),
                description=raw.get("description", ""),
                tool_hint=raw.get("tool_hint", "llm_generate"),
                dependencies=raw.get("dependencies", []),
                priority=raw.get("priority", 5),
            )
            tasks.append(task)

        plan = ExecutionPlan(
            plan_id=str(uuid.uuid4())[:10],
            goal=goal,
            strategy=strategy,
            tasks=tasks,
            context=context,
        )
        logger.info(f"Plan created: {plan.plan_id} ({len(tasks)} tasks) for '{goal[:40]}'")
        return plan

    # ── 실행 엔진 ────────────────────────────────────────────────

    def _execute_plan(self, plan: ExecutionPlan):
        """계획 실행 메인 루프"""
        plan.started_at = time.time()
        deadline = plan.started_at + self.MAX_EXECUTION_TIME

        logger.info(f"Executing plan {plan.plan_id}: '{plan.goal[:40]}'")

        while time.time() < deadline:
            ready = plan.get_ready_tasks()
            if not ready:
                # 모든 태스크 완료 or 막힘 체크
                pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                running = [t for t in plan.tasks if t.status == TaskStatus.RUNNING]
                if not pending and not running:
                    break
                if not running:
                    # 막힌 상태 — 실패 태스크로 의존성 해제 시도
                    self._unblock_stuck_tasks(plan)
                    ready = plan.get_ready_tasks()
                    if not ready:
                        break
                else:
                    time.sleep(0.5)
                    continue

            if plan.strategy == ExecutionStrategy.PARALLEL:
                # 병렬 실행 (최대 MAX_CONCURRENT_TASKS)
                batch = ready[:self.MAX_CONCURRENT_TASKS]
                threads = []
                for task in batch:
                    t = threading.Thread(
                        target=self._execute_task,
                        args=(task, plan),
                        daemon=True,
                    )
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join(timeout=120)
            else:
                # 순서대로 (SEQUENTIAL / ADAPTIVE)
                task = min(ready, key=lambda t: t.priority)
                self._execute_task(task, plan)

            # 진행 상황 브로드캐스트
            self._emit("plan_progress", {
                "plan_id": plan.plan_id,
                "progress": plan.progress_pct,
                "completed": plan.completed_tasks,
                "total": plan.total_tasks,
            })

        # 실행 완료
        plan.finished_at = time.time()
        plan.success = all(
            t.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED)
            for t in plan.tasks
        )
        plan.final_summary = self._generate_summary(plan)

        with self._lock:
            self._active_plans.pop(plan.plan_id, None)
            self._history.append(plan)
            self._stats["plans_executed"] += 1
            if plan.success:
                self._stats["total_goals_achieved"] += 1

        self._emit("plan_completed", {
            "plan_id": plan.plan_id,
            "success": plan.success,
            "duration": round(plan.finished_at - plan.started_at, 1),
            "summary": plan.final_summary[:300],
        })
        logger.info(f"Plan {plan.plan_id} {'SUCCEEDED' if plan.success else 'PARTIALLY FAILED'} "
                    f"({plan.completed_tasks}/{plan.total_tasks} tasks, "
                    f"{plan.finished_at - plan.started_at:.1f}s)")

    def _execute_task(self, task: SubTask, plan: ExecutionPlan):
        """단일 태스크 실행"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.attempts += 1

        self._emit("task_started", {
            "plan_id": plan.plan_id,
            "task_id": task.task_id,
            "title": task.title,
        })

        try:
            result = self._dispatch_task(task, plan)
            task.result = str(result)[:2000]
            task.status = TaskStatus.SUCCEEDED

            # 결과를 공유 컨텍스트에 저장
            plan.context[task.task_id] = task.result

            with self._lock:
                self._stats["tasks_completed"] += 1

        except Exception as e:
            task.error = str(e)
            if task.attempts < task.max_attempts:
                task.status = TaskStatus.RETRYING
                logger.warning(f"Task {task.task_id} failed (attempt {task.attempts}): {e}")
                time.sleep(2)
                task.status = TaskStatus.PENDING  # 재시도 대기
            else:
                task.status = TaskStatus.FAILED
                with self._lock:
                    self._stats["tasks_failed"] += 1
                logger.error(f"Task {task.task_id} FAILED after {task.attempts} attempts: {e}")

        task.finished_at = time.time()
        self._emit("task_finished", {
            "plan_id": plan.plan_id,
            "task_id": task.task_id,
            "status": task.status.value,
            "duration": round(task.duration, 2),
        })

    def _dispatch_task(self, task: SubTask, plan: ExecutionPlan) -> str:
        """도구 힌트에 따라 실제 도구 호출"""
        hint = task.tool_hint.lower()

        # 이전 태스크 결과를 컨텍스트로 제공
        prev_results = {
            dep: plan.context.get(dep, "")[:500]
            for dep in task.dependencies
        }
        context_str = "\n".join([f"[{k}] {v}" for k, v in prev_results.items()])

        if hint == "web_search":
            return self._tool_web_search(task.description, context_str)
        elif hint == "github_search":
            return self._tool_github_search(task.description)
        elif hint == "arxiv_search":
            return self._tool_arxiv_search(task.description)
        elif hint == "code_execute":
            return self._tool_code_execute(task.description, context_str)
        elif hint == "file_read":
            return self._tool_file_read(task.description)
        elif hint == "file_write":
            return self._tool_file_write(task.description, context_str)
        elif hint == "data_analysis":
            return self._tool_data_analysis(task.description, context_str)
        elif hint == "memory_search":
            return self._tool_memory_search(task.description)
        else:  # llm_generate, default
            return self._tool_llm_generate(task, context_str, plan.goal)

    # ── 도구 구현 ────────────────────────────────────────────────

    def _tool_web_search(self, description: str, context: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "web") and self.jarvis.web:
            try:
                results = self.jarvis.web.search(description, num_results=5)
                if isinstance(results, list):
                    return "\n".join([str(r) for r in results[:5]])
                return str(results)
            except Exception as e:
                pass
        return self._tool_llm_generate_simple(f"웹 검색 시뮬레이션: {description}")

    def _tool_github_search(self, description: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "web") and self.jarvis.web:
            try:
                results = self.jarvis.web.search_github(description)
                return str(results)[:1000]
            except Exception as e:
                pass
        return f"GitHub 검색: {description} (도구 없음 — LLM 대체)"

    def _tool_arxiv_search(self, description: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "web") and self.jarvis.web:
            try:
                results = self.jarvis.web.search_arxiv(description)
                return str(results)[:1000]
            except Exception as e:
                pass
        return f"ArXiv 검색: {description} (도구 없음)"

    def _tool_code_execute(self, description: str, context: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "executor") and self.jarvis.executor:
            code_prompt = f"다음 작업을 위한 Python 코드를 작성하세요:\n{description}\n컨텍스트:\n{context}"
            try:
                code = self.llm.generate(code_prompt, max_tokens=500)
                if isinstance(code, dict):
                    code = code.get("content", "")
                import re
                code = re.sub(r"```python\s*", "", code)
                code = re.sub(r"```\s*", "", code)
                result = self.jarvis.executor.execute(code.strip())
                return str(result)[:1000]
            except Exception as e:
                return f"코드 실행 실패: {e}"
        return "코드 실행기 없음"

    def _tool_file_read(self, description: str) -> str:
        import re
        path_match = re.search(r"(?:파일|file|path)[:\s]+([^\s,]+)", description)
        if not path_match:
            return "파일 경로를 찾을 수 없음"
        path = path_match.group(1)
        try:
            from pathlib import Path
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            return content[:2000]
        except Exception as e:
            return f"파일 읽기 실패: {e}"

    def _tool_file_write(self, description: str, context: str) -> str:
        prompt = f"다음 작업에 맞는 파일 내용을 생성하세요:\n{description}\n컨텍스트:\n{context}"
        try:
            content = self.llm.generate(prompt, max_tokens=1000)
            if isinstance(content, dict):
                content = content.get("content", "")
            import re
            path_match = re.search(r"(?:저장|파일명|filename)[:\s]+([^\s,]+)", description)
            if path_match:
                path = path_match.group(1)
                from pathlib import Path
                Path(path).write_text(content, encoding="utf-8")
                return f"파일 저장 완료: {path}"
            return f"생성된 내용:\n{content[:500]}"
        except Exception as e:
            return f"파일 쓰기 실패: {e}"

    def _tool_data_analysis(self, description: str, context: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "data_scientist"):
            ds = self.jarvis.data_scientist
            datasets = ds.list_datasets()
            if datasets:
                result = ds.ask(datasets[0]["name"], description)
                return result.answer
        return self._tool_llm_generate_simple(f"데이터 분석 시뮬레이션: {description}")

    def _tool_memory_search(self, description: str) -> str:
        if self.jarvis and hasattr(self.jarvis, "memory") and self.jarvis.memory:
            try:
                results = self.jarvis.memory.search(description, n_results=5)
                return str(results)[:1000]
            except Exception as e:
                pass
        return f"메모리 검색: {description} (결과 없음)"

    def _tool_llm_generate(self, task: SubTask, context: str, goal: str) -> str:
        prompt = f"""전체 목표: {goal}

현재 태스크: {task.title}
상세 설명: {task.description}

이전 단계 결과:
{context}

위 태스크를 최선을 다해 수행하고 결과를 제공하세요.
간결하고 구체적으로 (최대 500자)"""
        result = self.llm.generate(prompt, max_tokens=600)
        if isinstance(result, dict):
            result = result.get("content", "")
        return str(result)

    def _tool_llm_generate_simple(self, prompt: str) -> str:
        result = self.llm.generate(prompt, max_tokens=300)
        if isinstance(result, dict):
            result = result.get("content", "")
        return str(result)

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _unblock_stuck_tasks(self, plan: ExecutionPlan):
        """실패한 의존성으로 막힌 태스크 해제 (SKIPPED 처리)"""
        failed_ids = {t.task_id for t in plan.tasks if t.status == TaskStatus.FAILED}
        for task in plan.tasks:
            if task.status == TaskStatus.PENDING:
                if any(dep in failed_ids for dep in task.dependencies):
                    task.status = TaskStatus.SKIPPED
                    task.error = f"의존 태스크 실패로 건너뜀: {task.dependencies}"

    def _generate_summary(self, plan: ExecutionPlan) -> str:
        """실행 완료 후 최종 요약 생성"""
        succeeded = [t for t in plan.tasks if t.status == TaskStatus.SUCCEEDED]
        failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        results_text = "\n".join([
            f"[{t.task_id}] {t.title}: {(t.result or '')[:150]}"
            for t in succeeded[:8]
        ])

        prompt = f"""다음 자율 실행 결과를 3-4문장으로 요약하세요.

목표: {plan.goal}
성공: {len(succeeded)}개 태스크
실패: {len(failed)}개 태스크

주요 결과:
{results_text}

핵심 성과와 결론을 중심으로 요약하세요."""

        try:
            summary = self.llm.generate(prompt, max_tokens=300)
            if isinstance(summary, dict):
                summary = summary.get("content", "")
            return summary
        except Exception:
            return (f"목표 '{plan.goal}' 실행 완료. "
                    f"{len(succeeded)}/{len(plan.tasks)} 태스크 성공.")

    def _emit(self, event_type: str, data: Dict):
        if self._event_cb:
            try:
                self._event_cb({
                    "type": f"executor_{event_type}",
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
                "active_plans": len(self._active_plans),
                "history_count": len(self._history),
                "active_plan_ids": list(self._active_plans.keys()),
            }

    def get_history(self, n: int = 10) -> List[Dict]:
        return [p.to_dict() for p in self._history[-n:]]
