"""
JARVIS 멀티 에이전트 시스템
각 에이전트는 특정 도메인에 특화된 전문가
- ResearchAgent: 정보 수집 및 분석
- CodeAgent: 코드 작성, 리뷰, 실행
- SystemAgent: 시스템 모니터링 및 관리
- PlannerAgent: 복잡한 태스크 계획 및 분해
- SummaryAgent: 정보 요약 및 보고서 작성
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    SYSTEM = "system"
    PLANNER = "planner"
    SUMMARY = "summary"
    GENERAL = "general"


@dataclass
class AgentTask:
    task_id: str
    agent_type: AgentType
    description: str
    context: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, running, done, error
    result: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0


class BaseAgent:
    """기본 에이전트 클래스"""

    def __init__(self, name: str, system_prompt: str, llm_manager):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm_manager
        self.task_history = []

    def run(self, task: str, context: Dict = None) -> str:
        raise NotImplementedError


class ResearchAgent(BaseAgent):
    """연구/정보 수집 전문 에이전트"""

    SYSTEM_PROMPT = """당신은 JARVIS의 Research Agent입니다.
임무: 웹 검색, GitHub, ArXiv, Wikipedia에서 정보를 수집하고 분석합니다.
- 다양한 소스에서 정보를 수집합니다
- 사실과 의견을 구분합니다
- 최신 정보를 우선시합니다
- 한국어로 명확하게 보고합니다"""

    def __init__(self, llm_manager, web_intel):
        super().__init__("ResearchAgent", self.SYSTEM_PROMPT, llm_manager)
        self.web = web_intel

    def run(self, task: str, context: Dict = None) -> str:
        """연구 태스크 실행"""
        from jarvis.llm.manager import Message

        # 정보 수집
        search_results = self.web.search_web(task, max_results=5)
        arxiv_results = []

        # AI/ML 관련 쿼리면 논문도 검색
        ai_keywords = ["ai", "ml", "머신러닝", "딥러닝", "neural", "transformer", "llm"]
        if any(kw in task.lower() for kw in ai_keywords):
            arxiv_results = self.web.search_arxiv(task, max_results=3)

        # 결과 정리
        context_text = f"검색 결과:\n"
        for r in search_results:
            if "error" not in r:
                context_text += f"- {r.get('title', '')}: {r.get('snippet', '')[:200]}\n"

        if arxiv_results:
            context_text += "\n관련 논문:\n"
            for p in arxiv_results:
                if "error" not in p:
                    context_text += f"- {p.get('title', '')} ({p.get('published', '')})\n"
                    context_text += f"  {p.get('abstract', '')[:200]}\n"

        # LLM으로 분석
        messages = [
            Message(role="user", content=f"다음 정보를 분석하고 '{task}'에 대해 종합적으로 보고해주세요:\n\n{context_text}")
        ]
        response = self.llm.chat(messages, system=self.SYSTEM_PROMPT)
        return response.content


class CodeAgent(BaseAgent):
    """코드 작성 및 분석 전문 에이전트"""

    SYSTEM_PROMPT = """당신은 JARVIS의 Code Agent입니다.
임무: 코드 작성, 리뷰, 디버깅, 최적화를 수행합니다.
- 깔끔하고 효율적인 코드를 작성합니다
- 보안 취약점을 발견하고 수정합니다
- 다양한 프로그래밍 언어를 지원합니다
- 코드 설명을 한국어로 제공합니다"""

    def __init__(self, llm_manager, executor):
        super().__init__("CodeAgent", self.SYSTEM_PROMPT, llm_manager)
        self.executor = executor

    def run(self, task: str, context: Dict = None) -> str:
        """코드 태스크 실행"""
        from jarvis.llm.manager import Message

        messages = [
            Message(role="user", content=task)
        ]
        response = self.llm.chat(messages, system=self.SYSTEM_PROMPT, max_tokens=4096)

        # 코드 블록 추출 및 실행 여부 결정
        code = self._extract_code(response.content)
        if code and "execute:true" in task.lower():
            exec_result = self.executor.execute_python(code)
            return response.content + f"\n\n실행 결과:\n```\n{exec_result.get('output', exec_result.get('error', ''))}\n```"

        return response.content

    def _extract_code(self, text: str) -> Optional[str]:
        """텍스트에서 Python 코드 블록 추출"""
        import re
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else None


class SystemAgent(BaseAgent):
    """시스템 모니터링 전문 에이전트"""

    SYSTEM_PROMPT = """당신은 JARVIS의 System Agent입니다.
임무: 시스템 상태를 모니터링하고 이상을 감지합니다.
- CPU, 메모리, 디스크 사용량을 모니터링합니다
- 이상 프로세스를 감지합니다
- 성능 최적화 방안을 제안합니다
- 한국어로 보고합니다"""

    def __init__(self, llm_manager, computer):
        super().__init__("SystemAgent", self.SYSTEM_PROMPT, llm_manager)
        self.computer = computer

    def run(self, task: str, context: Dict = None) -> str:
        """시스템 태스크 실행"""
        from jarvis.llm.manager import Message

        sys_info = self.computer.get_system_info()
        processes = self.computer.get_running_processes(top_n=10)

        sys_text = json.dumps(sys_info, ensure_ascii=False, indent=2)
        proc_text = json.dumps(processes[:5], ensure_ascii=False)

        messages = [
            Message(
                role="user",
                content=f"태스크: {task}\n\n시스템 정보:\n{sys_text}\n\n상위 프로세스:\n{proc_text}"
            )
        ]
        response = self.llm.chat(messages, system=self.SYSTEM_PROMPT)
        return response.content


class PlannerAgent(BaseAgent):
    """태스크 계획 전문 에이전트"""

    SYSTEM_PROMPT = """당신은 JARVIS의 Planner Agent입니다.
임무: 복잡한 목표를 달성 가능한 단계로 분해합니다.
- 목표를 명확히 분석합니다
- 의존성을 고려한 실행 계획을 세웁니다
- 각 단계의 성공 기준을 정의합니다
- JSON 형식으로 계획을 제공합니다"""

    def __init__(self, llm_manager):
        super().__init__("PlannerAgent", self.SYSTEM_PROMPT, llm_manager)

    def run(self, task: str, context: Dict = None) -> str:
        """계획 수립"""
        from jarvis.llm.manager import Message

        messages = [
            Message(
                role="user",
                content=f"""다음 목표에 대한 실행 계획을 수립해주세요:
목표: {task}

다음 형식으로 응답해주세요:
1. 목표 분석
2. 단계별 계획 (각 단계: 설명, 도구, 예상 결과)
3. 위험 요소 및 대안
4. 성공 기준"""
            )
        ]
        response = self.llm.chat(messages, system=self.SYSTEM_PROMPT, max_tokens=4096)
        return response.content


class AgentManager:
    """
    JARVIS 에이전트 오케스트레이터
    태스크 라우팅, 병렬 실행, 결과 통합
    """

    def __init__(self, llm_manager, web_intel=None, computer=None, executor=None):
        self.llm = llm_manager
        self.tasks: Dict[str, AgentTask] = {}

        # 에이전트 초기화
        self.agents: Dict[AgentType, BaseAgent] = {}

        if web_intel:
            self.agents[AgentType.RESEARCH] = ResearchAgent(llm_manager, web_intel)
        if executor:
            self.agents[AgentType.CODE] = CodeAgent(llm_manager, executor)
        if computer:
            self.agents[AgentType.SYSTEM] = SystemAgent(llm_manager, computer)
        self.agents[AgentType.PLANNER] = PlannerAgent(llm_manager)

        logger.info(f"AgentManager initialized with {len(self.agents)} agents")

    def route_task(self, task: str) -> AgentType:
        """태스크 유형 자동 분류"""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["코드", "code", "python", "프로그래밍", "함수", "클래스", "debug"]):
            return AgentType.CODE
        elif any(kw in task_lower for kw in ["검색", "search", "찾아", "논문", "github", "트렌드", "최신"]):
            return AgentType.RESEARCH
        elif any(kw in task_lower for kw in ["시스템", "cpu", "메모리", "프로세스", "성능", "모니터"]):
            return AgentType.SYSTEM
        elif any(kw in task_lower for kw in ["계획", "plan", "단계", "전략", "어떻게 할"]):
            return AgentType.PLANNER
        else:
            return AgentType.GENERAL

    def run_agent(self, task: str, agent_type: AgentType = None, context: Dict = None) -> str:
        """에이전트 실행"""
        if agent_type is None:
            agent_type = self.route_task(task)

        task_id = f"task_{int(time.time())}_{agent_type.value}"
        agent_task = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            description=task,
            context=context or {},
            status="running",
        )
        self.tasks[task_id] = agent_task

        try:
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                result = agent.run(task, context)
            else:
                # GENERAL: 직접 LLM 사용
                from jarvis.llm.manager import Message
                from jarvis.config import JARVIS_SYSTEM_PROMPT
                messages = [Message(role="user", content=task)]
                response = self.llm.chat(messages, system=JARVIS_SYSTEM_PROMPT)
                result = response.content

            agent_task.status = "done"
            agent_task.result = result
            agent_task.completed_at = time.time()
            return result

        except Exception as e:
            agent_task.status = "error"
            agent_task.error = str(e)
            agent_task.completed_at = time.time()
            logger.error(f"Agent error [{agent_type}]: {e}")
            return f"에이전트 실행 오류: {e}"

    def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """태스크 상태 조회"""
        return self.tasks.get(task_id)

    def get_recent_tasks(self, n: int = 10) -> List[Dict]:
        """최근 태스크 목록"""
        tasks = sorted(self.tasks.values(), key=lambda t: t.created_at, reverse=True)
        return [
            {
                "task_id": t.task_id,
                "type": t.agent_type.value,
                "description": t.description[:100],
                "status": t.status,
                "duration": round(t.completed_at - t.created_at, 2) if t.completed_at else 0,
            }
            for t in tasks[:n]
        ]
