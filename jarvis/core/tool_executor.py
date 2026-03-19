"""
JARVIS 진짜 Tool Use 엔진 — Iteration 2
Anthropic의 실제 Tool Use API를 사용하는 구조화된 도구 호출 시스템
- 20+ 도구 정의
- 다중 도구 병렬 호출
- 도구 결과 피드백 루프
- 오류 복구 및 재시도
"""

import json
import time
import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── 전체 도구 정의 (Anthropic 형식) ────────────────────────────────────────
JARVIS_TOOLS = [
    {
        "name": "web_search",
        "description": "DuckDuckGo로 인터넷에서 실시간 정보를 검색합니다. 최신 뉴스, 기술 정보, 일반 지식 검색에 사용하세요.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 쿼리"},
                "max_results": {"type": "integer", "description": "최대 결과 수 (기본: 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "news_search",
        "description": "최신 뉴스 기사를 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "뉴스 검색 쿼리"},
                "max_results": {"type": "integer", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "github_search",
        "description": "GitHub에서 오픈소스 레포지토리, 코드, 프로젝트를 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "GitHub 검색 쿼리"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "arxiv_search",
        "description": "ArXiv에서 최신 AI/ML/CS 연구 논문을 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "논문 검색 쿼리"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "wikipedia_search",
        "description": "Wikipedia에서 개념, 역사, 인물에 대한 정보를 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "language": {"type": "string", "description": "ko 또는 en", "default": "ko"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_python",
        "description": "Python 코드를 안전하게 실행하고 결과를 반환합니다. 계산, 데이터 처리, 테스트에 사용하세요.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "실행할 Python 코드"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "execute_shell",
        "description": "안전한 Shell 명령을 실행합니다. 위험한 명령(rm -rf, format 등)은 자동 차단됩니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "실행할 Shell 명령"},
                "timeout": {"type": "integer", "description": "타임아웃 (초)", "default": 30},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "파일 내용을 읽습니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "읽을 파일 경로"},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_lines": {"type": "integer", "description": "최대 줄 수 (기본: 전체)", "default": 500},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "파일에 내용을 씁니다. 새 파일 생성 또는 기존 파일 덮어쓰기.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "append": {"type": "boolean", "description": "추가 모드", "default": False},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "디렉토리의 파일과 폴더 목록을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "탐색할 경로", "default": "."},
                "show_hidden": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "get_system_info",
        "description": "CPU, 메모리, 디스크, 네트워크 등 시스템 상태 정보를 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_processes",
        "description": "현재 실행 중인 프로세스 목록과 리소스 사용량을 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "상위 N개 프로세스", "default": 20},
                "sort_by": {"type": "string", "description": "cpu 또는 memory", "default": "cpu"},
            },
        },
    },
    {
        "name": "memory_save",
        "description": "중요한 정보를 JARVIS 장기 기억에 저장합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "기억 카테고리"},
                "content": {"type": "string", "description": "저장할 내용"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "태그 목록"},
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "memory_search",
        "description": "JARVIS 장기 기억에서 관련 정보를 검색합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_skill",
        "description": "스킬 라이브러리의 사전 정의된 스킬을 실행합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "스킬 이름"},
                "params": {"type": "object", "description": "스킬 파라미터", "default": {}},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "create_skill",
        "description": "새로운 스킬을 자동으로 작성하고 라이브러리에 저장합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "스킬의 기능 설명"},
                "category": {"type": "string", "default": "custom"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "analyze_screen",
        "description": "현재 화면을 캡처하고 Claude Vision으로 분석합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "화면에 대해 물어볼 질문", "default": "화면에 무엇이 있나요?"},
            },
        },
    },
    {
        "name": "reason_deeply",
        "description": "복잡한 문제를 고급 추론 엔진(CoT/ToT/ReAct)으로 심층 분석합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "strategy": {
                    "type": "string",
                    "enum": ["chain_of_thought", "tree_of_thought", "react", "self_reflection", "socratic"],
                    "description": "추론 전략",
                    "default": "chain_of_thought",
                },
            },
            "required": ["problem"],
        },
    },
    {
        "name": "run_agent",
        "description": "전문 에이전트(research/code/system/planner)를 실행하여 복잡한 작업을 수행합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": ["research", "code", "system", "planner", "summary"],
                },
                "task": {"type": "string", "description": "수행할 작업 설명"},
            },
            "required": ["agent_type", "task"],
        },
    },
    {
        "name": "get_ai_trends",
        "description": "최신 AI/ML 트렌드, 새로운 모델, 연구 동향을 가져옵니다.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


class ToolExecutor:
    """
    JARVIS 도구 실행 엔진
    Claude Tool Use API와 실제 도구들을 연결
    """

    def __init__(self, web, computer, executor, memory, agents, skills=None, vision=None, reasoning=None):
        self.web = web
        self.computer = computer
        self.executor = executor
        self.memory = memory
        self.agents = agents
        self.skills = skills
        self.vision = vision
        self.reasoning = reasoning

    def get_tools(self) -> List[Dict]:
        """Claude API에 전달할 도구 목록"""
        available = list(JARVIS_TOOLS)
        # Vision 도구: 사용 불가능하면 제거
        if not self.vision or not self.vision.get_status().get("available"):
            available = [t for t in available if t["name"] != "analyze_screen"]
        # Skills 없으면 제거
        if not self.skills:
            available = [t for t in available if t["name"] not in ("run_skill", "create_skill")]
        # Reasoning 없으면 제거
        if not self.reasoning:
            available = [t for t in available if t["name"] != "reason_deeply"]
        return available

    def execute(self, tool_name: str, tool_input: Dict) -> Any:
        """도구 이름과 입력으로 실행"""
        handlers = {
            "web_search": self._web_search,
            "news_search": self._news_search,
            "github_search": self._github_search,
            "arxiv_search": self._arxiv_search,
            "wikipedia_search": self._wikipedia_search,
            "execute_python": self._execute_python,
            "execute_shell": self._execute_shell,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "get_system_info": self._get_system_info,
            "get_processes": self._get_processes,
            "memory_save": self._memory_save,
            "memory_search": self._memory_search,
            "run_skill": self._run_skill,
            "create_skill": self._create_skill,
            "analyze_screen": self._analyze_screen,
            "reason_deeply": self._reason_deeply,
            "run_agent": self._run_agent,
            "get_ai_trends": self._get_ai_trends,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            start = time.perf_counter()
            result = handler(**tool_input)
            duration = round((time.perf_counter() - start) * 1000, 1)
            logger.info(f"Tool '{tool_name}' completed in {duration}ms")
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return {"error": str(e), "tool": tool_name}

    # ── Tool Use 루프 (Anthropic API) ──────────────────────────────────────────
    def run_with_tool_use(
        self,
        client,  # anthropic.Anthropic 클라이언트
        messages: List[Dict],
        system: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        max_iterations: int = 15,
    ) -> Tuple[str, List[Dict]]:
        """
        Claude Tool Use API 루프 실행
        - Claude가 도구를 선택하고 호출
        - 결과를 받아 다시 추론
        - 최종 텍스트 응답 반환
        """
        tools_used = []
        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=self.get_tools(),
                messages=current_messages,
            )

            # 텍스트 응답만 있으면 종료
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text
                return final_text, tools_used

            # 도구 호출 처리
            if response.stop_reason == "tool_use":
                # assistant 메시지 추가
                current_messages.append({
                    "role": "assistant",
                    "content": [
                        block.model_dump() if hasattr(block, "model_dump") else dict(block)
                        for block in response.content
                    ],
                })

                # 각 도구 실행
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info(f"[Tool Use] {block.name}: {json.dumps(block.input, ensure_ascii=False)[:200]}")
                        result = self.execute(block.name, block.input)
                        result_str = json.dumps(result, ensure_ascii=False, default=str)
                        if len(result_str) > 8000:
                            result_str = result_str[:8000] + "\n...[결과 truncated]"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                        tools_used.append({
                            "tool": block.name,
                            "input": block.input,
                            "result_preview": result_str[:200],
                        })

                # 도구 결과 추가
                current_messages.append({"role": "user", "content": tool_results})
                continue

            # 예상치 못한 stop_reason
            break

        # 최대 반복 초과 — 마지막 응답에서 텍스트 추출
        final = ""
        for block in response.content:
            if hasattr(block, "text"):
                final += block.text
        return final or "최대 반복 횟수에 도달했습니다.", tools_used

    def run_streaming_with_tool_use(
        self,
        client,
        messages: List[Dict],
        system: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        max_iterations: int = 15,
    ) -> Generator[Dict, None, None]:
        """
        스트리밍 Tool Use 루프
        yield {"type": "text"|"tool_start"|"tool_result"|"done", ...}
        """
        tools_used = []
        current_messages = list(messages)

        for iteration in range(max_iterations):
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=self.get_tools(),
                messages=current_messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        yield {"type": "text", "content": block.text}
                yield {"type": "done", "tools_used": tools_used}
                return

            if response.stop_reason == "tool_use":
                current_messages.append({
                    "role": "assistant",
                    "content": [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in response.content],
                })

                tool_results = []
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        yield {"type": "text", "content": block.text}
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "tool": block.name, "input": block.input}
                        result = self.execute(block.name, block.input)
                        result_str = json.dumps(result, ensure_ascii=False, default=str)
                        if len(result_str) > 8000:
                            result_str = result_str[:8000] + "\n...[truncated]"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                        tools_used.append({"tool": block.name})
                        yield {"type": "tool_result", "tool": block.name, "result": result_str[:300]}

                current_messages.append({"role": "user", "content": tool_results})
                continue

            break

        yield {"type": "done", "tools_used": tools_used}

    # ── 도구 핸들러 ─────────────────────────────────────────────────────────
    def _web_search(self, query: str, max_results: int = 5, **kw) -> Any:
        results = self.web.search_web(query, max_results=max_results)
        return {"results": results, "count": len(results), "query": query}

    def _news_search(self, query: str, max_results: int = 8, **kw) -> Any:
        results = self.web.search_news(query, max_results=max_results)
        return {"results": results, "count": len(results)}

    def _github_search(self, query: str, max_results: int = 5, **kw) -> Any:
        results = self.web.search_github(query, max_results=max_results)
        return {"results": results, "count": len(results)}

    def _arxiv_search(self, query: str, max_results: int = 5, **kw) -> Any:
        results = self.web.search_arxiv(query, max_results=max_results)
        return {"results": results, "count": len(results)}

    def _wikipedia_search(self, query: str, language: str = "ko", **kw) -> Any:
        result = self.web.search_wikipedia(query)
        return result

    def _execute_python(self, code: str, **kw) -> Any:
        return self.executor.execute_python(code)

    def _execute_shell(self, command: str, timeout: int = 30, **kw) -> Any:
        return self.computer.execute_command(command, timeout=timeout)

    def _read_file(self, path: str, encoding: str = "utf-8", max_lines: int = 500, **kw) -> Any:
        result = self.computer.read_file(path, encoding=encoding)
        if result.get("success") and result.get("content"):
            lines = result["content"].splitlines()
            if len(lines) > max_lines:
                result["content"] = "\n".join(lines[:max_lines]) + f"\n...[총 {len(lines)}줄 중 {max_lines}줄 표시]"
                result["truncated"] = True
        return result

    def _write_file(self, path: str, content: str, encoding: str = "utf-8", append: bool = False, **kw) -> Any:
        return self.computer.write_file(path, content, encoding=encoding, append=append)

    def _list_directory(self, path: str = ".", show_hidden: bool = False, **kw) -> Any:
        return self.computer.list_directory(path)

    def _get_system_info(self, **kw) -> Any:
        return self.computer.get_system_info()

    def _get_processes(self, top_n: int = 20, sort_by: str = "cpu", **kw) -> Any:
        return self.computer.get_processes(top_n=top_n, sort_by=sort_by)

    def _memory_save(self, key: str, content: str, tags: List[str] = None, **kw) -> Any:
        self.memory.add_knowledge(key, content, " ".join(tags or []))
        return {"saved": True, "key": key}

    def _memory_search(self, query: str, n_results: int = 5, **kw) -> Any:
        results = self.memory.search_similar(query, n_results=n_results)
        return {"results": results, "count": len(results)}

    def _run_skill(self, skill_name: str, params: Dict = None, **kw) -> Any:
        if not self.skills:
            return {"error": "SkillLibrary 없음"}
        return self.skills.run(skill_name, **(params or {}))

    def _create_skill(self, description: str, category: str = "custom", **kw) -> Any:
        if not self.skills:
            return {"error": "SkillLibrary 없음"}
        return self.skills.create_skill(description, category=category)

    def _analyze_screen(self, question: str = "화면에 무엇이 있나요?", **kw) -> Any:
        if not self.vision:
            return {"error": "VisionSystem 없음"}
        return self.vision.analyze_screen(question=question)

    def _reason_deeply(self, problem: str, strategy: str = "chain_of_thought", **kw) -> Any:
        if not self.reasoning:
            return {"error": "ReasoningEngine 없음"}
        from jarvis.core.reasoning import ReasoningStrategy
        strat = ReasoningStrategy(strategy)
        result = self.reasoning.reason(problem, strategy=strat)
        return {
            "strategy": result.strategy,
            "answer": result.final_answer,
            "confidence": result.confidence,
            "duration": result.duration,
            "steps": len(result.steps),
        }

    def _run_agent(self, agent_type: str, task: str, **kw) -> Any:
        from jarvis.agents.agent_manager import AgentType
        try:
            at = AgentType(agent_type)
            result = self.agents.run_agent(task, agent_type=at)
            return {"agent": agent_type, "result": result}
        except Exception as e:
            return {"error": str(e)}

    def _get_ai_trends(self, **kw) -> Any:
        try:
            return self.web.get_ai_trends()
        except Exception as e:
            return {"error": str(e)}
