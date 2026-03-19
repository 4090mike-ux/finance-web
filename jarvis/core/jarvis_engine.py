"""
JARVIS 핵심 엔진
모든 모듈을 통합하는 중앙 오케스트레이터
Chain-of-Thought 추론, Tool Use, 멀티에이전트 조율
"""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Generator, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class JarvisEngine:
    """
    JARVIS (Just A Rather Very Intelligent System)
    핵심 엔진 - 모든 모듈 통합 및 지능형 추론
    """

    # 도구 정의 (Claude Tool Use 스타일)
    TOOLS = {
        "web_search": "웹 검색",
        "github_search": "GitHub 검색",
        "arxiv_search": "ArXiv 논문 검색",
        "wikipedia": "Wikipedia 검색",
        "execute_python": "Python 코드 실행",
        "execute_shell": "Shell 명령 실행",
        "read_file": "파일 읽기",
        "write_file": "파일 쓰기",
        "list_directory": "디렉토리 목록",
        "system_info": "시스템 정보",
        "get_processes": "프로세스 목록",
        "run_agent": "에이전트 실행",
        "remember": "기억 저장",
        "recall": "기억 검색",
    }

    def __init__(
        self,
        llm_manager,
        memory_manager,
        computer_controller,
        web_intelligence,
        code_executor,
        agent_manager,
        voice_interface=None,
    ):
        self.llm = llm_manager
        self.memory = memory_manager
        self.computer = computer_controller
        self.web = web_intelligence
        self.executor = code_executor
        self.agents = agent_manager
        self.voice = voice_interface

        self.conversation_history = []
        self.is_thinking = False
        self.startup_time = time.time()

        logger.info("JARVIS Engine fully initialized - All systems operational")

    # ==================== 메인 채팅 ====================

    def chat(self, user_input: str, use_tools: bool = True) -> Dict:
        """
        메인 대화 함수
        1. 사용자 입력 분석
        2. 필요시 도구 사용
        3. LLM으로 최종 응답 생성
        """
        start_time = time.time()
        self.is_thinking = True

        # 메모리에 사용자 메시지 저장
        self.memory.add_message("user", user_input)
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # 도구 사용이 필요한지 판단
            tool_results = []
            if use_tools:
                tool_results = self._determine_and_use_tools(user_input)

            # 컨텍스트 구성
            context = self._build_context(user_input, tool_results)

            # LLM 응답 생성
            response = self._generate_response(context, tool_results)

            # 응답 저장
            self.memory.add_message("assistant", response)
            self.conversation_history.append({"role": "assistant", "content": response})

            # 태스크 이력 기록
            duration = time.time() - start_time
            self.memory.log_task(
                task_type="chat",
                description=user_input[:200],
                result=response[:500],
                status="success",
                duration=duration,
            )

            self.is_thinking = False
            return {
                "response": response,
                "tools_used": [t["tool"] for t in tool_results],
                "duration": round(duration, 2),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.is_thinking = False
            logger.error(f"Chat error: {e}")
            error_msg = f"죄송합니다, 처리 중 오류가 발생했습니다: {str(e)}"
            return {
                "response": error_msg,
                "tools_used": [],
                "error": str(e),
                "duration": round(time.time() - start_time, 2),
            }

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """스트리밍 채팅 (WebSocket용)"""
        # 도구 결과 먼저 처리
        tool_results = self._determine_and_use_tools(user_input)

        if tool_results:
            yield f"[도구 사용 완료: {', '.join(t['tool'] for t in tool_results)}]\n\n"

        context = self._build_context(user_input, tool_results)

        from jarvis.llm.manager import Message
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        messages = self._build_messages(context)
        full_response = ""

        for chunk in self.llm.stream_chat(messages, system=JARVIS_SYSTEM_PROMPT + self._get_context_suffix(tool_results)):
            full_response += chunk
            yield chunk

        # 저장
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", full_response)
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": full_response})

    # ==================== 도구 사용 ====================

    def _determine_and_use_tools(self, user_input: str) -> List[Dict]:
        """어떤 도구를 사용할지 결정하고 실행"""
        tools_to_use = self._classify_tools_needed(user_input)
        results = []

        for tool_name, tool_query in tools_to_use:
            try:
                result = self._execute_tool(tool_name, tool_query, user_input)
                if result:
                    results.append({
                        "tool": tool_name,
                        "query": tool_query,
                        "result": result,
                    })
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")

        return results

    def _classify_tools_needed(self, user_input: str) -> List[Tuple[str, str]]:
        """사용할 도구 목록 결정"""
        inp = user_input.lower()
        tools = []

        # 최신 정보 필요
        if any(kw in inp for kw in ["최신", "현재", "지금", "오늘", "뉴스", "트렌드", "최근"]):
            tools.append(("web_search", user_input))

        # 검색 요청
        if any(kw in inp for kw in ["검색", "찾아", "알려줘", "뭐야", "what is", "search"]):
            tools.append(("web_search", user_input))

        # GitHub 관련
        if any(kw in inp for kw in ["github", "깃허브", "레포", "오픈소스", "코드 찾아"]):
            tools.append(("github_search", user_input))

        # 논문 관련
        if any(kw in inp for kw in ["논문", "paper", "arxiv", "연구", "리서치"]):
            tools.append(("arxiv_search", user_input))

        # 코드 실행 요청
        if any(kw in inp for kw in ["실행", "run", "execute", "계산", "결과", "코드 돌려"]):
            # 코드 블록이 있는지 확인
            if "```" in user_input or "python" in inp:
                tools.append(("execute_python", user_input))

        # 시스템 정보
        if any(kw in inp for kw in ["시스템", "cpu", "메모리", "디스크", "컴퓨터 상태", "프로세스"]):
            tools.append(("system_info", ""))

        # 파일 작업
        if any(kw in inp for kw in ["파일", "폴더", "디렉토리", "읽어", "열어"]):
            tools.append(("list_directory", "."))

        # 기억 검색
        if any(kw in inp for kw in ["기억", "remember", "이전에", "지난번", "recall"]):
            tools.append(("recall", user_input))

        # 중복 제거
        seen = set()
        unique = []
        for t in tools:
            if t[0] not in seen:
                seen.add(t[0])
                unique.append(t)

        return unique[:3]  # 최대 3개 도구

    def _execute_tool(self, tool_name: str, query: str, original_input: str) -> Any:
        """개별 도구 실행"""
        if tool_name == "web_search":
            results = self.web.search_web(query, max_results=5)
            return self._format_search_results(results)

        elif tool_name == "github_search":
            results = self.web.search_github(query, max_results=5)
            return self._format_github_results(results)

        elif tool_name == "arxiv_search":
            results = self.web.search_arxiv(query, max_results=5)
            return self._format_arxiv_results(results)

        elif tool_name == "execute_python":
            # 입력에서 코드 추출
            code = self._extract_code_from_input(original_input)
            if code:
                return self.executor.execute_python(code)
            return None

        elif tool_name == "system_info":
            return self.computer.get_system_info()

        elif tool_name == "list_directory":
            return self.computer.list_directory(query or ".")

        elif tool_name == "recall":
            results = self.memory.search_similar(query, n_results=5)
            return {"memories": results}

        elif tool_name == "remember":
            self.memory.add_knowledge("user_info", query, original_input)
            return {"saved": True}

        return None

    def _extract_code_from_input(self, text: str) -> Optional[str]:
        """입력에서 코드 블록 추출"""
        pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0] if matches else None

    # ==================== 컨텍스트 구성 ====================

    def _build_context(self, user_input: str, tool_results: List[Dict]) -> str:
        """LLM에 전달할 컨텍스트 구성"""
        context_parts = [user_input]

        if tool_results:
            context_parts.append("\n\n[도구 실행 결과]")
            for tr in tool_results:
                context_parts.append(f"\n{tr['tool']} 결과:")
                result_str = json.dumps(tr["result"], ensure_ascii=False, indent=2)
                context_parts.append(result_str[:3000])

        return "\n".join(context_parts)

    def _build_messages(self, context: str) -> List:
        """LLM 메시지 리스트 구성"""
        from jarvis.llm.manager import Message

        messages = []

        # 최근 대화 히스토리 포함 (최대 10개)
        history = self.conversation_history[-10:]
        for h in history[:-1]:  # 마지막 user 메시지 제외 (context에 포함)
            messages.append(Message(role=h["role"], content=h["content"]))

        # 현재 사용자 메시지 (컨텍스트 포함)
        messages.append(Message(role="user", content=context))

        return messages

    def _generate_response(self, context: str, tool_results: List[Dict]) -> str:
        """LLM 응답 생성"""
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        system = JARVIS_SYSTEM_PROMPT + self._get_context_suffix(tool_results)
        messages = self._build_messages(context)

        response = self.llm.chat(messages, system=system, max_tokens=4096)
        return response.content

    def _get_context_suffix(self, tool_results: List[Dict]) -> str:
        """시스템 프롬프트에 추가할 컨텍스트"""
        if not tool_results:
            return ""

        suffix = "\n\n## 현재 실행된 도구 결과가 있습니다. 이 결과를 바탕으로 정확하고 상세한 답변을 제공하세요."
        return suffix

    # ==================== 포맷팅 함수 ====================

    def _format_search_results(self, results: List[Dict]) -> str:
        """검색 결과 포맷"""
        if not results or "error" in results[0]:
            return "검색 결과 없음"

        lines = []
        for i, r in enumerate(results[:5]):
            lines.append(f"{i+1}. {r.get('title', '')}")
            lines.append(f"   {r.get('snippet', '')[:200]}")
            lines.append(f"   URL: {r.get('url', '')}")
        return "\n".join(lines)

    def _format_github_results(self, results: List[Dict]) -> str:
        """GitHub 결과 포맷"""
        if not results or "error" in results[0]:
            return "GitHub 결과 없음"

        lines = []
        for r in results[:5]:
            lines.append(f"⭐ {r.get('stars', 0)} | {r.get('name', '')} ({r.get('language', '')})")
            lines.append(f"   {r.get('description', '')[:150]}")
            lines.append(f"   URL: {r.get('url', '')}")
        return "\n".join(lines)

    def _format_arxiv_results(self, results: List[Dict]) -> str:
        """ArXiv 결과 포맷"""
        if not results or "error" in results[0]:
            return "논문 검색 결과 없음"

        lines = []
        for r in results[:5]:
            lines.append(f"📄 {r.get('title', '')} ({r.get('published', '')})")
            lines.append(f"   저자: {', '.join(r.get('authors', [])[:3])}")
            lines.append(f"   {r.get('abstract', '')[:200]}")
        return "\n".join(lines)

    # ==================== 명령 파싱 ====================

    def execute_command(self, command: str) -> Dict:
        """
        직접 명령 실행 (채팅 없이)
        /search, /code, /system, /file 등
        """
        cmd = command.strip().lower()

        if cmd.startswith("/search "):
            query = command[8:]
            results = self.web.search_web(query)
            return {"type": "search", "results": results}

        elif cmd.startswith("/github "):
            query = command[8:]
            results = self.web.search_github(query)
            return {"type": "github", "results": results}

        elif cmd.startswith("/arxiv "):
            query = command[7:]
            results = self.web.search_arxiv(query)
            return {"type": "arxiv", "results": results}

        elif cmd.startswith("/code "):
            code = command[6:]
            result = self.executor.execute_python(code)
            return {"type": "code_execution", "result": result}

        elif cmd == "/system":
            info = self.computer.get_system_info()
            return {"type": "system_info", "info": info}

        elif cmd.startswith("/file "):
            path = command[6:]
            result = self.computer.read_file(path)
            return {"type": "file", "result": result}

        elif cmd == "/status":
            return self.get_status()

        elif cmd == "/memory":
            return {"type": "memory", "stats": self.memory.get_stats()}

        elif cmd == "/trends":
            trends = self.web.get_ai_trends()
            return {"type": "trends", "data": trends}

        else:
            return {"error": f"Unknown command: {command}"}

    # ==================== 상태 정보 ====================

    def get_status(self) -> Dict:
        """JARVIS 전체 상태"""
        uptime = time.time() - self.startup_time
        mem_stats = self.memory.get_stats()
        sys_info = self.computer.get_system_info()

        return {
            "status": "operational",
            "uptime_seconds": round(uptime, 0),
            "uptime_human": self._format_uptime(uptime),
            "providers": list(self.llm.providers.keys()) if hasattr(self.llm, 'providers') else [],
            "agents": list(self.agents.agents.keys()) if hasattr(self.agents, 'agents') else [],
            "memory": mem_stats,
            "system": {
                "cpu": sys_info.get("cpu", {}).get("usage_percent", 0),
                "memory_percent": sys_info.get("memory", {}).get("used_percent", 0),
                "disk_percent": sys_info.get("disk", {}).get("used_percent", 0),
            },
            "voice": self.voice.get_status() if self.voice else {"available": False},
            "conversation_turns": len(self.conversation_history) // 2,
            "timestamp": datetime.now().isoformat(),
        }

    def _format_uptime(self, seconds: float) -> str:
        """업타임 포맷"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}시간 {m}분 {s}초"

    def greet(self) -> str:
        """시작 인사"""
        sys_info = self.computer.get_system_info()
        cpu = sys_info.get("cpu", {}).get("usage_percent", 0)
        mem = sys_info.get("memory", {}).get("used_percent", 0)

        return f"""안녕하세요, 사용자님. JARVIS가 온라인 상태입니다.

🟢 모든 시스템 정상 작동 중
📊 CPU: {cpu:.1f}% | 메모리: {mem:.1f}%
🧠 LLM: {', '.join(self.llm.providers.keys()) if hasattr(self.llm, 'providers') and self.llm.providers else 'Mock mode'}
🤖 에이전트: {len(self.agents.agents) if hasattr(self.agents, 'agents') else 0}개 대기 중

어떻게 도와드릴까요?"""
