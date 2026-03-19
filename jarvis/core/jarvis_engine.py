"""
JARVIS 핵심 엔진 — Iteration 2
모든 모듈을 통합하는 중앙 오케스트레이터
실제 Claude Tool Use API + 고급 추론 + 오케스트레이터 + 스킬 라이브러리
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
    핵심 엔진 Iteration 2 — 진정한 도구 사용 AI
    """

    def __init__(
        self,
        llm_manager,
        memory_manager,
        computer_controller,
        web_intelligence,
        code_executor,
        agent_manager,
        voice_interface=None,
        skill_library=None,
        vision_system=None,
        orchestrator=None,
    ):
        self.llm = llm_manager
        self.memory = memory_manager
        self.computer = computer_controller
        self.web = web_intelligence
        self.executor = code_executor
        self.agents = agent_manager
        self.voice = voice_interface
        self.skills = skill_library
        self.vision = vision_system
        self.orchestrator = orchestrator

        self.conversation_history = []
        self.is_thinking = False
        self.startup_time = time.time()
        self._anthropic_client = None  # Claude Tool Use API용

        # ── 고급 추론 및 자기 개선 시스템 ──
        try:
            from jarvis.core.reasoning import ReasoningEngine
            from jarvis.core.self_improvement import SelfImprovementSystem
            from jarvis.core.knowledge_updater import KnowledgeUpdater
            from jarvis.config import MEMORY_DB_PATH

            self.reasoning = ReasoningEngine(llm_manager)
            self.improvement = SelfImprovementSystem(MEMORY_DB_PATH, llm_manager)
            self.knowledge_updater = KnowledgeUpdater(web_intelligence, memory_manager, llm_manager)
            self.knowledge_updater.start_auto_update()
            logger.info("Advanced reasoning and self-improvement systems active")
        except Exception as e:
            logger.warning(f"Advanced systems init failed: {e}")
            self.reasoning = None
            self.improvement = None
            self.knowledge_updater = None

        # ── 실제 Tool Use 엔진 초기화 ──
        try:
            from jarvis.core.tool_executor import ToolExecutor
            self.tool_executor = ToolExecutor(
                web=self.web,
                computer=self.computer,
                executor=self.executor,
                memory=self.memory,
                agents=self.agents,
                skills=self.skills,
                vision=self.vision,
                reasoning=self.reasoning,
            )
            self._init_anthropic_client()
            logger.info("Tool Use engine initialized — real API tool calling active")
        except Exception as e:
            logger.warning(f"ToolExecutor init failed: {e}")
            self.tool_executor = None

        logger.info("JARVIS Engine Iteration 2 — All systems operational")

    # ==================== Anthropic 클라이언트 초기화 ====================

    def _init_anthropic_client(self):
        """Claude Tool Use API용 클라이언트"""
        try:
            import anthropic
            from jarvis.config import ANTHROPIC_API_KEY
            if ANTHROPIC_API_KEY:
                self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("Anthropic client for Tool Use ready")
        except Exception as e:
            logger.warning(f"Anthropic client init failed: {e}")

    # ==================== 메인 채팅 ====================

    def chat(self, user_input: str, use_tools: bool = True) -> Dict:
        """
        메인 대화 함수 — Iteration 2
        실제 Claude Tool Use API 사용
        """
        start_time = time.time()
        self.is_thinking = True

        self.memory.add_message("user", user_input)
        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            # ── 실제 Tool Use API (Anthropic 클라이언트 있을 때) ──
            if use_tools and self.tool_executor and self._anthropic_client:
                response, tools_used = self._chat_with_tool_use(user_input)
                tools_list = [t["tool"] for t in tools_used]
            else:
                # 폴백: 기존 키워드 기반
                tool_results = self._determine_and_use_tools(user_input) if use_tools else []
                context = self._build_context(user_input, tool_results)
                response = self._generate_response(context, tool_results)
                tools_list = [t["tool"] for t in tool_results]

            self.memory.add_message("assistant", response)
            self.conversation_history.append({"role": "assistant", "content": response})

            duration = time.time() - start_time
            self.memory.log_task("chat", user_input[:200], response[:500], "success", duration)

            self.is_thinking = False
            return {
                "response": response,
                "tools_used": tools_list,
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

    def _chat_with_tool_use(self, user_input: str) -> Tuple[str, List[Dict]]:
        """실제 Claude Tool Use API로 채팅"""
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        # 대화 히스토리 포함
        messages = []
        for h in self.conversation_history[-10:]:
            messages.append({"role": h["role"], "content": h["content"]})

        # 현재 메시지 추가 (마지막 user는 이미 추가됨)
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": user_input})

        return self.tool_executor.run_with_tool_use(
            client=self._anthropic_client,
            messages=messages,
            system=JARVIS_SYSTEM_PROMPT,
            max_tokens=8192,
        )

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """스트리밍 채팅 (WebSocket용) — Tool Use 지원"""
        from jarvis.config import JARVIS_SYSTEM_PROMPT

        self.memory.add_message("user", user_input)
        self.conversation_history.append({"role": "user", "content": user_input})

        full_response = ""
        tools_used = []

        # ── 실제 Tool Use 스트리밍 ──
        if self.tool_executor and self._anthropic_client:
            messages = [{"role": h["role"], "content": h["content"]} for h in self.conversation_history[-10:]]

            for event in self.tool_executor.run_streaming_with_tool_use(
                client=self._anthropic_client,
                messages=messages,
                system=JARVIS_SYSTEM_PROMPT,
            ):
                if event["type"] == "text":
                    chunk = event["content"]
                    full_response += chunk
                    yield chunk
                elif event["type"] == "tool_start":
                    yield f"\n🔧 **{event['tool']}** 실행 중...\n"
                elif event["type"] == "tool_result":
                    yield f"✅ **{event['tool']}** 완료\n\n"
                elif event["type"] == "done":
                    tools_used = event.get("tools_used", [])
        else:
            # 폴백
            tool_results = self._determine_and_use_tools(user_input)
            if tool_results:
                yield f"[도구: {', '.join(t['tool'] for t in tool_results)}]\n\n"
            context = self._build_context(user_input, tool_results)
            from jarvis.llm.manager import Message
            messages = self._build_messages(context)
            for chunk in self.llm.stream_chat(messages, system=JARVIS_SYSTEM_PROMPT):
                full_response += chunk
                yield chunk

        self.memory.add_message("assistant", full_response)
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

        elif cmd.startswith("/reason "):
            problem = command[8:]
            if self.reasoning:
                strategy = self.reasoning.select_strategy(problem)
                result = self.reasoning.reason(problem, strategy=strategy)
                return {"type": "reasoning", "strategy": result.strategy,
                        "answer": result.final_answer, "confidence": result.confidence,
                        "duration": result.duration}
            return {"error": "Reasoning engine not available"}

        elif cmd == "/improve":
            if self.improvement:
                report = self.improvement.get_performance_report()
                suggestions = self.improvement.suggest_improvements()
                return {"type": "improvement", "report": report, "suggestions": suggestions}
            return {"error": "Self-improvement not available"}

        elif cmd == "/update":
            if self.knowledge_updater:
                result = self.knowledge_updater.run_update_cycle()
                return {"type": "knowledge_update", "result": result}
            return {"error": "Knowledge updater not available"}

        elif cmd == "/knowledge":
            if self.knowledge_updater:
                summary = self.knowledge_updater.get_knowledge_summary()
                kstatus = self.knowledge_updater.get_status()
                return {"type": "knowledge", "summary": summary, "status": kstatus}
            return {"type": "knowledge", "summary": "지식 갱신 시스템 없음"}

        elif cmd.startswith("/wikipedia "):
            query = command[11:]
            result = self.web.search_wikipedia(query)
            return {"type": "wikipedia", "result": result}

        elif cmd.startswith("/news "):
            query = command[6:]
            results = self.web.search_news(query, max_results=10)
            return {"type": "news", "results": results}

        elif cmd.startswith("/agent "):
            parts = command[7:].split(" ", 1)
            if len(parts) == 2:
                agent_type_str, task = parts
                from jarvis.agents.agent_manager import AgentType
                try:
                    at = AgentType(agent_type_str)
                    result = self.agents.run_agent(task, agent_type=at)
                    return {"type": "agent", "agent": agent_type_str, "result": result}
                except ValueError:
                    return {"error": f"Unknown agent type: {agent_type_str}"}
            return {"error": "Usage: /agent [type] [task]"}

        # ── Iteration 2 신규 명령 ──
        elif cmd.startswith("/goal "):
            goal = command[6:]
            if self.orchestrator:
                result = self.orchestrator.execute_goal(goal, background=False)
                return {"type": "orchestrator", "result": result}
            return self.chat(goal)

        elif cmd.startswith("/goal_bg "):
            goal = command[9:]
            if self.orchestrator:
                result = self.orchestrator.execute_goal(goal, background=True)
                return {"type": "orchestrator_bg", "task_id": result.get("task_id")}
            return {"error": "Orchestrator not available"}

        elif cmd.startswith("/task "):
            task_id = command[6:].strip()
            if self.orchestrator:
                return {"type": "task_status", "task": self.orchestrator.get_task(task_id)}
            return {"error": "Orchestrator not available"}

        elif cmd == "/tasks":
            if self.orchestrator:
                return {"type": "task_list", "tasks": self.orchestrator.list_tasks()}
            return {"error": "Orchestrator not available"}

        elif cmd == "/skills":
            if self.skills:
                return {"type": "skills", "skills": self.skills.list_skills(), "stats": self.skills.get_stats()}
            return {"error": "SkillLibrary not available"}

        elif cmd.startswith("/skill "):
            name = command[7:].strip()
            if self.skills:
                code = self.skills.get_skill_code(name)
                return {"type": "skill_code", "name": name, "code": code}
            return {"error": "SkillLibrary not available"}

        elif cmd.startswith("/create_skill "):
            desc = command[14:]
            if self.skills:
                result = self.skills.create_skill(desc)
                return {"type": "skill_created", "result": result}
            return {"error": "SkillLibrary not available"}

        elif cmd == "/screen":
            if self.vision:
                return self.vision.analyze_screen()
            return {"error": "VisionSystem not available"}

        elif cmd.startswith("/screen "):
            question = command[8:]
            if self.vision:
                return self.vision.analyze_screen(question=question)
            return {"error": "VisionSystem not available"}

        elif cmd == "/vision":
            if self.vision:
                return {"type": "vision_status", "status": self.vision.get_status()}
            return {"error": "VisionSystem not available"}

        elif cmd == "/tools":
            if self.tool_executor:
                return {"type": "tools", "tools": [t["name"] for t in self.tool_executor.get_tools()], "count": len(self.tool_executor.get_tools())}
            return {"error": "ToolExecutor not available"}

        else:
            return {"error": f"Unknown command: {command}. Try /status, /system, /search, /code, /goal, /skills, /screen, /tools, /tasks"}

    # ==================== 상태 정보 ====================

    def get_status(self) -> Dict:
        """JARVIS 전체 상태 — Iteration 2"""
        uptime = time.time() - self.startup_time
        mem_stats = self.memory.get_stats()
        sys_info = self.computer.get_system_info()

        status = {
            "status": "operational",
            "iteration": 2,
            "uptime_seconds": round(uptime, 0),
            "uptime_human": self._format_uptime(uptime),
            "providers": list(self.llm.providers.keys()) if hasattr(self.llm, 'providers') else [],
            "agents": [k.value for k in self.agents.agents.keys()] if hasattr(self.agents, 'agents') else [],
            "memory": mem_stats,
            "system": {
                "cpu": sys_info.get("cpu", {}).get("usage_percent", 0),
                "memory_percent": sys_info.get("memory", {}).get("used_percent", 0),
                "disk_percent": sys_info.get("disk", {}).get("used_percent", 0),
            },
            "voice": self.voice.get_status() if self.voice else {"available": False},
            "conversation_turns": len(self.conversation_history) // 2,
            "advanced_systems": {
                "reasoning": self.reasoning is not None,
                "self_improvement": self.improvement is not None,
                "knowledge_updater": self.knowledge_updater is not None,
                "auto_update_active": self.knowledge_updater.is_running if self.knowledge_updater else False,
                # Iteration 2
                "tool_use_api": self._anthropic_client is not None and self.tool_executor is not None,
                "skill_library": self.skills is not None,
                "vision_system": self.vision is not None,
                "orchestrator": self.orchestrator is not None,
            },
            "tools": {
                "count": len(self.tool_executor.get_tools()) if self.tool_executor else 0,
                "names": [t["name"] for t in self.tool_executor.get_tools()] if self.tool_executor else [],
            },
            "skills": self.skills.get_stats() if self.skills else {},
            "orchestrator": self.orchestrator.get_stats() if self.orchestrator else {},
            "vision": self.vision.get_status() if self.vision else {"available": False},
            "timestamp": datetime.now().isoformat(),
        }
        return status

    def _format_uptime(self, seconds: float) -> str:
        """업타임 포맷"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}시간 {m}분 {s}초"

    def greet(self) -> str:
        """시작 인사 — Iteration 2"""
        sys_info = self.computer.get_system_info()
        cpu = sys_info.get("cpu", {}).get("usage_percent", 0)
        mem = sys_info.get("memory", {}).get("used_percent", 0)
        providers = list(self.llm.providers.keys()) if hasattr(self.llm, 'providers') and self.llm.providers else ["Mock"]
        tool_count = len(self.tool_executor.get_tools()) if self.tool_executor else 0
        skill_count = len(self.skills._registry) if self.skills else 0

        return f"""안녕하세요, 사용자님. JARVIS Iteration 2가 온라인입니다.

🟢 모든 시스템 정상 작동 중
📊 CPU: {cpu:.1f}% | 메모리: {mem:.1f}%
🧠 LLM: {', '.join(providers)}
🔧 도구: {tool_count}개 | 스킬: {skill_count}개
🤖 에이전트: {len(self.agents.agents) if hasattr(self.agents, 'agents') else 0}개
{'🔍 Vision: 활성화' if self.vision and self.vision.get_status().get('available') else ''}
{'🎯 Orchestrator: 활성화' if self.orchestrator else ''}
{'⚡ 실제 Tool Use API: 활성화' if self._anthropic_client and self.tool_executor else '⚠ Tool Use: 폴백 모드'}

**사용 가능한 특수 명령:**
`/goal [복잡한목표]` — 자율 목표 달성
`/skills` — 스킬 목록
`/screen` — 화면 분석
`/tools` — 사용 가능 도구
`/reason [문제]` — 심층 추론

어떻게 도와드릴까요, 사용자님?"""
