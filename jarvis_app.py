"""
JARVIS 웹 애플리케이션
Flask + SocketIO 기반 실시간 채팅 인터페이스
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis")

# JARVIS 모듈 임포트
from jarvis.config import (
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GITHUB_TOKEN,
    MEMORY_DB_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL,
    JARVIS_SYSTEM_PROMPT, ALLOWED_DIRECTORIES,
)
from jarvis.llm import LLMManager
from jarvis.memory import MemoryManager
from jarvis.computer import ComputerController
from jarvis.web import WebIntelligence
from jarvis.executor import CodeExecutor
from jarvis.agents import AgentManager
from jarvis.voice import VoiceInterface
from jarvis.core import JarvisEngine
# Iteration 2
from jarvis.skills import SkillLibrary
from jarvis.computer.vision import VisionSystem
from jarvis.agents.orchestrator import Orchestrator
from jarvis.llm.ollama_provider import OllamaProvider
# Iteration 3
from jarvis.core.autonomous_loop import AutonomousLoop
from jarvis.core.self_modifier import SelfModifier
from jarvis.research.deep_researcher import DeepResearcher
from jarvis.agents.code_intelligence import CodeIntelligence

# ==================== 앱 초기화 ====================

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SECRET_KEY", "jarvis-secret-2026")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ==================== JARVIS 초기화 ====================

def create_jarvis() -> JarvisEngine:
    """JARVIS 엔진 생성 — Iteration 2"""
    logger.info("Initializing JARVIS Iteration 2...")

    # ── 핵심 모듈 ──
    llm = LLMManager(
        anthropic_key=ANTHROPIC_API_KEY,
        openai_key=OPENAI_API_KEY,
    )

    # Ollama 로컬 LLM 시도
    try:
        ollama = OllamaProvider()
        if ollama.available:
            llm.add_provider("ollama", ollama)
            logger.info(f"Ollama local LLM added: {ollama.model}")
    except Exception as e:
        logger.info(f"Ollama not available: {e}")

    memory = MemoryManager(
        db_path=MEMORY_DB_PATH,
        chroma_path=CHROMA_DB_PATH,
        embedding_model=EMBEDDING_MODEL,
    )

    computer = ComputerController(allowed_dirs=ALLOWED_DIRECTORIES)
    web = WebIntelligence(github_token=GITHUB_TOKEN)
    executor = CodeExecutor(timeout=30)

    agents = AgentManager(
        llm_manager=llm,
        web_intel=web,
        computer=computer,
        executor=executor,
    )

    voice = VoiceInterface()

    # ── Iteration 2 신규 모듈 ──
    skills = SkillLibrary(llm_manager=llm)
    logger.info(f"SkillLibrary: {len(skills._registry)} skills loaded")

    vision = VisionSystem(llm_manager=llm)
    logger.info(f"VisionSystem: screenshot={'available' if vision._screenshot_available else 'unavailable'}")

    # 임시 엔진 생성 (Orchestrator용)
    temp_jarvis = JarvisEngine(
        llm_manager=llm,
        memory_manager=memory,
        computer_controller=computer,
        web_intelligence=web,
        code_executor=executor,
        agent_manager=agents,
        voice_interface=voice,
        skill_library=skills,
        vision_system=vision,
    )

    # ToolExecutor를 Orchestrator에 전달
    orchestrator = Orchestrator(
        llm_manager=llm,
        agent_manager=agents,
        tool_executor=temp_jarvis.tool_executor,
        memory_manager=memory,
    )

    # 오케스트레이터 진행 상황 → WebSocket
    def orch_progress(task_id, event, data):
        socketio.emit("orchestrator_update", {
            "task_id": task_id,
            "event": event,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }, namespace="/jarvis")

    orchestrator.on_progress(orch_progress)

    # ── Iteration 3 신규 모듈 ──
    deep_researcher = DeepResearcher(
        web_intelligence=web,
        llm_manager=llm,
        memory_manager=memory,
    )
    logger.info("DeepResearcher initialized")

    code_intelligence = CodeIntelligence(
        llm_manager=llm,
        code_executor=executor,
    )
    logger.info("CodeIntelligence initialized")

    self_modifier = SelfModifier(llm_manager=llm)
    logger.info("SelfModifier initialized")

    # 최종 엔진 (Orchestrator + Iteration 3 포함)
    final_jarvis = JarvisEngine(
        llm_manager=llm,
        memory_manager=memory,
        computer_controller=computer,
        web_intelligence=web,
        code_executor=executor,
        agent_manager=agents,
        voice_interface=voice,
        skill_library=skills,
        vision_system=vision,
        orchestrator=orchestrator,
        # Iteration 3
        self_modifier=self_modifier,
        deep_researcher=deep_researcher,
        code_intelligence=code_intelligence,
    )

    # 자율 루프 초기화 (JarvisEngine 참조 포함)
    def auto_event_callback(event):
        socketio.emit("autonomous_event", {
            "type": event.type.value,
            "title": event.title,
            "content": event.content,
            "priority": event.priority,
            "timestamp": event.timestamp,
        }, namespace="/jarvis")

    autonomous_loop = AutonomousLoop(
        jarvis_engine=final_jarvis,
        memory_manager=memory,
        web_intelligence=web,
        skill_library=skills,
        event_callback=auto_event_callback,
    )
    final_jarvis.autonomous_loop = autonomous_loop

    logger.info("JARVIS Iteration 3 — All systems operational")
    return final_jarvis


# 전역 JARVIS 인스턴스
jarvis: JarvisEngine = None

def get_jarvis() -> JarvisEngine:
    global jarvis
    if jarvis is None:
        jarvis = create_jarvis()
    return jarvis


# ==================== HTTP 라우트 ====================

@app.route("/jarvis")
def jarvis_ui():
    """JARVIS 메인 UI"""
    return render_template("jarvis/index.html")


@app.route("/jarvis/api/status")
def api_status():
    """JARVIS 상태 조회"""
    try:
        j = get_jarvis()
        status = j.get_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/chat", methods=["POST"])
def api_chat():
    """채팅 API"""
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        j = get_jarvis()
        result = j.chat(user_input)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/stream", methods=["POST"])
def api_stream():
    """스트리밍 채팅 API (SSE)"""
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    def generate():
        j = get_jarvis()
        yield "data: " + json.dumps({"type": "start"}) + "\n\n"

        full_response = ""
        for chunk in j.stream_chat(user_input):
            full_response += chunk
            yield "data: " + json.dumps({"type": "chunk", "content": chunk}) + "\n\n"

        yield "data: " + json.dumps({"type": "done", "full": full_response}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/jarvis/api/command", methods=["POST"])
def api_command():
    """명령 실행 API"""
    try:
        data = request.get_json()
        command = data.get("command", "").strip()

        if not command:
            return jsonify({"error": "Empty command"}), 400

        j = get_jarvis()
        result = j.execute_command(command)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/system")
def api_system():
    """시스템 정보 API"""
    try:
        j = get_jarvis()
        info = j.computer.get_system_info()
        processes = j.computer.get_running_processes(top_n=15)
        return jsonify({"system": info, "processes": processes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/memory")
def api_memory():
    """메모리 조회 API"""
    try:
        j = get_jarvis()
        stats = j.memory.get_stats()
        history = j.memory.get_conversation_history(limit=20)
        knowledge = j.memory.get_knowledge(limit=10)
        tasks = j.memory.get_task_history(limit=10)

        return jsonify({
            "stats": stats,
            "recent_history": history[-10:],
            "knowledge": knowledge,
            "recent_tasks": tasks,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/search", methods=["POST"])
def api_search():
    """검색 API"""
    try:
        data = request.get_json()
        query = data.get("query", "")
        search_type = data.get("type", "web")  # web, github, arxiv, news

        j = get_jarvis()
        if search_type == "web":
            results = j.web.search_web(query, max_results=10)
        elif search_type == "github":
            results = j.web.search_github(query, max_results=10)
        elif search_type == "arxiv":
            results = j.web.search_arxiv(query, max_results=10)
        elif search_type == "news":
            results = j.web.search_news(query, max_results=10)
        else:
            results = j.web.comprehensive_search(query)

        return jsonify({"query": query, "type": search_type, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/execute", methods=["POST"])
def api_execute():
    """코드 실행 API"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        lang = data.get("lang", "python")

        j = get_jarvis()
        if lang == "python":
            result = j.executor.execute_python(code)
        elif lang == "shell":
            result = j.executor.execute_shell(code)
        else:
            result = {"error": f"Unsupported language: {lang}"}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/agent", methods=["POST"])
def api_agent():
    """에이전트 실행 API"""
    try:
        data = request.get_json()
        task = data.get("task", "")
        agent_type = data.get("agent", None)

        j = get_jarvis()
        if agent_type:
            from jarvis.agents.agent_manager import AgentType
            at = AgentType(agent_type)
            result = j.agents.run_agent(task, agent_type=at)
        else:
            result = j.agents.run_agent(task)

        return jsonify({"result": result, "task": task})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/trends")
def api_trends():
    """AI 트렌드 API"""
    try:
        j = get_jarvis()
        trends = j.web.get_ai_trends()
        return jsonify(trends)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/reason", methods=["POST"])
def api_reason():
    """고급 추론 API"""
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        strategy = data.get("strategy", None)

        j = get_jarvis()
        if not j.reasoning:
            return jsonify({"error": "Reasoning engine not available"}), 503

        from jarvis.core.reasoning import ReasoningStrategy
        if strategy:
            try:
                strat = ReasoningStrategy(strategy)
            except ValueError:
                strat = j.reasoning.select_strategy(problem)
        else:
            strat = j.reasoning.select_strategy(problem)

        result = j.reasoning.reason(problem, strategy=strat)
        return jsonify({
            "strategy": result.strategy,
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "steps": len(result.steps),
            "duration": result.duration,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/feedback", methods=["POST"])
def api_feedback():
    """피드백 제출 API"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if j.improvement:
            j.improvement.record_feedback(
                session_id=j.memory.session_id,
                query=data.get("query", ""),
                response=data.get("response", ""),
                rating=int(data.get("rating", 3)),
                feedback_text=data.get("feedback", ""),
            )
            return jsonify({"success": True, "message": "피드백이 기록되었습니다."})
        return jsonify({"success": True, "message": "Self-improvement disabled"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/knowledge")
def api_knowledge():
    """지식 베이스 API"""
    try:
        j = get_jarvis()
        knowledge = j.memory.get_knowledge(limit=20)
        status = j.knowledge_updater.get_status() if j.knowledge_updater else {}
        return jsonify({"knowledge": knowledge, "updater_status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/update_knowledge", methods=["POST"])
def api_update_knowledge():
    """지식 즉시 갱신"""
    try:
        j = get_jarvis()
        if j.knowledge_updater:
            result = j.knowledge_updater.run_update_cycle()
            return jsonify(result)
        return jsonify({"error": "Knowledge updater not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/performance")
def api_performance():
    """성능 지표 API"""
    try:
        j = get_jarvis()
        if j.improvement:
            report = j.improvement.get_performance_report()
            patterns = j.improvement.get_learned_patterns()
            suggestions = j.improvement.suggest_improvements()
            return jsonify({
                "report": report,
                "learned_patterns": patterns,
                "suggestions": suggestions,
            })
        return jsonify({"error": "Self-improvement not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/greet")
def api_greet():
    try:
        j = get_jarvis()
        return jsonify({"greeting": j.greet()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 2: 스킬 API ====================

@app.route("/jarvis/api/skills")
def api_skills():
    try:
        j = get_jarvis()
        if not j.skills:
            return jsonify({"error": "SkillLibrary not available"}), 503
        category = request.args.get("category")
        return jsonify({
            "skills": j.skills.list_skills(category=category),
            "stats": j.skills.get_stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/skills/run", methods=["POST"])
def api_skill_run():
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.skills:
            return jsonify({"error": "SkillLibrary not available"}), 503
        result = j.skills.run(data.get("skill_name", ""), **(data.get("params", {})))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/skills/create", methods=["POST"])
def api_skill_create():
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.skills:
            return jsonify({"error": "SkillLibrary not available"}), 503
        result = j.skills.create_skill(
            description=data.get("description", ""),
            category=data.get("category", "custom"),
            tags=data.get("tags", []),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/skills/search")
def api_skill_search():
    try:
        q = request.args.get("q", "")
        j = get_jarvis()
        if not j.skills:
            return jsonify({"error": "SkillLibrary not available"}), 503
        return jsonify({"results": j.skills.search_skills(q)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 2: 오케스트레이터 API ====================

@app.route("/jarvis/api/goal", methods=["POST"])
def api_goal():
    """복잡한 목표 자율 실행"""
    try:
        data = request.get_json()
        goal = data.get("goal", "")
        background = data.get("background", False)
        j = get_jarvis()
        if not j.orchestrator:
            return jsonify({"error": "Orchestrator not available"}), 503
        result = j.orchestrator.execute_goal(goal, background=background)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/tasks")
def api_tasks():
    try:
        j = get_jarvis()
        if not j.orchestrator:
            return jsonify({"error": "Orchestrator not available"}), 503
        return jsonify({
            "tasks": j.orchestrator.list_tasks(),
            "stats": j.orchestrator.get_stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/tasks/<task_id>")
def api_task_get(task_id):
    try:
        j = get_jarvis()
        if not j.orchestrator:
            return jsonify({"error": "Orchestrator not available"}), 503
        task = j.orchestrator.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        return jsonify(task)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 2: Vision API ====================

@app.route("/jarvis/api/vision/screen", methods=["POST"])
def api_vision_screen():
    """화면 캡처 및 분석"""
    try:
        data = request.get_json() or {}
        question = data.get("question", "화면에 무엇이 있나요?")
        j = get_jarvis()
        if not j.vision:
            return jsonify({"error": "VisionSystem not available"}), 503
        result = j.vision.analyze_screen(question=question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/vision/status")
def api_vision_status():
    try:
        j = get_jarvis()
        if not j.vision:
            return jsonify({"available": False})
        return jsonify(j.vision.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 2: Tool Use API ====================

@app.route("/jarvis/api/tools")
def api_tools():
    """사용 가능한 도구 목록"""
    try:
        j = get_jarvis()
        if not j.tool_executor:
            return jsonify({"tools": [], "count": 0})
        tools = j.tool_executor.get_tools()
        return jsonify({"tools": tools, "count": len(tools)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/tools/run", methods=["POST"])
def api_tool_run():
    try:
        data = request.get_json()
        tool_name = data.get("tool")
        tool_input = data.get("input", {})
        j = get_jarvis()
        if not j.tool_executor:
            return jsonify({"error": "ToolExecutor not available"}), 503
        result = j.tool_executor.execute(tool_name, tool_input)
        return jsonify({"tool": tool_name, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 3: 딥 리서치 API ====================

@app.route("/jarvis/api/research", methods=["POST"])
def api_research():
    """딥 멀티홉 리서치"""
    try:
        data = request.get_json()
        topic = data.get("topic", "")
        depth = int(data.get("depth", 2))
        j = get_jarvis()
        if not j.deep_researcher:
            return jsonify({"error": "DeepResearcher not available"}), 503
        report = j.deep_researcher.research(topic, depth=depth)
        return jsonify({
            "topic": report.topic,
            "summary": report.executive_summary,
            "findings": report.key_findings,
            "knowledge_gaps": report.knowledge_gaps,
            "follow_up": report.follow_up_questions,
            "sources_count": len(report.sources),
            "confidence": report.confidence,
            "markdown": j.deep_researcher.format_report_markdown(report),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 3: 코드 인텔리전스 API ====================

@app.route("/jarvis/api/code/generate", methods=["POST"])
def api_code_generate():
    """코드 자동 생성"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.code_intelligence:
            return jsonify({"error": "CodeIntelligence not available"}), 503
        result = j.code_intelligence.generate(
            requirement=data.get("requirement", ""),
            language=data.get("language", "python"),
            include_tests=data.get("include_tests", True),
            style=data.get("style", "production"),
        )
        return jsonify({
            "code": result.code,
            "explanation": result.explanation,
            "tests": result.tests,
            "complexity": result.complexity,
            "language": result.language,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/code/debug", methods=["POST"])
def api_code_debug():
    """코드 버그 감지 및 수정"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.code_intelligence:
            return jsonify({"error": "CodeIntelligence not available"}), 503
        code = data.get("code", "")
        language = data.get("language", "python")
        bugs = j.code_intelligence.detect_bugs(code, language)
        fixed_code = j.code_intelligence.fix_bugs(code, bugs, language) if bugs else code
        return jsonify({
            "bugs": [
                {"severity": b.severity, "location": b.location,
                 "description": b.description, "fix": b.fix_suggestion}
                for b in bugs
            ],
            "bugs_count": len(bugs),
            "fixed_code": fixed_code,
            "clean": len(bugs) == 0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/code/explain", methods=["POST"])
def api_code_explain():
    """코드 설명"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.code_intelligence:
            return jsonify({"error": "CodeIntelligence not available"}), 503
        explanation = j.code_intelligence.explain(
            code=data.get("code", ""),
            detail_level=data.get("detail", "medium"),
        )
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/code/optimize", methods=["POST"])
def api_code_optimize():
    """코드 최적화"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.code_intelligence:
            return jsonify({"error": "CodeIntelligence not available"}), 503
        result = j.code_intelligence.optimize(
            code=data.get("code", ""),
            language=data.get("language", "python"),
            goal=data.get("goal", "speed"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 3: 자율 루프 API ====================

@app.route("/jarvis/api/autoloop/status")
def api_autoloop_status():
    try:
        j = get_jarvis()
        if not j.autonomous_loop:
            return jsonify({"available": False})
        return jsonify(j.autonomous_loop.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/autoloop/start", methods=["POST"])
def api_autoloop_start():
    try:
        j = get_jarvis()
        if not j.autonomous_loop:
            return jsonify({"error": "AutonomousLoop not available"}), 503
        j.autonomous_loop.start()
        return jsonify({"success": True, "status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/autoloop/stop", methods=["POST"])
def api_autoloop_stop():
    try:
        j = get_jarvis()
        if not j.autonomous_loop:
            return jsonify({"error": "AutonomousLoop not available"}), 503
        j.autonomous_loop.stop()
        return jsonify({"success": True, "status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/autoloop/events")
def api_autoloop_events():
    try:
        j = get_jarvis()
        if not j.autonomous_loop:
            return jsonify({"events": []})
        n = int(request.args.get("n", 20))
        event_type = request.args.get("type")
        return jsonify({"events": j.autonomous_loop.get_recent_events(n, event_type)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/autoloop/trigger", methods=["POST"])
def api_autoloop_trigger():
    try:
        data = request.get_json()
        task = data.get("task", "knowledge_update")
        j = get_jarvis()
        if not j.autonomous_loop:
            return jsonify({"error": "AutonomousLoop not available"}), 503
        result = j.autonomous_loop.trigger_now(task)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 3: 자기 수정 API ====================

@app.route("/jarvis/api/selfmod/status")
def api_selfmod_status():
    try:
        j = get_jarvis()
        if not j.self_modifier:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.self_modifier.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/selfmod/analyze")
def api_selfmod_analyze():
    try:
        j = get_jarvis()
        if not j.self_modifier:
            return jsonify({"error": "SelfModifier not available"}), 503
        return jsonify({"files": j.self_modifier.analyze_all()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/selfmod/suggest", methods=["POST"])
def api_selfmod_suggest():
    try:
        data = request.get_json()
        file_path = data.get("file_path", "")
        j = get_jarvis()
        if not j.self_modifier:
            return jsonify({"error": "SelfModifier not available"}), 503
        result = j.self_modifier.suggest_improvement(file_path, focus=data.get("focus", "general"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== WebSocket 이벤트 ====================

@socketio.on("connect", namespace="/jarvis")
def ws_connect():
    """클라이언트 연결"""
    logger.info(f"WebSocket client connected")
    j = get_jarvis()
    greeting = j.greet()
    emit("jarvis_ready", {
        "message": greeting,
        "timestamp": datetime.now().isoformat(),
    })


@socketio.on("chat", namespace="/jarvis")
def ws_chat(data):
    """실시간 채팅"""
    user_input = data.get("message", "").strip()
    if not user_input:
        return

    emit("thinking", {"status": True})

    try:
        j = get_jarvis()
        result = j.chat(user_input)

        emit("response", {
            "message": result["response"],
            "tools_used": result.get("tools_used", []),
            "duration": result.get("duration", 0),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        emit("error", {"message": str(e)})
    finally:
        emit("thinking", {"status": False})


@socketio.on("stream_chat", namespace="/jarvis")
def ws_stream_chat(data):
    """스트리밍 채팅"""
    user_input = data.get("message", "").strip()
    if not user_input:
        return

    emit("thinking", {"status": True})
    emit("stream_start", {})

    try:
        j = get_jarvis()
        for chunk in j.stream_chat(user_input):
            emit("stream_chunk", {"content": chunk})
            socketio.sleep(0)
        emit("stream_end", {})
    except Exception as e:
        emit("error", {"message": str(e)})
    finally:
        emit("thinking", {"status": False})


@socketio.on("get_status", namespace="/jarvis")
def ws_status():
    """상태 요청"""
    try:
        j = get_jarvis()
        status = j.get_status()
        emit("status_update", status)
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("execute_code", namespace="/jarvis")
def ws_execute_code(data):
    """코드 실행"""
    code = data.get("code", "")
    lang = data.get("lang", "python")
    try:
        j = get_jarvis()
        if lang == "python":
            result = j.executor.execute_python(code)
        else:
            result = j.executor.execute_shell(code)
        emit("code_result", result)
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("run_goal", namespace="/jarvis")
def ws_run_goal(data):
    """오케스트레이터 목표 실행 (실시간 진행 상황)"""
    goal = data.get("goal", "").strip()
    if not goal:
        return

    emit("thinking", {"status": True})
    try:
        j = get_jarvis()
        if j.orchestrator:
            result = j.orchestrator.execute_goal(goal, background=False)
            emit("goal_result", result)
        else:
            # 폴백: 일반 채팅
            chat_result = j.chat(goal)
            emit("goal_result", {"result": chat_result["response"], "status": "done"})
    except Exception as e:
        emit("error", {"message": str(e)})
    finally:
        emit("thinking", {"status": False})


@socketio.on("run_skill", namespace="/jarvis")
def ws_run_skill(data):
    """스킬 실행"""
    skill_name = data.get("skill_name", "")
    params = data.get("params", {})
    try:
        j = get_jarvis()
        if j.skills:
            result = j.skills.run(skill_name, **params)
            emit("skill_result", {"skill": skill_name, "result": result})
        else:
            emit("error", {"message": "SkillLibrary not available"})
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("analyze_screen", namespace="/jarvis")
def ws_analyze_screen(data):
    """화면 분석"""
    question = data.get("question", "화면에 무엇이 있나요?")
    try:
        j = get_jarvis()
        if j.vision:
            result = j.vision.analyze_screen(question=question)
            emit("screen_analysis", result)
        else:
            emit("error", {"message": "VisionSystem not available"})
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("deep_research", namespace="/jarvis")
def ws_deep_research(data):
    """딥 리서치 (실시간 스트리밍)"""
    topic = data.get("topic", "").strip()
    depth = int(data.get("depth", 2))
    if not topic:
        return

    emit("thinking", {"status": True})
    emit("research_start", {"topic": topic})
    try:
        j = get_jarvis()
        if j.deep_researcher:
            for event in j.deep_researcher.research_streaming(topic, depth=depth):
                emit("research_progress", event)
                socketio.sleep(0)
        else:
            result = j.chat(f"'{topic}'에 대해 자세히 연구하고 보고서 형식으로 정리해줘")
            emit("research_progress", {"type": "done", "report": {"summary": result["response"]}})
    except Exception as e:
        emit("error", {"message": str(e)})
    finally:
        emit("thinking", {"status": False})


@socketio.on("code_generate", namespace="/jarvis")
def ws_code_generate(data):
    """코드 생성"""
    requirement = data.get("requirement", "")
    language = data.get("language", "python")
    emit("thinking", {"status": True})
    try:
        j = get_jarvis()
        if j.code_intelligence:
            result = j.code_intelligence.generate(requirement, language=language)
            emit("code_result", {
                "code": result.code,
                "explanation": result.explanation,
                "tests": result.tests,
                "complexity": result.complexity,
            })
        else:
            chat_result = j.chat(f"{language}로 다음을 구현해줘: {requirement}")
            emit("code_result", {"code": chat_result["response"]})
    except Exception as e:
        emit("error", {"message": str(e)})
    finally:
        emit("thinking", {"status": False})


@socketio.on("autoloop_control", namespace="/jarvis")
def ws_autoloop_control(data):
    """자율 루프 제어"""
    action = data.get("action", "status")
    try:
        j = get_jarvis()
        if not j.autonomous_loop:
            emit("autoloop_status", {"error": "Not available"})
            return
        if action == "start":
            j.autonomous_loop.start()
            emit("autoloop_status", {"status": "started"})
        elif action == "stop":
            j.autonomous_loop.stop()
            emit("autoloop_status", {"status": "stopped"})
        else:
            emit("autoloop_status", j.autonomous_loop.get_status())
    except Exception as e:
        emit("error", {"message": str(e)})


# ==================== 시스템 모니터 백그라운드 ====================

def system_monitor_thread():
    """주기적 시스템 상태 브로드캐스트"""
    while True:
        try:
            j = get_jarvis()
            sys_info = j.computer.get_system_info()
            socketio.emit("system_update", {
                "cpu": sys_info.get("cpu", {}).get("usage_percent", 0),
                "memory": sys_info.get("memory", {}).get("used_percent", 0),
                "disk": sys_info.get("disk", {}).get("used_percent", 0),
                "timestamp": datetime.now().isoformat(),
            }, namespace="/jarvis")
        except Exception:
            pass
        socketio.sleep(5)  # 5초마다


# ==================== 시작 ====================

if __name__ == "__main__":
    # JARVIS 사전 초기화
    logger.info("Starting JARVIS Iteration 3 system...")
    j = get_jarvis()

    # 자율 루프 자동 시작
    if j.autonomous_loop:
        j.autonomous_loop.start()
        logger.info("Autonomous Loop auto-started")

    # 시스템 모니터 시작
    socketio.start_background_task(system_monitor_thread)

    port = int(os.getenv("JARVIS_PORT", "5555"))
    logger.info(f"JARVIS Web Interface starting on http://localhost:{port}/jarvis")
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
