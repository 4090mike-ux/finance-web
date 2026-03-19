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

    # 최종 엔진 (Orchestrator 포함)
    jarvis = JarvisEngine(
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
    )

    logger.info("JARVIS Iteration 2 — All systems operational")
    return jarvis


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
    """단일 도구 직접 실행"""
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
    logger.info("Starting JARVIS system...")
    get_jarvis()

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
