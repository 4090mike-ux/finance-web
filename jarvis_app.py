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
    """JARVIS 엔진 생성"""
    logger.info("Initializing JARVIS...")

    llm = LLMManager(
        anthropic_key=ANTHROPIC_API_KEY,
        openai_key=OPENAI_API_KEY,
    )

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

    jarvis = JarvisEngine(
        llm_manager=llm,
        memory_manager=memory,
        computer_controller=computer,
        web_intelligence=web,
        code_executor=executor,
        agent_manager=agents,
        voice_interface=voice,
    )

    logger.info("JARVIS fully initialized - All systems operational")
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


@app.route("/jarvis/api/greet")
def api_greet():
    """JARVIS 인사"""
    try:
        j = get_jarvis()
        greeting = j.greet()
        return jsonify({"greeting": greeting})
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

    logger.info("JARVIS Web Interface starting on http://localhost:5001")
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
