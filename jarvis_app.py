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
# Iteration 4
from jarvis.core.debate_engine import DebateEngine
from jarvis.intelligence.document_processor import DocumentProcessor
from jarvis.core.prediction_engine import PredictionEngine
# Iteration 5
from jarvis.core.tree_of_thoughts import TreeOfThoughts
from jarvis.core.goal_hierarchy import GoalHierarchy
from jarvis.intelligence.knowledge_graph import KnowledgeGraph
from jarvis.core.consciousness_loop import ConsciousnessLoop
from jarvis.agents.swarm import AgentSwarm
from jarvis.core.meta_learner import MetaLearner
# Iteration 6
from jarvis.core.executive import ExecutiveController
from jarvis.intelligence.live_monitor import LiveMonitor
from jarvis.core.memory_palace import MemoryPalace
# Iteration 7
from jarvis.intelligence.causal_engine import CausalEngine
from jarvis.tools.web_agent import WebAgent
from jarvis.core.recursive_improver import RecursiveImprover
# Iteration 8
from jarvis.intelligence.hypothesis_engine import HypothesisEngine
from jarvis.intelligence.temporal_engine import TemporalEngine
from jarvis.core.global_workspace import GlobalWorkspace
# Iteration 9
from jarvis.intelligence.emotion_engine import EmotionEngine
from jarvis.core.rl_optimizer import RLOptimizer
from jarvis.intelligence.reality_simulator import RealitySimulator
from jarvis.core.agent_genesis import AgentGenesis
from jarvis.intelligence.experience_distiller import ExperienceDistiller

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
    """JARVIS 엔진 생성 — Iteration 4"""
    logger.info("Initializing JARVIS Iteration 4...")

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

    # ── Iteration 4 신규 모듈 ──
    debate_engine = DebateEngine(llm_manager=llm)
    logger.info("DebateEngine initialized (5 agents)")

    document_processor = DocumentProcessor(llm_manager=llm)
    logger.info("DocumentProcessor initialized")

    def pred_bg_callback(event):
        socketio.emit("prediction_event", event, namespace="/jarvis")

    prediction_engine = PredictionEngine(
        llm_manager=llm,
        background_callback=pred_bg_callback,
    )
    logger.info(f"PredictionEngine initialized ({prediction_engine.get_stats()['total_interactions']} patterns)")

    # ── Iteration 5 신규 모듈 ──
    tree_of_thoughts = TreeOfThoughts(llm_manager=llm)
    logger.info("TreeOfThoughts initialized")

    def goal_event_cb(event):
        socketio.emit("goal_event", event, namespace="/jarvis")

    goal_hierarchy = GoalHierarchy(
        llm_manager=llm,
        memory_manager=memory,
        event_callback=goal_event_cb,
    )
    logger.info(f"GoalHierarchy initialized: {goal_hierarchy.get_stats()['total_goals']} goals")

    knowledge_graph = KnowledgeGraph(llm_manager=llm)
    knowledge_graph.compute_pagerank()
    logger.info(f"KnowledgeGraph initialized: {knowledge_graph.get_stats()['total_nodes']} nodes")

    def consciousness_event_cb(event):
        socketio.emit("consciousness_event", event, namespace="/jarvis")

    consciousness_loop = ConsciousnessLoop(
        llm_manager=llm,
        memory_manager=memory,
        event_callback=consciousness_event_cb,
    )
    logger.info("ConsciousnessLoop initialized")

    def swarm_progress_cb(event):
        socketio.emit("swarm_event", event, namespace="/jarvis")

    agent_swarm = AgentSwarm(llm_manager=llm, progress_callback=swarm_progress_cb)
    logger.info("AgentSwarm initialized")

    meta_learner = MetaLearner(llm_manager=llm)
    logger.info(f"MetaLearner initialized")

    # 최종 엔진 (Orchestrator + Iteration 3 + Iteration 4 + Iteration 5)
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
        # Iteration 4
        debate_engine=debate_engine,
        document_processor=document_processor,
        prediction_engine=prediction_engine,
        # Iteration 5
        tree_of_thoughts=tree_of_thoughts,
        goal_hierarchy=goal_hierarchy,
        knowledge_graph=knowledge_graph,
        consciousness_loop=consciousness_loop,
        agent_swarm=agent_swarm,
        meta_learner=meta_learner,
    )
    # GoalHierarchy에 tool_executor 주입 (엔진 초기화 후)
    goal_hierarchy.tool_executor = final_jarvis.tool_executor

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

    # ── Iteration 6 신규 모듈 ──
    def live_monitor_cb(event):
        socketio.emit("live_feed_event", event, namespace="/jarvis")

    live_monitor = LiveMonitor(llm_manager=llm, event_callback=live_monitor_cb)
    live_monitor.start()
    logger.info("LiveMonitor started — real-time feeds active")

    memory_palace = MemoryPalace(llm_manager=llm)
    logger.info(f"MemoryPalace initialized: {memory_palace.get_stats()['total_memories']} memories")

    def exec_progress_cb(event):
        socketio.emit("executive_event", event, namespace="/jarvis")

    # ExecutiveController는 JarvisEngine 완성 후 생성
    final_jarvis.live_monitor = live_monitor
    final_jarvis.memory_palace = memory_palace

    executive = ExecutiveController(jarvis_engine=final_jarvis, progress_callback=exec_progress_cb)
    final_jarvis.executive = executive
    logger.info("ExecutiveController online — all systems under command")

    # ── Iteration 7 신규 모듈 ──
    causal_engine = CausalEngine(llm_manager=llm)
    final_jarvis.causal_engine = causal_engine
    logger.info(f"CausalEngine initialized: {len(causal_engine.nodes)} nodes, {len(causal_engine.edges)} edges")

    web_agent = WebAgent(llm_manager=llm)
    final_jarvis.web_agent = web_agent
    logger.info(f"WebAgent initialized — mode: {web_agent._mode}")

    def improver_cb(event):
        socketio.emit("improvement_event", event, namespace="/jarvis")

    recursive_improver = RecursiveImprover(
        jarvis_engine=final_jarvis,
        llm_manager=llm,
        event_callback=improver_cb,
    )
    final_jarvis.recursive_improver = recursive_improver
    logger.info("RecursiveImprover initialized")

    # ── Iteration 8 신규 모듈 ──
    hypothesis_engine = HypothesisEngine(
        llm_manager=llm,
        knowledge_graph=knowledge_graph,
    )
    final_jarvis.hypothesis_engine = hypothesis_engine
    logger.info(f"HypothesisEngine: {len(hypothesis_engine.hypotheses)} hypotheses, "
                f"{len(hypothesis_engine.theories)} theories")

    temporal_engine = TemporalEngine(llm_manager=llm)
    final_jarvis.temporal_engine = temporal_engine
    logger.info(f"TemporalEngine: {len(temporal_engine.events)} events")

    def gw_event_cb(event):
        socketio.emit("workspace_event", event, namespace="/jarvis")

    global_workspace = GlobalWorkspace(
        jarvis_engine=final_jarvis,
        event_callback=gw_event_cb,
    )
    global_workspace.start()
    final_jarvis.global_workspace = global_workspace
    logger.info("GlobalWorkspace started — Baars cognitive architecture active")

    logger.info("JARVIS Iteration 8 — Hypothesis Engine + Temporal Reasoning + Global Workspace 🌌")

    # ── Iteration 9 신규 모듈 ──
    def emotion_event_cb(event):
        socketio.emit("emotion_event", event, namespace="/jarvis")

    emotion_engine = EmotionEngine(llm_manager=llm, event_callback=emotion_event_cb)
    final_jarvis.emotion_engine = emotion_engine
    logger.info(f"EmotionEngine initialized — current mood: {emotion_engine.get_emotional_context()['dominant_emotion']}")

    rl_optimizer = RLOptimizer(llm_manager=llm)
    final_jarvis.rl_optimizer = rl_optimizer
    logger.info(f"RLOptimizer initialized — {rl_optimizer.get_stats()['total_steps']} steps")

    def sim_event_cb(event):
        socketio.emit("simulation_event", event, namespace="/jarvis")

    reality_simulator = RealitySimulator(llm_manager=llm, event_callback=sim_event_cb)
    final_jarvis.reality_simulator = reality_simulator
    logger.info("RealitySimulator initialized — MCTS multiverse ready")

    def genesis_event_cb(event):
        socketio.emit("genesis_event", event, namespace="/jarvis")

    agent_genesis = AgentGenesis(llm_manager=llm, event_callback=genesis_event_cb)
    final_jarvis.agent_genesis = agent_genesis
    logger.info(f"AgentGenesis initialized — {agent_genesis.get_agent_roster()['total_agents']} agents")

    experience_distiller = ExperienceDistiller(llm_manager=llm)
    final_jarvis.experience_distiller = experience_distiller
    logger.info(f"ExperienceDistiller initialized — {experience_distiller.get_wisdom_summary()['total_episodes']} episodes")

    logger.info("JARVIS Iteration 9 — 감정 + RL최적화 + 멀티버스 + 에이전트 창조 + 경험증류")
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


# ==================== Iteration 4: 멀티 에이전트 토론 API ====================

@app.route("/jarvis/api/debate", methods=["POST"])
def api_debate():
    """멀티 에이전트 토론"""
    try:
        data = request.get_json()
        question = data.get("question", "")
        fast_mode = data.get("fast_mode", False)
        agents = data.get("agents", None)
        j = get_jarvis()
        if not j.debate_engine:
            return jsonify({"error": "DebateEngine not available"}), 503
        result = j.debate_engine.debate(question, agents=agents, fast_mode=fast_mode)
        return jsonify({
            "question": result.question,
            "synthesis": result.synthesis,
            "consensus_level": result.consensus_level,
            "confidence": result.confidence,
            "dissenting_views": result.dissenting_views,
            "recommended_action": result.recommended_action,
            "duration": result.duration,
            "perspectives": [
                {"agent": p.agent_name, "role": p.role, "confidence": p.confidence,
                 "argument": p.argument[:500], "key_points": p.key_points}
                for p in result.perspectives
            ],
            "markdown": j.debate_engine.format_debate_markdown(result),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/debate/factcheck", methods=["POST"])
def api_factcheck():
    """사실 검증"""
    try:
        data = request.get_json()
        claim = data.get("claim", "")
        j = get_jarvis()
        if not j.debate_engine:
            return jsonify({"error": "DebateEngine not available"}), 503
        return jsonify(j.debate_engine.quick_fact_check(claim))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/debate/history")
def api_debate_history():
    try:
        j = get_jarvis()
        if not j.debate_engine:
            return jsonify({"history": []})
        return jsonify({"history": j.debate_engine.get_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 4: 문서 처리 API ====================

@app.route("/jarvis/api/document/analyze", methods=["POST"])
def api_document_analyze():
    """문서 분석"""
    try:
        j = get_jarvis()
        if not j.document_processor:
            return jsonify({"error": "DocumentProcessor not available"}), 503

        # 파일 경로 방식
        data = request.get_json(silent=True) or {}
        file_path = data.get("file_path", "")

        if file_path:
            result = j.document_processor.process(file_path)
        else:
            return jsonify({"error": "file_path required"}), 400

        return jsonify({
            "title": result.title,
            "summary": result.summary,
            "key_facts": result.key_facts,
            "entities": result.entities,
            "word_count": result.word_count,
            "page_count": result.page_count,
            "tables_count": len(result.tables),
            "language": result.language,
            "markdown": j.document_processor.format_markdown(result),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/document/upload", methods=["POST"])
def api_document_upload():
    """파일 업로드 후 분석"""
    try:
        j = get_jarvis()
        if not j.document_processor:
            return jsonify({"error": "DocumentProcessor not available"}), 503
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        f = request.files["file"]
        content = f.read()
        result = j.document_processor.process_bytes(content, f.filename)
        return jsonify({
            "title": result.title,
            "summary": result.summary,
            "key_facts": result.key_facts,
            "word_count": result.word_count,
            "tables_count": len(result.tables),
            "markdown": j.document_processor.format_markdown(result),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/document/ask", methods=["POST"])
def api_document_ask():
    """문서 Q&A"""
    try:
        data = request.get_json()
        j = get_jarvis()
        if not j.document_processor:
            return jsonify({"error": "DocumentProcessor not available"}), 503
        result = j.document_processor.process(data.get("file_path", ""))
        answer = j.document_processor.ask(result, data.get("question", ""))
        return jsonify({"question": data.get("question"), "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 4: 예측 엔진 API ====================

@app.route("/jarvis/api/predict")
def api_predict():
    """다음 요청 예측"""
    try:
        j = get_jarvis()
        if not j.prediction_engine:
            return jsonify({"error": "PredictionEngine not available"}), 503
        current = request.args.get("context", "")
        predictions = j.prediction_engine.predict_next(current)
        suggestions = j.prediction_engine.get_proactive_suggestions()
        return jsonify({
            "predictions": [
                {"intent": p.intent, "topic": p.topic, "action": p.suggested_action,
                 "confidence": p.confidence, "reason": p.reason}
                for p in predictions
            ],
            "suggestions": suggestions,
            "stats": j.prediction_engine.get_stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Tree of Thoughts API ====================

@app.route("/jarvis/api/tot", methods=["POST"])
def api_tot():
    """Tree of Thoughts 추론"""
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        strategy = data.get("strategy", "beam")
        branching = int(data.get("branching", 3))
        beam_width = int(data.get("beam_width", 2))
        max_depth = int(data.get("max_depth", 4))

        j = get_jarvis()
        if not j.tot:
            return jsonify({"error": "TreeOfThoughts not available"}), 503

        tree = j.tot.think(
            problem=problem,
            strategy=strategy,
            branching=branching,
            beam_width=beam_width,
            max_depth=max_depth,
        )
        return jsonify({
            "problem": tree.problem,
            "final_answer": tree.final_answer,
            "confidence": tree.confidence,
            "total_thoughts": tree.total_thoughts,
            "pruned_branches": tree.pruned_branches,
            "duration": tree.duration,
            "strategy": tree.strategy,
            "best_path": tree.get_path_contents(),
            "markdown": j.tot.format_tree_markdown(tree),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/tot/stream", methods=["POST"])
def api_tot_stream():
    """Tree of Thoughts 스트리밍 추론"""
    data = request.get_json()
    problem = data.get("problem", "")

    def generate():
        j = get_jarvis()
        if not j.tot:
            yield "data: " + json.dumps({"error": "Not available"}) + "\n\n"
            return
        for event in j.tot.think_streaming(problem, branching=int(data.get("branching", 3))):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/jarvis/api/tot/stats")
def api_tot_stats():
    try:
        j = get_jarvis()
        if not j.tot:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.tot.get_stats(), "history": j.tot.get_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Goal Hierarchy API ====================

@app.route("/jarvis/api/goals", methods=["GET"])
def api_goals_list():
    """목표 목록 조회"""
    try:
        j = get_jarvis()
        if not j.goals:
            return jsonify({"error": "GoalHierarchy not available"}), 503
        status = request.args.get("status")
        root_only = request.args.get("root_only", "true").lower() == "true"
        return jsonify({
            "goals": j.goals.list_goals(status=status, root_only=root_only),
            "stats": j.goals.get_stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/goals", methods=["POST"])
def api_goals_create():
    """목표 생성 및 자동 분해"""
    try:
        data = request.get_json()
        goal_desc = data.get("goal", "")
        priority = data.get("priority", "MEDIUM")
        auto_execute = data.get("auto_execute", False)
        j = get_jarvis()
        if not j.goals:
            return jsonify({"error": "GoalHierarchy not available"}), 503

        goal = j.goals.create_goal(
            description=goal_desc,
            priority=priority,
            auto_decompose=True,
        )

        result = {"goal_id": goal.id, "goal": j.goals.get_goal_tree(goal.id)}
        if auto_execute:
            j.goals.execute_goal_background(goal.id)
            result["execution"] = "started_background"

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/goals/<goal_id>")
def api_goal_get(goal_id):
    try:
        j = get_jarvis()
        if not j.goals:
            return jsonify({"error": "GoalHierarchy not available"}), 503
        tree = j.goals.get_goal_tree(goal_id)
        if not tree:
            return jsonify({"error": "Goal not found"}), 404
        return jsonify(tree)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/goals/<goal_id>/execute", methods=["POST"])
def api_goal_execute(goal_id):
    """목표 실행"""
    try:
        data = request.get_json() or {}
        background = data.get("background", True)
        j = get_jarvis()
        if not j.goals:
            return jsonify({"error": "GoalHierarchy not available"}), 503

        if background:
            result = j.goals.execute_goal_background(goal_id)
        else:
            result = j.goals.execute_goal(goal_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/goals/<goal_id>/cancel", methods=["POST"])
def api_goal_cancel(goal_id):
    try:
        j = get_jarvis()
        if not j.goals:
            return jsonify({"error": "GoalHierarchy not available"}), 503
        success = j.goals.cancel_goal(goal_id)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Knowledge Graph API ====================

@app.route("/jarvis/api/kg/stats")
def api_kg_stats():
    try:
        j = get_jarvis()
        if not j.kg:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.kg.get_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/nodes")
def api_kg_nodes():
    try:
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        limit = int(request.args.get("limit", 100))
        return jsonify({"nodes": j.kg.get_all_nodes(limit=limit)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/edges")
def api_kg_edges():
    try:
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        return jsonify({"edges": j.kg.get_all_edges()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/search")
def api_kg_search():
    try:
        q = request.args.get("q", "")
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        return jsonify({"query": q, "results": j.kg.semantic_search(q)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/neighbors")
def api_kg_neighbors():
    try:
        name = request.args.get("name", "")
        depth = int(request.args.get("depth", 1))
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        return jsonify(j.kg.get_neighbors(name, depth=depth))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/path")
def api_kg_path():
    try:
        src = request.args.get("from", "")
        tgt = request.args.get("to", "")
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        path = j.kg.find_path(src, tgt)
        if not path:
            return jsonify({"found": False, "from": src, "to": tgt})
        node_names = [j.kg._nodes[nid].name for nid in path.nodes if nid in j.kg._nodes]
        return jsonify({"found": True, "path": node_names, "description": path.description, "hops": len(path.nodes) - 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/reason", methods=["POST"])
def api_kg_reason():
    """지식 그래프 기반 추론"""
    try:
        data = request.get_json()
        question = data.get("question", "")
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        return jsonify(j.kg.reason(question))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/extract", methods=["POST"])
def api_kg_extract():
    """텍스트에서 지식 추출"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        source = data.get("source", "manual")
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        result = j.kg.extract_from_text(text, source=source)
        j.kg.compute_pagerank()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/kg/clusters")
def api_kg_clusters():
    try:
        j = get_jarvis()
        if not j.kg:
            return jsonify({"error": "KnowledgeGraph not available"}), 503
        return jsonify({"clusters": j.kg.get_clusters()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Consciousness API ====================

@app.route("/jarvis/api/consciousness/status")
def api_consciousness_status():
    try:
        j = get_jarvis()
        if not j.consciousness:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.consciousness.get_cognitive_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/consciousness/reflect")
def api_consciousness_reflect():
    """자기 성찰"""
    try:
        j = get_jarvis()
        if not j.consciousness:
            return jsonify({"error": "ConsciousnessLoop not available"}), 503
        return jsonify(j.consciousness.self_reflect())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/consciousness/evaluate", methods=["POST"])
def api_consciousness_evaluate():
    """응답 품질 즉시 평가"""
    try:
        data = request.get_json()
        query = data.get("query", "")
        response_text = data.get("response", "")
        deep = data.get("deep", False)
        j = get_jarvis()
        if not j.consciousness:
            return jsonify({"error": "ConsciousnessLoop not available"}), 503

        if deep:
            return jsonify(j.consciousness.deep_evaluate(query, response_text))

        eval_result = j.consciousness.evaluate_response(query, response_text)
        return jsonify({
            "quality_score": eval_result.quality_score,
            "confidence": eval_result.confidence,
            "hallucination_risk": eval_result.hallucination_risk,
            "needs_rethink": eval_result.needs_rethink,
            "rethink_reason": eval_result.rethink_reason,
            "uncertainty_flags": eval_result.uncertainty_flags,
            "contradictions": eval_result.contradictions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/consciousness/history")
def api_consciousness_history():
    try:
        j = get_jarvis()
        if not j.consciousness:
            return jsonify({"evaluations": []})
        n = int(request.args.get("n", 20))
        return jsonify({"evaluations": j.consciousness.get_recent_evaluations(n)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Agent Swarm API ====================

@app.route("/jarvis/api/swarm", methods=["POST"])
def api_swarm():
    """에이전트 스웜 실행"""
    try:
        data = request.get_json()
        goal = data.get("goal", "")
        roles = data.get("roles", None)
        max_agents = int(data.get("max_agents", 4))
        j = get_jarvis()
        if not j.swarm:
            return jsonify({"error": "AgentSwarm not available"}), 503

        result = j.swarm.execute(goal=goal, roles=roles, max_agents=max_agents)
        return jsonify({
            "goal": result.goal,
            "synthesis": result.synthesis,
            "confidence": result.confidence,
            "success_rate": result.success_rate,
            "consensus_points": result.consensus_points,
            "contradictions": result.contradictions,
            "recommended_action": result.recommended_action,
            "total_duration": result.total_duration,
            "agents": [
                {"role": a.role, "success": a.success, "confidence": a.confidence, "result": a.result[:300]}
                for a in result.agents
            ],
            "markdown": j.swarm.format_markdown(result),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/swarm/history")
def api_swarm_history():
    try:
        j = get_jarvis()
        if not j.swarm:
            return jsonify({"history": []})
        return jsonify({"history": j.swarm.get_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: Meta Learner API ====================

@app.route("/jarvis/api/meta/status")
def api_meta_status():
    try:
        j = get_jarvis()
        if not j.meta_learner:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.meta_learner.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/meta/optimize", methods=["POST"])
def api_meta_optimize():
    """전략 최적화 실행"""
    try:
        j = get_jarvis()
        if not j.meta_learner:
            return jsonify({"error": "MetaLearner not available"}), 503
        result = j.meta_learner.optimize_strategies()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 5: 초지능 통합 API ====================

@app.route("/jarvis/api/superintelligence", methods=["POST"])
def api_superintelligence():
    """
    JARVIS 초지능 모드 — 모든 Iteration 5 시스템을 통합하여
    최고 수준의 문제 해결 수행
    """
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        mode = data.get("mode", "full")  # full / fast / research

        j = get_jarvis()
        results = {"problem": problem, "mode": mode, "components": {}}

        # 1. Tree of Thoughts 추론
        if j.tot and mode in ("full", "fast"):
            strategy = "greedy" if mode == "fast" else "beam"
            tree = j.tot.think(problem, strategy=strategy, branching=3, max_depth=3)
            results["components"]["tree_of_thoughts"] = {
                "answer": tree.final_answer,
                "confidence": tree.confidence,
                "thoughts_explored": tree.total_thoughts,
            }

        # 2. 멀티 에이전트 스웜
        if j.swarm and mode == "full":
            swarm_result = j.swarm.execute(goal=problem, max_agents=4)
            results["components"]["swarm"] = {
                "synthesis": swarm_result.synthesis[:500],
                "confidence": swarm_result.confidence,
                "consensus_points": swarm_result.consensus_points[:3],
            }

        # 3. 지식 그래프 추론
        if j.kg and j.kg.get_stats()["total_nodes"] > 0:
            kg_result = j.kg.reason(problem)
            results["components"]["knowledge_graph"] = {
                "answer": kg_result.get("answer", "")[:300],
                "relevant_concepts": kg_result.get("relevant_concepts", [])[:5],
            }

        # 4. 딥 리서치 (research 모드)
        if j.deep_researcher and mode == "research":
            research = j.deep_researcher.research(problem, depth=2)
            results["components"]["deep_research"] = {
                "summary": research.executive_summary[:400],
                "key_findings": research.key_findings[:3],
            }

        # 5. 최종 통합 답변 (LLM)
        component_texts = []
        for comp_name, comp_data in results["components"].items():
            answer = comp_data.get("answer") or comp_data.get("synthesis") or comp_data.get("summary", "")
            if answer:
                component_texts.append(f"[{comp_name}]\n{answer}")

        if component_texts:
            from jarvis.llm.manager import Message
            synthesis_prompt = f"""문제: {problem}

여러 AI 시스템의 분석 결과:
{chr(10).join(component_texts[:3])}

이 모든 분석을 통합하여 최고 수준의 종합 답변을 제공하세요.
각 시스템의 핵심 인사이트를 모두 반영하되, 명확하고 실용적으로 작성하세요."""

            resp = j.llm.chat([Message(role="user", content=synthesis_prompt)], max_tokens=4096)
            results["final_answer"] = resp.content
        else:
            # 단순 채팅으로 폴백
            chat_result = j.chat(problem)
            results["final_answer"] = chat_result["response"]

        # 6. 의식 루프 평가
        if j.consciousness and results.get("final_answer"):
            eval_r = j.consciousness.evaluate_response(problem, results["final_answer"])
            results["quality"] = {
                "score": eval_r.quality_score,
                "confidence": eval_r.confidence,
                "hallucination_risk": eval_r.hallucination_risk,
            }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Superintelligence API error: {e}")
        return jsonify({"error": str(e)}), 500



# ==================== Iteration 6: Executive Controller API ====================

@app.route("/jarvis/api/executive", methods=["POST"])
def api_executive():
    """총사령관 모드 — 최적 시스템 자동 선택 및 실행"""
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        force_ultra = data.get("force_ultra", False)
        j = get_jarvis()
        if not hasattr(j, "executive") or not j.executive:
            return jsonify({"error": "ExecutiveController not available"}), 503
        result = j.executive.execute(problem, force_ultra=force_ultra)
        return jsonify({
            "problem": result.problem,
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "duration": result.total_duration,
            "systems_used": result.systems_used,
            "insights": result.insights,
            "complexity": result.plan.complexity,
            "rationale": result.plan.rationale,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/executive/stream", methods=["POST"])
def api_executive_stream():
    """총사령관 스트리밍"""
    data = request.get_json()
    problem = data.get("problem", "")

    def generate():
        j = get_jarvis()
        if not hasattr(j, "executive") or not j.executive:
            yield "data: " + json.dumps({"error": "Not available"}) + "\n\n"
            return
        for event in j.executive.execute_streaming(problem):
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/jarvis/api/executive/stats")
def api_executive_stats():
    try:
        j = get_jarvis()
        if not hasattr(j, "executive") or not j.executive:
            return jsonify({"available": False})
        return jsonify({"available": True, **j.executive.get_stats(), "history": j.executive.get_history()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 6: Live Monitor API ====================

@app.route("/jarvis/api/feed")
def api_feed():
    """실시간 피드 조회"""
    try:
        j = get_jarvis()
        monitor = getattr(j, "live_monitor", None)
        if not monitor:
            return jsonify({"error": "LiveMonitor not available"}), 503
        category = request.args.get("category")
        source = request.args.get("source")
        limit = int(request.args.get("limit", 20))
        min_score = float(request.args.get("min_score", 0))
        new_only = request.args.get("new_only", "false").lower() == "true"
        return jsonify({
            "items": monitor.get_feed(category=category, source=source, limit=limit,
                                       min_score=min_score, new_only=new_only),
            "stats": monitor.get_stats(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/feed/digest")
def api_feed_digest():
    """AI 뉴스 다이제스트"""
    try:
        j = get_jarvis()
        monitor = getattr(j, "live_monitor", None)
        if not monitor:
            return jsonify({"error": "LiveMonitor not available"}), 503
        hours = int(request.args.get("hours", 24))
        return jsonify(monitor.get_ai_digest(hours=hours))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/feed/stats")
def api_feed_stats():
    try:
        j = get_jarvis()
        monitor = getattr(j, "live_monitor", None)
        if not monitor:
            return jsonify({"available": False})
        return jsonify({"available": True, **monitor.get_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 6: Memory Palace API ====================

@app.route("/jarvis/api/palace/stats")
def api_palace_stats():
    try:
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"available": False})
        return jsonify({"available": True, **palace.get_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/palace/recall")
def api_palace_recall():
    """기억 검색"""
    try:
        q = request.args.get("q", "")
        memory_type = request.args.get("type")
        top_k = int(request.args.get("k", 5))
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"error": "MemoryPalace not available"}), 503
        memories = palace.recall(q, memory_type=memory_type, top_k=top_k)
        return jsonify({
            "query": q,
            "memories": [
                {"id": m.id, "type": m.type, "content": m.content[:300], "summary": m.summary,
                 "importance": m.importance, "strength": m.current_strength(), "tags": m.tags}
                for m in memories
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/palace/remember", methods=["POST"])
def api_palace_remember():
    """기억 저장"""
    try:
        data = request.get_json()
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"error": "MemoryPalace not available"}), 503
        mem = palace.remember(
            content=data.get("content", ""),
            memory_type=data.get("type", "semantic"),
            summary=data.get("summary", ""),
            importance=float(data.get("importance", 0.5)),
            tags=data.get("tags", []),
        )
        return jsonify({"id": mem.id, "type": mem.type, "summary": mem.summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/palace/memories")
def api_palace_memories():
    try:
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"error": "MemoryPalace not available"}), 503
        memory_type = request.args.get("type")
        limit = int(request.args.get("limit", 30))
        return jsonify({"memories": palace.get_all_memories(memory_type=memory_type, limit=limit)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/palace/consolidate", methods=["POST"])
def api_palace_consolidate():
    """기억 통합 (수면 효과)"""
    try:
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"error": "MemoryPalace not available"}), 503
        return jsonify(palace.consolidate())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/palace/context")
def api_palace_context():
    """현재 작업 컨텍스트"""
    try:
        j = get_jarvis()
        palace = getattr(j, "memory_palace", None)
        if not palace:
            return jsonify({"context": ""})
        return jsonify({
            "context": palace.get_working_context(),
            "preferences": palace.get_user_preferences(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 7: Causal Reasoning API ====================

@app.route("/jarvis/api/causal/extract", methods=["POST"])
def api_causal_extract():
    """텍스트에서 인과관계 추출"""
    try:
        data = request.get_json()
        text = data.get("text", "")
        domain = data.get("domain", "")
        j = get_jarvis()
        ce = getattr(j, "causal_engine", None)
        if not ce:
            return jsonify({"error": "CausalEngine not available"}), 503
        relations = ce.extract_causal_relations(text, domain=domain)
        return jsonify({"relations": relations, "graph_stats": ce.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/causal/counterfactual", methods=["POST"])
def api_causal_counterfactual():
    """반사실적 추론: '만약 X가 달랐다면?'"""
    try:
        data = request.get_json()
        antecedent = data.get("antecedent", "")
        consequent = data.get("consequent", "")
        j = get_jarvis()
        ce = getattr(j, "causal_engine", None)
        if not ce:
            return jsonify({"error": "CausalEngine not available"}), 503
        result = ce.counterfactual(antecedent, consequent)
        return jsonify({
            "antecedent": result.antecedent,
            "consequent": result.consequent,
            "answer": result.answer,
            "confidence": result.confidence,
            "reasoning_chain": result.reasoning_chain,
            "counterfactual_world": result.counterfactual_world,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/causal/intervene", methods=["POST"])
def api_causal_intervene():
    """개입 계획: 목표 달성을 위한 do-calculus"""
    try:
        data = request.get_json()
        goal = data.get("goal", "")
        j = get_jarvis()
        ce = getattr(j, "causal_engine", None)
        if not ce:
            return jsonify({"error": "CausalEngine not available"}), 503
        plan = ce.plan_intervention(goal)
        return jsonify({
            "goal": plan.goal,
            "target_variable": plan.target_variable,
            "interventions": plan.required_interventions,
            "expected_outcome": plan.expected_outcome,
            "side_effects": plan.side_effects,
            "feasibility": plan.feasibility,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/causal/path", methods=["GET"])
def api_causal_path():
    """인과 경로 탐색"""
    try:
        from_node = request.args.get("from", "")
        to_node = request.args.get("to", "")
        j = get_jarvis()
        ce = getattr(j, "causal_engine", None)
        if not ce:
            return jsonify({"error": "CausalEngine not available"}), 503
        paths = ce.find_causal_path(from_node, to_node)
        root_causes = ce.get_root_causes(to_node) if to_node else []
        return jsonify({"paths": paths, "root_causes": root_causes, "status": ce.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/causal/status")
def api_causal_status():
    try:
        j = get_jarvis()
        ce = getattr(j, "causal_engine", None)
        return jsonify(ce.get_status() if ce else {"available": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 7: Web Agent API ====================

@app.route("/jarvis/api/webagent/run", methods=["POST"])
def api_webagent_run():
    """자율 웹 탐색 태스크 실행"""
    try:
        data = request.get_json()
        goal = data.get("goal", "")
        start_url = data.get("url", "https://www.google.com")
        j = get_jarvis()
        wa = getattr(j, "web_agent", None)
        if not wa:
            return jsonify({"error": "WebAgent not available"}), 503
        task = wa.run(goal, start_url)
        return jsonify({
            "task_id": task.id,
            "goal": task.goal,
            "success": task.success,
            "result": task.result,
            "actions_taken": len(task.actions_taken),
            "duration": round(task.duration, 2),
            "error": task.error,
            "mode": wa._mode,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/webagent/status")
def api_webagent_status():
    try:
        j = get_jarvis()
        wa = getattr(j, "web_agent", None)
        return jsonify(wa.get_status() if wa else {"available": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 7: Recursive Improver API ====================

@app.route("/jarvis/api/improve/run", methods=["POST"])
def api_improve_run():
    """즉시 개선 사이클 실행"""
    try:
        j = get_jarvis()
        ri = getattr(j, "recursive_improver", None)
        if not ri:
            return jsonify({"error": "RecursiveImprover not available"}), 503

        def run_in_thread():
            try:
                ri.run_cycle()
            except Exception as ex:
                logger.error(f"Improvement cycle error: {ex}")

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()
        return jsonify({"success": True, "message": "개선 사이클 시작됨", "cycle": ri.cycle_count + 1})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/improve/start", methods=["POST"])
def api_improve_start():
    """자동 개선 루프 시작"""
    try:
        j = get_jarvis()
        ri = getattr(j, "recursive_improver", None)
        if not ri:
            return jsonify({"error": "RecursiveImprover not available"}), 503
        ri.start()
        return jsonify({"success": True, "message": "자동 개선 루프 시작"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/improve/stop", methods=["POST"])
def api_improve_stop():
    """자동 개선 루프 중지"""
    try:
        j = get_jarvis()
        ri = getattr(j, "recursive_improver", None)
        if not ri:
            return jsonify({"error": "RecursiveImprover not available"}), 503
        ri.stop()
        return jsonify({"success": True, "message": "자동 개선 루프 중지"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/improve/status")
def api_improve_status():
    try:
        j = get_jarvis()
        ri = getattr(j, "recursive_improver", None)
        return jsonify(ri.get_status() if ri else {"available": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 8: Hypothesis Engine API ====================

@app.route("/jarvis/api/hypothesis/observe", methods=["POST"])
def api_hyp_observe():
    try:
        data = request.get_json()
        j = get_jarvis()
        he = getattr(j, "hypothesis_engine", None)
        if not he:
            return jsonify({"error": "HypothesisEngine not available"}), 503
        obs = he.observe(data.get("content", ""), data.get("source", "api"),
                         float(data.get("reliability", 0.8)))
        return jsonify({"id": obs.id, "content": obs.content, "source": obs.source})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/hypothesis/generate", methods=["POST"])
def api_hyp_generate():
    try:
        data = request.get_json()
        j = get_jarvis()
        he = getattr(j, "hypothesis_engine", None)
        if not he:
            return jsonify({"error": "HypothesisEngine not available"}), 503
        hyps = he.generate_hypotheses(
            topic=data.get("topic", ""),
            max_hyp=int(data.get("max_hyp", 3)),
        )
        return jsonify({"hypotheses": [
            {"id": h.id, "claim": h.claim, "rationale": h.rationale,
             "prior_probability": h.prior_probability,
             "predictions": h.predictions, "domain": h.domain,
             "status": h.status.value}
            for h in hyps
        ]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/hypothesis/test", methods=["POST"])
def api_hyp_test():
    try:
        data = request.get_json()
        j = get_jarvis()
        he = getattr(j, "hypothesis_engine", None)
        if not he:
            return jsonify({"error": "HypothesisEngine not available"}), 503
        hyp = he.test_hypothesis(data.get("hyp_id", ""), data.get("evidence", ""))
        return jsonify({
            "id": hyp.id, "claim": hyp.claim,
            "posterior_probability": hyp.posterior_probability,
            "status": hyp.status.value,
            "evidence_for":     hyp.evidence_for[-3:],
            "evidence_against": hyp.evidence_against[-3:],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/hypothesis/synthesize", methods=["POST"])
def api_hyp_synthesize():
    try:
        data = request.get_json()
        j = get_jarvis()
        he = getattr(j, "hypothesis_engine", None)
        if not he:
            return jsonify({"error": "HypothesisEngine not available"}), 503
        theory = he.synthesize_theory(domain=data.get("domain", ""))
        if not theory:
            return jsonify({"error": "지지된 가설이 충분하지 않습니다 (최소 2개 필요)"}), 400
        return jsonify({"name": theory.name, "description": theory.description,
                        "confidence": theory.confidence, "domain": theory.domain,
                        "key_principles": theory.key_principles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/hypothesis/list")
def api_hyp_list():
    try:
        j = get_jarvis()
        he = getattr(j, "hypothesis_engine", None)
        if not he:
            return jsonify({"available": False})
        return jsonify({
            "hypotheses": [
                {"id": h.id, "claim": h.claim,
                 "posterior_probability": h.posterior_probability,
                 "status": h.status.value, "domain": h.domain}
                for h in sorted(he.hypotheses.values(),
                                key=lambda x: x.posterior_probability, reverse=True)[:20]
            ],
            "theories": [
                {"name": t.name, "description": t.description[:200],
                 "confidence": t.confidence, "domain": t.domain}
                for t in he.theories[-5:]
            ],
            "status": he.get_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 8: Temporal Engine API ====================

@app.route("/jarvis/api/temporal/add_event", methods=["POST"])
def api_temporal_add():
    try:
        data = request.get_json()
        j = get_jarvis()
        te = getattr(j, "temporal_engine", None)
        if not te:
            return jsonify({"error": "TemporalEngine not available"}), 503
        evt = te.add_event(
            description=data.get("description", ""),
            timestamp=data.get("timestamp"),
            domain=data.get("domain", ""),
        )
        return jsonify({"id": evt.id, "description": evt.description,
                        "timestamp": evt.timestamp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/temporal/extract", methods=["POST"])
def api_temporal_extract():
    try:
        data = request.get_json()
        j = get_jarvis()
        te = getattr(j, "temporal_engine", None)
        if not te:
            return jsonify({"error": "TemporalEngine not available"}), 503
        events = te.extract_events_from_text(data.get("text", ""),
                                              data.get("domain", ""))
        return jsonify({"events": [
            {"id": e.id, "description": e.description,
             "timestamp": e.timestamp, "domain": e.domain}
            for e in events
        ]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/temporal/patterns", methods=["POST"])
def api_temporal_patterns():
    try:
        j = get_jarvis()
        te = getattr(j, "temporal_engine", None)
        if not te:
            return jsonify({"error": "TemporalEngine not available"}), 503
        patterns = te.detect_patterns()
        return jsonify({"patterns": [
            {"type": p.pattern_type, "description": p.description,
             "confidence": p.confidence, "trend": p.trend_direction}
            for p in patterns
        ]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/temporal/predict", methods=["POST"])
def api_temporal_predict():
    try:
        data = request.get_json()
        j = get_jarvis()
        te = getattr(j, "temporal_engine", None)
        if not te:
            return jsonify({"error": "TemporalEngine not available"}), 503
        preds = te.predict_future(
            domain=data.get("domain", ""),
            horizon_days=int(data.get("horizon_days", 30)),
        )
        return jsonify({"predictions": [
            {"description": p.event_description,
             "predicted_time": p.predicted_time,
             "confidence": p.confidence, "basis": p.basis}
            for p in preds
        ]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/temporal/timeline")
def api_temporal_timeline():
    try:
        domain = request.args.get("domain", "")
        j = get_jarvis()
        te = getattr(j, "temporal_engine", None)
        if not te:
            return jsonify({"available": False})
        return jsonify({
            "timeline": te.get_timeline(domain=domain, limit=50),
            "predictions": [
                {"description": p.event_description,
                 "predicted_time": p.predicted_time,
                 "confidence": p.confidence}
                for p in sorted(te.predictions,
                                key=lambda x: x.predicted_time)[:10]
            ],
            "status": te.get_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Iteration 8: Global Workspace API ====================

@app.route("/jarvis/api/workspace/contribute", methods=["POST"])
def api_ws_contribute():
    try:
        data = request.get_json()
        j = get_jarvis()
        gw = getattr(j, "global_workspace", None)
        if not gw:
            return jsonify({"error": "GlobalWorkspace not available"}), 503
        item = gw.contribute(
            module_id=data.get("module", "user"),
            content=data.get("content", ""),
            salience_boost=float(data.get("salience_boost", 0.0)),
            tags=data.get("tags", []),
        )
        return jsonify({"item_id": item.id, "salience": item.salience})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/workspace/broadcast", methods=["POST"])
def api_ws_broadcast():
    try:
        j = get_jarvis()
        gw = getattr(j, "global_workspace", None)
        if not gw:
            return jsonify({"error": "GlobalWorkspace not available"}), 503
        evt = gw.compete_and_broadcast()
        if not evt:
            return jsonify({"message": "작업공간이 비어 있습니다"})
        return jsonify({"winner_module": evt.winner_module,
                        "salience": evt.salience,
                        "responders": evt.responders,
                        "preview": evt.content_preview})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/workspace/state")
def api_ws_state():
    try:
        j = get_jarvis()
        gw = getattr(j, "global_workspace", None)
        if not gw:
            return jsonify({"available": False})
        return jsonify({
            "workspace": gw.get_workspace_state(),
            "recent_broadcasts": gw.get_recent_broadcasts(15),
            "status": gw.get_status(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== 통합 상태 API ====================

@app.route("/jarvis/api/superintelligence/status")
def api_superintelligence_status():
    """초지능 시스템 전체 상태 (Iteration 7)"""
    try:
        j = get_jarvis()
        live_monitor = getattr(j, "live_monitor", None)
        memory_palace = getattr(j, "memory_palace", None)
        executive = getattr(j, "executive", None)
        causal_engine = getattr(j, "causal_engine", None)
        web_agent = getattr(j, "web_agent", None)
        recursive_improver = getattr(j, "recursive_improver", None)
        return jsonify({
            "iteration": 8,
            "codename": "Global Workspace & Hypothesis Engine",
            "systems": {
                "tree_of_thoughts": j.tot is not None,
                "goal_hierarchy": j.goals is not None,
                "knowledge_graph": j.kg is not None,
                "consciousness_loop": j.consciousness is not None,
                "agent_swarm": j.swarm is not None,
                "meta_learner": j.meta_learner is not None,
                "debate_engine": j.debate_engine is not None,
                "deep_researcher": j.deep_researcher is not None,
                "autonomous_loop": j.autonomous_loop is not None,
                "self_modifier": j.self_modifier is not None,
                "code_intelligence": j.code_intelligence is not None,
                "prediction_engine": j.prediction_engine is not None,
                "tool_executor": j.tool_executor is not None,
                "vision_system": j.vision is not None,
                "skill_library": j.skills is not None,
                "executive_controller": executive is not None,
                "live_monitor": live_monitor is not None,
                "memory_palace": memory_palace is not None,
                # Iteration 7
                "causal_engine": causal_engine is not None,
                "web_agent": web_agent is not None,
                "recursive_improver": recursive_improver is not None,
                # Iteration 8
                "hypothesis_engine": getattr(j, "hypothesis_engine", None) is not None,
                "temporal_engine": getattr(j, "temporal_engine", None) is not None,
                "global_workspace": getattr(j, "global_workspace", None) is not None,
            },
            "stats": {
                "kg_nodes": j.kg.get_stats()["total_nodes"] if j.kg else 0,
                "kg_edges": j.kg.get_stats()["total_edges"] if j.kg else 0,
                "goals_total": j.goals.get_stats()["total_goals"] if j.goals else 0,
                "consciousness_evals": j.consciousness.get_cognitive_status()["stats"]["total_evaluations"] if j.consciousness else 0,
                "tot_runs": j.tot.get_stats()["total_runs"] if j.tot else 0,
                "meta_strategies": len(j.meta_learner.get_status().get("strategies", {})) if j.meta_learner else 0,
                "live_feed_items": live_monitor.get_status()["feed_items"] if live_monitor else 0,
                "palace_memories": memory_palace.get_stats()["total_memories"] if memory_palace else 0,
                "executive_executions": executive.get_stats().get("total_executions", 0) if executive else 0,
                # Iteration 7
                "causal_nodes": causal_engine.get_status()["nodes"] if causal_engine else 0,
                "web_tasks": web_agent.get_status()["tasks_total"] if web_agent else 0,
                "improvement_cycles": recursive_improver.get_status()["total_cycles"] if recursive_improver else 0,
                "quality_delta": recursive_improver.get_status()["total_quality_delta"] if recursive_improver else 0.0,
                # Iteration 8
                "hypothesis_count": getattr(j, "hypothesis_engine", None) and j.hypothesis_engine.get_status()["total_hypotheses"] or 0,
                "theories_count":   getattr(j, "hypothesis_engine", None) and j.hypothesis_engine.get_status()["total_theories"] or 0,
                "timeline_events":  getattr(j, "temporal_engine", None) and j.temporal_engine.get_status()["total_events"] or 0,
                "workspace_broadcasts": getattr(j, "global_workspace", None) and j.global_workspace.get_status()["total_broadcasts"] or 0,
            },
            # For UI header stats
            "memory_palace": {"available": memory_palace is not None, "total_memories": memory_palace.get_stats()["total_memories"] if memory_palace else 0},
            "knowledge_graph": {"available": j.kg is not None, "nodes": j.kg.get_stats()["total_nodes"] if j.kg else 0},
            "live_monitor": {"available": live_monitor is not None, "is_running": live_monitor.is_running if live_monitor else False, "feed_items": live_monitor.get_status()["feed_items"] if live_monitor else 0},
            "tree_of_thoughts": {"available": j.tot is not None},
            "agent_swarm": {"available": j.swarm is not None, "agent_count": 6},
            "consciousness_loop": {"available": j.consciousness is not None},
            "goal_hierarchy": {"available": j.goals is not None, "active_goals": j.goals.get_stats().get("active_goals", 0) if j.goals else 0},
            "meta_learner": {"available": j.meta_learner is not None, "strategies": len(j.meta_learner.get_status().get("strategies", {})) if j.meta_learner else 0},
            "executive_controller": {"available": executive is not None},
            "causal_engine": causal_engine.get_status() if causal_engine else {"available": False},
            "web_agent": web_agent.get_status() if web_agent else {"available": False},
            "recursive_improver": recursive_improver.get_status() if recursive_improver else {"available": False},
            # Iteration 8
            "hypothesis_engine": getattr(j, "hypothesis_engine", None) and j.hypothesis_engine.get_status() or {"available": False},
            "temporal_engine":   getattr(j, "temporal_engine", None) and j.temporal_engine.get_status() or {"available": False},
            "global_workspace":  getattr(j, "global_workspace", None) and j.global_workspace.get_status() or {"available": False},
        })
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


@socketio.on("debate", namespace="/jarvis")
def ws_debate(data):
    """멀티 에이전트 토론 (실시간)"""
    question = data.get("question", "").strip()
    if not question:
        return
    emit("thinking", {"status": True})
    try:
        j = get_jarvis()
        if j.debate_engine:
            for event in j.debate_engine.debate_streaming(question):
                emit("debate_progress", event)
                socketio.sleep(0)
        else:
            result = j.chat(question)
            emit("debate_progress", {"type": "done", "synthesis": result["response"]})
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

@app.route("/jarvis/api/emotion")
def api_emotion():
    """감정 상태 조회"""
    try:
        j = get_jarvis()
        if hasattr(j, 'emotion_engine') and j.emotion_engine:
            return jsonify(j.emotion_engine.get_emotional_context())
        return jsonify({"error": "EmotionEngine not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/simulate", methods=["POST"])
def api_simulate():
    """현실 시뮬레이션 실행"""
    try:
        data = request.get_json()
        question = data.get("question", "")
        context = data.get("context", "")
        depth = data.get("depth", 3)
        j = get_jarvis()
        if hasattr(j, 'reality_simulator') and j.reality_simulator:
            results = j.reality_simulator.simulate(question, context, depth=depth)
            return jsonify({"success": True, "branches": [b.__dict__ if not hasattr(b, 'to_dict') else b.to_dict() for b in results[:5]]})
        return jsonify({"error": "RealitySimulator not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/genesis", methods=["POST"])
def api_genesis():
    """새 에이전트 생성"""
    try:
        data = request.get_json()
        task = data.get("task", "")
        j = get_jarvis()
        if hasattr(j, 'agent_genesis') and j.agent_genesis:
            agent = j.agent_genesis.genesis(task)
            return jsonify({"success": True, "agent": agent.to_dict()})
        return jsonify({"error": "AgentGenesis not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/agents/roster")
def api_agent_roster():
    """생성된 에이전트 목록"""
    try:
        j = get_jarvis()
        if hasattr(j, 'agent_genesis') and j.agent_genesis:
            return jsonify(j.agent_genesis.get_agent_roster())
        return jsonify({"error": "AgentGenesis not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/wisdom")
def api_wisdom():
    """경험 증류 요약"""
    try:
        j = get_jarvis()
        if hasattr(j, 'experience_distiller') and j.experience_distiller:
            return jsonify(j.experience_distiller.get_wisdom_summary())
        return jsonify({"error": "ExperienceDistiller not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/jarvis/api/rl/stats")
def api_rl_stats():
    """RL 최적화 통계"""
    try:
        j = get_jarvis()
        if hasattr(j, 'rl_optimizer') and j.rl_optimizer:
            return jsonify(j.rl_optimizer.get_stats())
        return jsonify({"error": "RLOptimizer not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    logger.info("Starting JARVIS Iteration 9 system...")
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
