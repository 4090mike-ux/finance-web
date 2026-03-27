"""
JARVIS Iteration 16: Distributed Multi-Agent Network
Multiple JARVIS instances working in parallel, coordinating via message passing.
Enables true parallel thinking — like having dozens of minds working simultaneously.
"""
import asyncio
import hashlib
import json
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    CREATIVE = "creative"
    FORECASTER = "forecaster"
    DEVIL_ADVOCATE = "devil_advocate"


class MessageType(Enum):
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    VOTE = "vote"
    CONSENSUS = "consensus"
    CRITIQUE = "critique"
    PING = "ping"
    PONG = "pong"


@dataclass
class AgentMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    receiver: str = ""  # "broadcast" for all
    type: MessageType = MessageType.TASK
    content: Any = None
    priority: int = 5  # 1=highest, 10=lowest
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentNode:
    """A single agent in the network."""
    id: str
    role: AgentRole
    specialization: str
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    active: bool = True
    tasks_completed: int = 0
    total_messages: int = 0
    reputation: float = 1.0  # Trust score 0-1
    capabilities: list[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)


class DistributedNetwork:
    """
    Multi-agent distributed reasoning network.

    Architecture:
    - 1 Coordinator (orchestrates tasks)
    - N Specialists (parallel execution)
    - 1 Synthesizer (merges results)
    - 1 Validator (checks quality)

    Consensus mechanism: Weighted voting by agent reputation.
    """

    AGENT_CONFIGS = [
        (AgentRole.COORDINATOR, "master", ["planning", "routing", "synthesis"]),
        (AgentRole.RESEARCHER, "information_gathering", ["web_search", "arxiv", "github"]),
        (AgentRole.ANALYST, "deep_analysis", ["statistics", "patterns", "insights"]),
        (AgentRole.CRITIC, "quality_control", ["fact_check", "logic_check", "bias_detect"]),
        (AgentRole.SYNTHESIZER, "knowledge_fusion", ["summarize", "connect", "conclude"]),
        (AgentRole.EXECUTOR, "implementation", ["code", "execute", "deploy"]),
        (AgentRole.VALIDATOR, "verification", ["test", "validate", "verify"]),
        (AgentRole.CREATIVE, "innovation", ["brainstorm", "ideate", "design"]),
        (AgentRole.FORECASTER, "prediction", ["trend", "forecast", "scenario"]),
        (AgentRole.DEVIL_ADVOCATE, "adversarial", ["challenge", "critique", "refute"]),
    ]

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self.nodes: dict[str, AgentNode] = {}
        self.message_bus: asyncio.Queue = None
        self.shared_memory: dict = {}
        self.task_registry: dict[str, dict] = {}
        self.consensus_log: list[dict] = []
        self.network_stats = {
            "messages_routed": 0,
            "tasks_completed": 0,
            "consensus_rounds": 0,
            "avg_quality_score": 0.0
        }
        self._initialized = False
        self._running = False
        print("[DistributedNetwork] Multi-agent network initialized")

    async def initialize(self):
        """Create all agent nodes."""
        self.message_bus = asyncio.Queue()

        for role, specialization, capabilities in self.AGENT_CONFIGS:
            node_id = f"{role.value}_{str(uuid.uuid4())[:4]}"
            node = AgentNode(
                id=node_id,
                role=role,
                specialization=specialization,
                inbox=asyncio.Queue(),
                capabilities=capabilities
            )
            self.nodes[node_id] = node

        self._initialized = True
        print(f"[DistributedNetwork] {len(self.nodes)} agents online")

    def _get_node_by_role(self, role: AgentRole) -> Optional[AgentNode]:
        for node in self.nodes.values():
            if node.role == role and node.active:
                return node
        return None

    def _get_nodes_by_roles(self, roles: list[AgentRole]) -> list[AgentNode]:
        result = []
        for role in roles:
            node = self._get_node_by_role(role)
            if node:
                result.append(node)
        return result

    async def think_parallel(self, question: str, agent_roles: list[AgentRole] = None) -> dict:
        """
        Have multiple agents think about the same question in parallel.
        Returns synthesized answer with confidence scores.
        """
        if not self._initialized:
            await self.initialize()

        if agent_roles is None:
            agent_roles = [
                AgentRole.RESEARCHER,
                AgentRole.ANALYST,
                AgentRole.CRITIC,
                AgentRole.CREATIVE,
                AgentRole.FORECASTER,
            ]

        print(f"[DistributedNetwork] Parallel thinking: {len(agent_roles)} agents on '{question[:50]}'")

        # Create thinking tasks for each role
        thinking_tasks = []
        for role in agent_roles:
            task = asyncio.create_task(
                self._agent_think(role, question)
            )
            thinking_tasks.append((role, task))

        # Run in parallel
        results = {}
        for role, task in thinking_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=30.0)
                results[role.value] = result
            except asyncio.TimeoutError:
                results[role.value] = {"perspective": "timeout", "confidence": 0.0}
            except Exception as e:
                results[role.value] = {"perspective": f"error: {e}", "confidence": 0.0}

        # Synthesize all perspectives
        synthesis = await self._synthesize_perspectives(question, results)

        self.network_stats["tasks_completed"] += 1
        self.network_stats["consensus_rounds"] += 1

        return {
            "question": question,
            "agent_perspectives": results,
            "synthesis": synthesis,
            "agents_consulted": len(results),
            "confidence": self._calculate_consensus_confidence(results)
        }

    async def _agent_think(self, role: AgentRole, question: str) -> dict:
        """Have a single agent generate its perspective."""
        role_prompts = {
            AgentRole.RESEARCHER: "You are a meticulous researcher. Focus on facts, data, and evidence.",
            AgentRole.ANALYST: "You are a systems analyst. Focus on patterns, correlations, and deep analysis.",
            AgentRole.CRITIC: "You are a critical thinker. Find flaws, gaps, and questionable assumptions.",
            AgentRole.CREATIVE: "You are a creative innovator. Think outside the box with novel approaches.",
            AgentRole.FORECASTER: "You are a trend forecaster. Focus on future implications and predictions.",
            AgentRole.DEVIL_ADVOCATE: "You are a devil's advocate. Argue the opposite of conventional wisdom.",
            AgentRole.VALIDATOR: "You are a fact-checker. Verify claims and assess reliability.",
            AgentRole.EXECUTOR: "You are a practical executor. Focus on how to implement solutions.",
            AgentRole.SYNTHESIZER: "You are a synthesizer. Connect ideas and find unified conclusions.",
            AgentRole.COORDINATOR: "You are a coordinator. Plan the optimal approach.",
        }

        system = role_prompts.get(role, "You are a helpful AI assistant.")
        prompt = f"{system}\n\nQuestion: {question}\n\nProvide your unique perspective in 2-3 sentences. Be specific to your role."

        try:
            response = await self.llm.generate_async(prompt, max_tokens=400, temperature=0.7)
            confidence = 0.8 if response and len(response) > 50 else 0.3
            return {
                "role": role.value,
                "perspective": response or "No response",
                "confidence": confidence,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"role": role.value, "perspective": str(e), "confidence": 0.1}

    async def _synthesize_perspectives(self, question: str, perspectives: dict) -> str:
        """Merge all agent perspectives into a final answer."""
        persp_text = "\n".join([
            f"[{role.upper()}] {data.get('perspective', '')}"
            for role, data in perspectives.items()
            if data.get('confidence', 0) > 0.3
        ])

        prompt = f"""Multiple AI agents analyzed: "{question}"

Their perspectives:
{persp_text}

Synthesize these into a single, comprehensive answer that:
1. Integrates all valid perspectives
2. Resolves contradictions
3. Highlights the most important insights
4. Reaches a clear conclusion

Synthesis:"""

        try:
            return await self.llm.generate_async(prompt, max_tokens=1000, temperature=0.3) or "Synthesis unavailable"
        except Exception as e:
            return f"Synthesis error: {e}"

    def _calculate_consensus_confidence(self, results: dict) -> float:
        """Calculate how much agents agree."""
        confidences = [r.get("confidence", 0) for r in results.values() if isinstance(r, dict)]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    async def delegate_task(self, task: str, preferred_roles: list[AgentRole] = None) -> dict:
        """Delegate a task to the most suitable agent(s)."""
        if not self._initialized:
            await self.initialize()

        # Determine best roles for task
        if preferred_roles is None:
            preferred_roles = await self._classify_task(task)

        # Assign to primary agent
        primary_role = preferred_roles[0] if preferred_roles else AgentRole.ANALYST
        primary_node = self._get_node_by_role(primary_role)

        if not primary_node:
            return {"error": "No available agents", "task": task}

        # Execute task
        result = await self._agent_think(primary_role, task)

        # Get critique from critic agent
        critic = self._get_node_by_role(AgentRole.CRITIC)
        if critic:
            critique_prompt = f"Critique this response to '{task}':\n{result.get('perspective')}\n\nIdentify any issues (2 sentences max)."
            critique = await self._agent_think(AgentRole.CRITIC, critique_prompt)
        else:
            critique = {"perspective": "No critique available"}

        primary_node.tasks_completed += 1
        primary_node.last_active = time.time()
        self.task_registry[str(uuid.uuid4())[:8]] = {
            "task": task,
            "agent": primary_node.id,
            "result": result,
            "critique": critique,
            "timestamp": time.time()
        }

        return {
            "task": task,
            "assigned_to": primary_role.value,
            "result": result.get("perspective"),
            "critique": critique.get("perspective"),
            "confidence": result.get("confidence", 0.5)
        }

    async def _classify_task(self, task: str) -> list[AgentRole]:
        """Determine which agent roles are best for a task."""
        task_lower = task.lower()
        roles = []

        if any(kw in task_lower for kw in ["search", "find", "research", "what is", "who"]):
            roles.append(AgentRole.RESEARCHER)
        if any(kw in task_lower for kw in ["analyze", "why", "explain", "understand"]):
            roles.append(AgentRole.ANALYST)
        if any(kw in task_lower for kw in ["code", "implement", "build", "create", "write"]):
            roles.append(AgentRole.EXECUTOR)
        if any(kw in task_lower for kw in ["predict", "future", "trend", "forecast"]):
            roles.append(AgentRole.FORECASTER)
        if any(kw in task_lower for kw in ["idea", "creative", "novel", "design", "innovate"]):
            roles.append(AgentRole.CREATIVE)

        if not roles:
            roles = [AgentRole.ANALYST, AgentRole.RESEARCHER]

        return roles

    async def consensus_vote(self, proposition: str, voters: int = 5) -> dict:
        """
        Democratic consensus: multiple agents vote on a proposition.
        Returns majority decision with rationale.
        """
        if not self._initialized:
            await self.initialize()

        vote_prompt = lambda role: f"""As a {role} agent, vote YES or NO on this proposition and explain why in one sentence:

Proposition: {proposition}

Format: YES/NO - [reason]"""

        roles_to_vote = [
            AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.CRITIC,
            AgentRole.VALIDATOR, AgentRole.FORECASTER
        ][:voters]

        # Parallel voting
        vote_tasks = [
            asyncio.create_task(self._agent_think(role, vote_prompt(role.value)))
            for role in roles_to_vote
        ]

        votes = []
        for task in vote_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=20.0)
                perspective = result.get("perspective", "")
                vote = "YES" if perspective.upper().startswith("YES") else "NO"
                votes.append({
                    "vote": vote,
                    "reason": perspective,
                    "confidence": result.get("confidence", 0.5)
                })
            except Exception:
                pass

        yes_votes = sum(1 for v in votes if v["vote"] == "YES")
        no_votes = len(votes) - yes_votes
        decision = "YES" if yes_votes > no_votes else "NO"

        self.consensus_log.append({
            "proposition": proposition,
            "decision": decision,
            "yes": yes_votes,
            "no": no_votes,
            "timestamp": time.time()
        })
        self.network_stats["consensus_rounds"] += 1

        return {
            "proposition": proposition,
            "decision": decision,
            "yes_votes": yes_votes,
            "no_votes": no_votes,
            "votes": votes,
            "confidence": yes_votes / len(votes) if votes else 0.5
        }

    async def brainstorm(self, topic: str, n_ideas: int = 10) -> list[str]:
        """Generate N diverse ideas from multiple agent perspectives."""
        if not self._initialized:
            await self.initialize()

        roles = [AgentRole.CREATIVE, AgentRole.RESEARCHER, AgentRole.ANALYST,
                 AgentRole.FORECASTER, AgentRole.DEVIL_ADVOCATE]

        idea_tasks = []
        for role in roles:
            prompt = f"Generate {n_ideas // len(roles) + 1} unique ideas about: {topic}\nAs a {role.value} agent, list ideas as bullet points."
            task = asyncio.create_task(self._agent_think(role, prompt))
            idea_tasks.append(task)

        all_ideas = []
        for task in idea_tasks:
            try:
                result = await asyncio.wait_for(task, timeout=20.0)
                perspective = result.get("perspective", "")
                # Extract bullet points
                lines = [l.strip().lstrip("•-*123456789. ") for l in perspective.split('\n')
                         if l.strip() and len(l.strip()) > 10]
                all_ideas.extend(lines[:3])
            except Exception:
                pass

        return all_ideas[:n_ideas]

    def get_network_status(self) -> dict:
        active_agents = sum(1 for n in self.nodes.values() if n.active)
        return {
            "initialized": self._initialized,
            "total_agents": len(self.nodes),
            "active_agents": active_agents,
            "roles": [n.role.value for n in self.nodes.values()],
            "stats": self.network_stats,
            "consensus_decisions": len(self.consensus_log)
        }

    def get_status(self) -> dict:
        return self.get_network_status()
