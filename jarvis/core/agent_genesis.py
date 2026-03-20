"""
JARVIS 에이전트 창조 엔진 — Iteration 9
JARVIS가 새로운 특화 에이전트를 동적으로 생성

핵심 개념:
  - 에이전트 청사진 라이브러리 (템플릿)
  - 작업 분석 → 에이전트 스펙 생성 → 인스턴스화
  - 에이전트 수명 관리: 생성, 활성화, 모니터링, 은퇴
  - 에이전트 DNA: 성격, 기술, 추론 스타일 게놈
  - 적합도 평가: 작업 성과 기반 점수
  - 자연 선택: 최고 에이전트 유지, 저성과자 은퇴
  - 재귀적 에이전트 생성 (최대 깊이 3)
"""

import json
import random
import sqlite3
import threading
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 에이전트 유형 및 특성
# ════════════════════════════════════════════════════════════════

class AgentStatus(Enum):
    INCUBATING  = "incubating"   # 생성 중
    ACTIVE      = "active"       # 활성
    IDLE        = "idle"         # 대기
    RETIRED     = "retired"      # 은퇴
    EVOLVING    = "evolving"     # 진화 중


# 에이전트 특성 (DNA 트레이트)
PERSONALITY_TRAITS = [
    "analytical",    # 분석적
    "creative",      # 창의적
    "skeptical",     # 회의적
    "optimistic",    # 낙관적
    "systematic",    # 체계적
    "intuitive",     # 직관적
    "cautious",      # 신중한
    "bold",          # 대담한
]

SKILL_DOMAINS = [
    "logic",         # 논리 추론
    "mathematics",   # 수학
    "language",      # 언어 처리
    "coding",        # 코딩
    "research",      # 연구
    "creativity",    # 창의적 사고
    "planning",      # 계획 수립
    "critique",      # 비판적 분석
]

# 에이전트 청사진 템플릿
AGENT_BLUEPRINTS = {
    "analyzer": {
        "personality": ["analytical", "systematic", "skeptical"],
        "skills": ["logic", "research", "critique"],
        "reasoning_style": "systematic-deductive",
        "description": "데이터와 논리를 통한 깊이 있는 분석",
    },
    "creative": {
        "personality": ["creative", "intuitive", "optimistic"],
        "skills": ["creativity", "language", "planning"],
        "reasoning_style": "lateral-divergent",
        "description": "창의적 아이디어와 혁신적 접근",
    },
    "coder": {
        "personality": ["systematic", "analytical", "cautious"],
        "skills": ["coding", "logic", "mathematics"],
        "reasoning_style": "algorithmic-precise",
        "description": "코드 작성 및 기술적 문제 해결",
    },
    "researcher": {
        "personality": ["analytical", "systematic", "cautious"],
        "skills": ["research", "language", "logic"],
        "reasoning_style": "empirical-inductive",
        "description": "심층 조사 및 증거 기반 연구",
    },
    "critic": {
        "personality": ["skeptical", "analytical", "cautious"],
        "skills": ["critique", "logic", "language"],
        "reasoning_style": "adversarial-questioning",
        "description": "가정 도전 및 약점 발견",
    },
    "planner": {
        "personality": ["systematic", "cautious", "analytical"],
        "skills": ["planning", "logic", "research"],
        "reasoning_style": "strategic-hierarchical",
        "description": "장기 계획 및 전략 수립",
    },
    "mathematician": {
        "personality": ["analytical", "systematic", "bold"],
        "skills": ["mathematics", "logic", "coding"],
        "reasoning_style": "formal-axiomatic",
        "description": "수리적 추론 및 정형 검증",
    },
    "generalist": {
        "personality": ["intuitive", "optimistic", "bold"],
        "skills": ["logic", "language", "creativity"],
        "reasoning_style": "adaptive-holistic",
        "description": "다양한 도메인의 종합적 사고",
    },
}


# ════════════════════════════════════════════════════════════════
# 에이전트 DNA
# ════════════════════════════════════════════════════════════════

@dataclass
class AgentDNA:
    """에이전트의 유전체 — 성격과 능력 정의"""
    agent_id: str = ""
    personality_traits: List[str] = field(default_factory=list)
    skill_domains: List[str] = field(default_factory=list)
    reasoning_style: str = "adaptive"
    specialization: str = "generalist"
    mutation_rate: float = 0.1   # 돌연변이율
    generation: int = 0           # 세대

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "personality_traits": self.personality_traits,
            "skill_domains": self.skill_domains,
            "reasoning_style": self.reasoning_style,
            "specialization": self.specialization,
            "mutation_rate": self.mutation_rate,
            "generation": self.generation,
        }

    def dominant_trait(self) -> str:
        """가장 강한 성격 특성"""
        return self.personality_traits[0] if self.personality_traits else "neutral"

    def primary_skill(self) -> str:
        """주요 기술 도메인"""
        return self.skill_domains[0] if self.skill_domains else "logic"

    def mutate(self) -> 'AgentDNA':
        """DNA 돌연변이 — 새로운 특성 추가 또는 교체"""
        new_dna = AgentDNA(
            agent_id=self.agent_id,
            personality_traits=list(self.personality_traits),
            skill_domains=list(self.skill_domains),
            reasoning_style=self.reasoning_style,
            specialization=self.specialization,
            mutation_rate=self.mutation_rate,
            generation=self.generation + 1,
        )

        if random.random() < self.mutation_rate:
            # 성격 특성 돌연변이
            new_trait = random.choice(PERSONALITY_TRAITS)
            if new_trait not in new_dna.personality_traits:
                new_dna.personality_traits.append(new_trait)
            if len(new_dna.personality_traits) > 4:
                new_dna.personality_traits.pop(random.randint(0, len(new_dna.personality_traits) - 1))

        if random.random() < self.mutation_rate:
            # 기술 돌연변이
            new_skill = random.choice(SKILL_DOMAINS)
            if new_skill not in new_dna.skill_domains:
                new_dna.skill_domains.append(new_skill)

        return new_dna


# ════════════════════════════════════════════════════════════════
# 동적 에이전트
# ════════════════════════════════════════════════════════════════

@dataclass
class DynamicAgent:
    """런타임에 생성된 특화 에이전트"""
    agent_id: str
    name: str
    dna: AgentDNA
    task_description: str
    system_prompt: str = ""
    status: AgentStatus = AgentStatus.INCUBATING
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    fitness_score: float = 0.5     # 적합도 (0~1)
    tasks_completed: int = 0
    tasks_failed: int = 0
    parent_agent_id: Optional[str] = None   # 부모 에이전트 (재귀 생성)
    spawn_depth: int = 0             # 재귀 깊이

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "dna": self.dna.to_dict(),
            "task_description": self.task_description[:200],
            "status": self.status.value,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "fitness_score": round(self.fitness_score, 3),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "spawn_depth": self.spawn_depth,
            "system_prompt_preview": self.system_prompt[:100],
        }

    def execute(self, task: str, llm=None) -> Dict:
        """에이전트 작업 실행"""
        self.status = AgentStatus.ACTIVE
        self.last_active = time.time()

        try:
            if llm:
                prompt = f"{self.system_prompt}\n\n작업: {task}"
                response = llm.generate(prompt, max_tokens=800, temperature=0.7)
                self.tasks_completed += 1
                result = {
                    "success": True,
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "response": response or "응답 없음",
                    "specialization": self.dna.specialization,
                }
            else:
                result = {
                    "success": False,
                    "agent_id": self.agent_id,
                    "error": "LLM 없음 — 폴백 모드",
                    "response": f"[{self.name}] 폴백: {task[:100]}",
                }
        except Exception as e:
            self.tasks_failed += 1
            result = {"success": False, "error": str(e), "agent_id": self.agent_id}

        self.status = AgentStatus.IDLE
        return result


# ════════════════════════════════════════════════════════════════
# 메인 에이전트 창조 엔진
# ════════════════════════════════════════════════════════════════

class AgentGenesis:
    """
    JARVIS 에이전트 창조 엔진 — Iteration 9

    기능:
    - 작업 분석 → 최적 에이전트 DNA 설계
    - 에이전트 인스턴스화 (시스템 프롬프트 생성)
    - 적합도 평가 및 자연 선택
    - 재귀적 에이전트 생성 (최대 깊이 3)
    - SQLite 레지스트리
    """

    MAX_SPAWN_DEPTH = 3
    MAX_ACTIVE_AGENTS = 20
    FITNESS_THRESHOLD = 0.3  # 이 점수 미만이면 은퇴

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        db_path: Optional[str] = None,
    ):
        self.llm = llm_manager
        self.event_callback = event_callback
        self._lock = threading.RLock()

        # DB 경로
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "agent_genesis.db")
        self.db_path = db_path
        self._init_db()

        # 에이전트 레지스트리 {agent_id: DynamicAgent}
        self._registry: Dict[str, DynamicAgent] = {}

        # 통계
        self._total_created = 0
        self._total_retired = 0

        # DB에서 이전 에이전트 복원
        self._load_from_db()

        logger.info(f"AgentGenesis initialized — {len(self._registry)} agents loaded")

    def _init_db(self):
        """SQLite 에이전트 레지스트리 테이블"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT,
                    dna_json TEXT,
                    task_description TEXT,
                    system_prompt TEXT,
                    status TEXT,
                    created_at REAL,
                    last_active REAL,
                    fitness_score REAL,
                    tasks_completed INTEGER,
                    tasks_failed INTEGER,
                    spawn_depth INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"AgentGenesis DB init failed: {e}")

    def _load_from_db(self):
        """DB에서 에이전트 로드"""
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT * FROM agents WHERE status != 'retired' LIMIT 50"
            ).fetchall()
            conn.close()

            for row in rows:
                try:
                    (agent_id, name, dna_json, task_desc, sys_prompt, status,
                     created_at, last_active, fitness, completed, failed, depth) = row

                    dna_data = json.loads(dna_json) if dna_json else {}
                    dna = AgentDNA(
                        agent_id=agent_id,
                        personality_traits=dna_data.get("personality_traits", []),
                        skill_domains=dna_data.get("skill_domains", []),
                        reasoning_style=dna_data.get("reasoning_style", "adaptive"),
                        specialization=dna_data.get("specialization", "generalist"),
                        generation=dna_data.get("generation", 0),
                    )

                    agent = DynamicAgent(
                        agent_id=agent_id,
                        name=name,
                        dna=dna,
                        task_description=task_desc or "",
                        system_prompt=sys_prompt or "",
                        status=AgentStatus(status) if status else AgentStatus.IDLE,
                        created_at=created_at or time.time(),
                        last_active=last_active or time.time(),
                        fitness_score=fitness or 0.5,
                        tasks_completed=completed or 0,
                        tasks_failed=failed or 0,
                        spawn_depth=depth or 0,
                    )
                    self._registry[agent_id] = agent
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"Agent load from DB failed: {e}")

    def _save_agent(self, agent: DynamicAgent):
        """에이전트를 DB에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO agents
                (agent_id, name, dna_json, task_description, system_prompt, status,
                 created_at, last_active, fitness_score, tasks_completed, tasks_failed, spawn_depth)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                agent.agent_id, agent.name,
                json.dumps(agent.dna.to_dict()),
                agent.task_description[:500],
                agent.system_prompt[:1000],
                agent.status.value,
                agent.created_at, agent.last_active,
                agent.fitness_score, agent.tasks_completed,
                agent.tasks_failed, agent.spawn_depth,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Agent save failed: {e}")

    def genesis(self, task_description: str, parent_agent_id: Optional[str] = None) -> DynamicAgent:
        """
        새 에이전트 창조 — 메인 함수

        작업 설명 → DNA 설계 → 에이전트 인스턴스화

        Returns:
            새로 생성된 DynamicAgent
        """
        # 재귀 깊이 확인
        spawn_depth = 0
        if parent_agent_id and parent_agent_id in self._registry:
            spawn_depth = self._registry[parent_agent_id].spawn_depth + 1

        if spawn_depth > self.MAX_SPAWN_DEPTH:
            logger.warning(f"Max spawn depth reached ({self.MAX_SPAWN_DEPTH})")
            spawn_depth = self.MAX_SPAWN_DEPTH

        # DNA 설계
        dna = self.create_agent_dna(task_description)

        # 에이전트 인스턴스화
        agent = self.instantiate_agent(dna, task_description)
        agent.parent_agent_id = parent_agent_id
        agent.spawn_depth = spawn_depth

        # 레지스트리 등록
        with self._lock:
            # 최대 에이전트 수 초과 시 정리
            if len(self._registry) >= self.MAX_ACTIVE_AGENTS:
                self.evolve_agents()

            self._registry[agent.agent_id] = agent
            self._total_created += 1

        # DB 저장
        self._save_agent(agent)

        # 이벤트 알림
        self._emit_event("agent_created", {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "specialization": dna.specialization,
            "spawn_depth": spawn_depth,
        })

        logger.info(f"Agent created: {agent.name} ({dna.specialization}) depth={spawn_depth}")
        return agent

    def create_agent_dna(self, task_description: str) -> AgentDNA:
        """
        작업 설명을 분석하여 최적 에이전트 DNA 설계

        LLM 있으면 정밀 설계, 없으면 키워드 기반
        """
        if self.llm:
            try:
                return self._llm_create_dna(task_description)
            except Exception as e:
                logger.debug(f"LLM DNA creation failed: {e}")

        return self._heuristic_create_dna(task_description)

    def _llm_create_dna(self, task_description: str) -> AgentDNA:
        """LLM 기반 에이전트 DNA 설계"""
        personalities_str = ", ".join(PERSONALITY_TRAITS)
        skills_str = ", ".join(SKILL_DOMAINS)
        types_str = ", ".join(AGENT_BLUEPRINTS.keys())

        prompt = f"""다음 작업을 위한 최적 AI 에이전트의 특성을 설계하세요.

작업: "{task_description[:300]}"

JSON 형식:
{{
  "specialization": "{types_str}" 중 하나,
  "personality_traits": ["{personalities_str}" 중 2-3개],
  "skill_domains": ["{skills_str}" 중 2-3개],
  "reasoning_style": "추론 스타일 설명",
  "name": "에이전트 이름 (영어, 10자 이내)"
}}

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=300, temperature=0.6)
        if response:
            import re
            match = re.search(r'\{[\s\S]*?\}', response)
            if match:
                data = json.loads(match.group())
                agent_id = self._generate_id()
                return AgentDNA(
                    agent_id=agent_id,
                    personality_traits=data.get("personality_traits", ["analytical"])[:3],
                    skill_domains=data.get("skill_domains", ["logic"])[:3],
                    reasoning_style=data.get("reasoning_style", "adaptive"),
                    specialization=data.get("specialization", "generalist"),
                )

        return self._heuristic_create_dna(task_description)

    def _heuristic_create_dna(self, task_description: str) -> AgentDNA:
        """키워드 기반 DNA 설계 (폴백)"""
        text_lower = task_description.lower()
        agent_id = self._generate_id()

        # 작업 유형 감지
        if any(w in text_lower for w in ["코드", "code", "프로그램", "개발"]):
            blueprint = AGENT_BLUEPRINTS["coder"]
            spec = "coder"
        elif any(w in text_lower for w in ["연구", "분석", "조사", "research"]):
            blueprint = AGENT_BLUEPRINTS["researcher"]
            spec = "researcher"
        elif any(w in text_lower for w in ["창작", "아이디어", "creative", "novel"]):
            blueprint = AGENT_BLUEPRINTS["creative"]
            spec = "creative"
        elif any(w in text_lower for w in ["계획", "plan", "전략", "strategy"]):
            blueprint = AGENT_BLUEPRINTS["planner"]
            spec = "planner"
        elif any(w in text_lower for w in ["수학", "math", "계산", "statistics"]):
            blueprint = AGENT_BLUEPRINTS["mathematician"]
            spec = "mathematician"
        else:
            blueprint = AGENT_BLUEPRINTS["analyzer"]
            spec = "analyzer"

        return AgentDNA(
            agent_id=agent_id,
            personality_traits=blueprint["personality"][:3],
            skill_domains=blueprint["skills"][:3],
            reasoning_style=blueprint["reasoning_style"],
            specialization=spec,
        )

    def instantiate_agent(self, dna: AgentDNA, task_description: str) -> DynamicAgent:
        """DNA로부터 에이전트 인스턴스 생성"""
        # 시스템 프롬프트 생성
        system_prompt = self._generate_system_prompt(dna, task_description)

        # 에이전트 이름 생성
        name = self._generate_agent_name(dna, task_description)

        agent = DynamicAgent(
            agent_id=dna.agent_id,
            name=name,
            dna=dna,
            task_description=task_description,
            system_prompt=system_prompt,
            status=AgentStatus.ACTIVE,
        )

        return agent

    def _generate_system_prompt(self, dna: AgentDNA, task: str) -> str:
        """에이전트 시스템 프롬프트 생성"""
        traits_str = ", ".join(dna.personality_traits)
        skills_str = ", ".join(dna.skill_domains)
        blueprint = AGENT_BLUEPRINTS.get(dna.specialization, AGENT_BLUEPRINTS["generalist"])

        return f"""당신은 JARVIS 시스템이 {task[:100]}를 위해 특별히 창조한 전문 AI 에이전트입니다.

전문화: {dna.specialization} — {blueprint['description']}
성격 특성: {traits_str}
핵심 기술: {skills_str}
추론 스타일: {dna.reasoning_style}
세대: {dna.generation}

이 특성들을 바탕으로 주어진 작업에 최적화된 방식으로 응답하세요.
항상 당신의 전문 관점과 추론 방식을 명확히 드러내세요."""

    def _generate_agent_name(self, dna: AgentDNA, task: str) -> str:
        """에이전트 이름 생성"""
        prefix_map = {
            "analyzer": "Σ-Analyst",
            "creative": "Φ-Creative",
            "coder": "Ω-Coder",
            "researcher": "Π-Research",
            "critic": "Δ-Critic",
            "planner": "Λ-Planner",
            "mathematician": "Θ-Math",
            "generalist": "Γ-General",
        }
        prefix = prefix_map.get(dna.specialization, "ξ-Agent")
        suffix = hashlib.md5(task[:50].encode()).hexdigest()[:4].upper()
        return f"{prefix}-{suffix}"

    def _generate_id(self) -> str:
        """고유 에이전트 ID 생성"""
        return f"agt_{hashlib.md5(f'{time.time()}{random.random()}'.encode()).hexdigest()[:10]}"

    def evaluate_agent(self, agent: DynamicAgent, task_result: Dict) -> float:
        """에이전트 성과 평가 및 적합도 업데이트"""
        success = task_result.get("success", False)
        quality = task_result.get("quality", 0.5)  # 0~1

        # 적합도 업데이트 (이동 평균)
        if success:
            new_fitness = agent.fitness_score * 0.8 + quality * 0.2
        else:
            new_fitness = agent.fitness_score * 0.8 + 0.1 * 0.2

        agent.fitness_score = min(1.0, max(0.0, new_fitness))

        # DB 업데이트
        threading.Thread(target=self._save_agent, args=(agent,), daemon=True).start()

        return agent.fitness_score

    def evolve_agents(self) -> Dict:
        """
        자연 선택 — 저성과 에이전트 은퇴, 우수 에이전트 돌연변이

        Returns: 진화 결과 요약
        """
        with self._lock:
            active_agents = [
                a for a in self._registry.values()
                if a.status not in (AgentStatus.RETIRED, AgentStatus.INCUBATING)
            ]

            retired = []
            mutated = []

            for agent in active_agents:
                # 저성과 에이전트 은퇴
                if (agent.fitness_score < self.FITNESS_THRESHOLD and
                        agent.tasks_completed + agent.tasks_failed > 2):
                    agent.status = AgentStatus.RETIRED
                    self._registry.pop(agent.agent_id, None)
                    retired.append(agent.name)
                    self._total_retired += 1

            # 상위 에이전트 돌연변이 (진화)
            top_agents = sorted(
                [a for a in self._registry.values() if a.fitness_score > 0.7],
                key=lambda a: a.fitness_score,
                reverse=True,
            )[:3]

            for agent in top_agents:
                mutated_dna = agent.dna.mutate()
                agent.dna = mutated_dna
                agent.status = AgentStatus.EVOLVING
                mutated.append(agent.name)

            return {
                "retired": retired,
                "mutated": mutated,
                "remaining_agents": len(self._registry),
            }

    def get_agent_roster(self) -> Dict:
        """활성 에이전트 목록"""
        with self._lock:
            active = [
                a.to_dict() for a in self._registry.values()
                if a.status != AgentStatus.RETIRED
            ]
            return {
                "total_agents": len(active),
                "total_created": self._total_created,
                "total_retired": self._total_retired,
                "agents": sorted(active, key=lambda a: a["fitness_score"], reverse=True)[:10],
                "available_blueprints": list(AGENT_BLUEPRINTS.keys()),
            }

    def get_agent(self, agent_id: str) -> Optional[DynamicAgent]:
        """특정 에이전트 조회"""
        return self._registry.get(agent_id)

    def _emit_event(self, event_type: str, data: Dict):
        """이벤트 콜백 호출"""
        if self.event_callback:
            try:
                self.event_callback({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    def get_stats(self) -> Dict:
        """통계"""
        with self._lock:
            return {
                "total_agents": len(self._registry),
                "total_created": self._total_created,
                "total_retired": self._total_retired,
                "active_count": sum(
                    1 for a in self._registry.values()
                    if a.status == AgentStatus.ACTIVE
                ),
                "avg_fitness": round(
                    sum(a.fitness_score for a in self._registry.values()) / max(len(self._registry), 1),
                    3
                ),
            }
