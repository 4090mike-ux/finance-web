"""
JARVIS 경험 압축 증류기 — Iteration 9
원시 경험을 재사용 가능한 일반화된 기술로 압축

영감:
  - 지식 증류 (Knowledge Distillation)
  - 에피소드 기억 (Episodic Memory)
  - 기술 추출 (Skill Extraction)
  - 유추 추론 (Analogical Reasoning)

핵심 개념:
  JARVIS는 단순히 기억하지 않는다 — 원칙을 추출한다
  원시 경험 → 패턴 발견 → 원칙 추상화 → 행동 가능한 기술
  도메인 간 유추 전이 (Transfer Learning)
"""

import json
import sqlite3
import threading
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """단일 경험 레코드"""
    episode_id: str
    input_summary: str          # 입력 요약
    process_summary: str        # 처리 과정 요약
    output_summary: str         # 출력 요약
    success: bool               # 성공 여부
    domain: str = "general"     # 도메인 (coding, research, analysis, etc.)
    duration: float = 0.0       # 처리 시간
    quality_score: float = 0.5  # 품질 점수 (0~1)
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "input_summary": self.input_summary[:200],
            "process_summary": self.process_summary[:300],
            "output_summary": self.output_summary[:200],
            "success": self.success,
            "domain": self.domain,
            "duration": self.duration,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


@dataclass
class Pattern:
    """여러 에피소드에서 발견된 반복 패턴"""
    pattern_id: str
    description: str
    frequency: int = 1          # 발생 빈도
    domains: List[str] = field(default_factory=list)
    episode_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    last_seen: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description[:300],
            "frequency": self.frequency,
            "domains": self.domains,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DistilledPrinciple:
    """여러 패턴으로부터 추상화된 원칙"""
    principle_id: str
    title: str
    principle: str              # 원칙 내용
    supporting_patterns: List[str] = field(default_factory=list)
    applicability: List[str] = field(default_factory=list)  # 적용 도메인
    confidence: float = 0.5
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "principle_id": self.principle_id,
            "title": self.title,
            "principle": self.principle[:400],
            "applicability": self.applicability,
            "confidence": round(self.confidence, 3),
            "usage_count": self.usage_count,
        }


@dataclass
class DistilledSkill:
    """원칙으로부터 결정화된 실행 가능한 기술"""
    skill_id: str
    name: str
    description: str
    domain: str
    trigger_conditions: List[str] = field(default_factory=list)  # 활성화 조건
    action_template: str = ""    # 행동 템플릿
    examples: List[str] = field(default_factory=list)
    success_rate: float = 0.5
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description[:300],
            "domain": self.domain,
            "trigger_conditions": self.trigger_conditions[:3],
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class Analogy:
    """도메인 간 유추"""
    source_domain: str
    target_domain: str
    mapping: str                # 유추 내용
    confidence: float = 0.5
    transfer_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "mapping": self.mapping[:300],
            "confidence": round(self.confidence, 3),
            "transfer_rules": self.transfer_rules[:3],
        }


# ════════════════════════════════════════════════════════════════
# 메인 경험 증류기
# ════════════════════════════════════════════════════════════════

class ExperienceDistiller:
    """
    JARVIS 경험 압축 증류기 — Iteration 9

    기능:
    - 에피소드 수집 (원시 대화/작업 기록)
    - 패턴 마이닝 (반복 패턴 발견)
    - 원칙 추출 (패턴 → 추상적 원칙)
    - 기술 결정화 (원칙 → 실행 가능 기술)
    - 도메인 간 유추 전이
    - 50 에피소드마다 자동 증류 실행
    - SQLite 영속성
    """

    AUTO_DISTILL_THRESHOLD = 50  # 자동 증류 임계값
    MIN_PATTERN_FREQUENCY = 2    # 패턴으로 인정할 최소 발생 빈도

    def __init__(
        self,
        llm_manager=None,
        db_path: Optional[str] = None,
        auto_distill: bool = True,
    ):
        self.llm = llm_manager
        self.auto_distill = auto_distill
        self._lock = threading.RLock()

        # DB 경로
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "experience_distiller.db")
        self.db_path = db_path
        self._init_db()

        # 인메모리 캐시
        self._episodes: List[Episode] = []
        self._patterns: Dict[str, Pattern] = {}
        self._principles: Dict[str, DistilledPrinciple] = {}
        self._skills: Dict[str, DistilledSkill] = {}
        self._analogies: List[Analogy] = []

        # 통계
        self._distillation_count = 0
        self._episodes_since_distill = 0

        # DB에서 기존 데이터 로드
        self._load_from_db()

        logger.info(
            f"ExperienceDistiller initialized — "
            f"{len(self._episodes)} episodes, "
            f"{len(self._principles)} principles, "
            f"{len(self._skills)} skills"
        )

    def _init_db(self):
        """SQLite 테이블 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    input_summary TEXT,
                    process_summary TEXT,
                    output_summary TEXT,
                    success INTEGER,
                    domain TEXT,
                    duration REAL,
                    quality_score REAL,
                    timestamp REAL,
                    tags TEXT
                );
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    description TEXT,
                    frequency INTEGER,
                    domains TEXT,
                    confidence REAL,
                    last_seen REAL
                );
                CREATE TABLE IF NOT EXISTS principles (
                    principle_id TEXT PRIMARY KEY,
                    title TEXT,
                    principle TEXT,
                    applicability TEXT,
                    confidence REAL,
                    usage_count INTEGER,
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    domain TEXT,
                    trigger_conditions TEXT,
                    action_template TEXT,
                    success_rate REAL,
                    created_at REAL
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"ExperienceDistiller DB init failed: {e}")

    def _load_from_db(self):
        """DB에서 기존 데이터 로드"""
        try:
            conn = sqlite3.connect(self.db_path)

            # 에피소드 (최근 200개)
            rows = conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 200"
            ).fetchall()
            for row in rows:
                try:
                    (ep_id, inp, proc, out, success, domain, duration, quality, ts, tags_json) = row
                    ep = Episode(
                        episode_id=ep_id,
                        input_summary=inp or "",
                        process_summary=proc or "",
                        output_summary=out or "",
                        success=bool(success),
                        domain=domain or "general",
                        duration=duration or 0.0,
                        quality_score=quality or 0.5,
                        timestamp=ts or time.time(),
                        tags=json.loads(tags_json) if tags_json else [],
                    )
                    self._episodes.append(ep)
                except Exception:
                    continue

            # 원칙
            rows = conn.execute("SELECT * FROM principles LIMIT 100").fetchall()
            for row in rows:
                try:
                    (pid, title, principle, applicability, conf, usage, created) = row
                    p = DistilledPrinciple(
                        principle_id=pid,
                        title=title or "",
                        principle=principle or "",
                        applicability=json.loads(applicability) if applicability else [],
                        confidence=conf or 0.5,
                        usage_count=usage or 0,
                        created_at=created or time.time(),
                    )
                    self._principles[pid] = p
                except Exception:
                    continue

            # 기술
            rows = conn.execute("SELECT * FROM skills LIMIT 100").fetchall()
            for row in rows:
                try:
                    (sid, name, desc, domain, triggers, template, success_rate, created) = row
                    s = DistilledSkill(
                        skill_id=sid,
                        name=name or "",
                        description=desc or "",
                        domain=domain or "general",
                        trigger_conditions=json.loads(triggers) if triggers else [],
                        action_template=template or "",
                        success_rate=success_rate or 0.5,
                        created_at=created or time.time(),
                    )
                    self._skills[sid] = s
                except Exception:
                    continue

            conn.close()
        except Exception as e:
            logger.debug(f"Experience distiller load failed: {e}")

    def ingest_episode(
        self,
        input_text: str,
        process_text: str,
        output_text: str,
        success: bool,
        domain: str = "general",
        duration: float = 0.0,
        quality_score: float = 0.5,
        tags: List[str] = None,
    ) -> Episode:
        """
        새 경험 에피소드 수집

        Args:
            input_text: 입력 (사용자 질문 등)
            process_text: 처리 과정 (사용된 도구, 추론 등)
            output_text: 출력 (응답)
            success: 성공 여부
            domain: 도메인 분류
        """
        ep_id = f"ep_{hashlib.md5(f'{input_text[:50]}{time.time()}'.encode()).hexdigest()[:10]}"

        episode = Episode(
            episode_id=ep_id,
            input_summary=input_text[:300],
            process_summary=process_text[:400],
            output_summary=output_text[:300],
            success=success,
            domain=self._classify_domain(input_text, domain),
            duration=duration,
            quality_score=quality_score,
            tags=tags or [],
        )

        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > 500:
                self._episodes = self._episodes[-500:]
            self._episodes_since_distill += 1

            # 자동 증류 체크
            if self.auto_distill and self._episodes_since_distill >= self.AUTO_DISTILL_THRESHOLD:
                threading.Thread(target=self._auto_distill, daemon=True).start()
                self._episodes_since_distill = 0

        # DB 저장
        threading.Thread(target=self._save_episode, args=(episode,), daemon=True).start()
        return episode

    def _classify_domain(self, text: str, default: str) -> str:
        """텍스트로부터 도메인 분류"""
        text_lower = text.lower()
        if any(w in text_lower for w in ["코드", "code", "프로그램", "함수"]):
            return "coding"
        if any(w in text_lower for w in ["연구", "논문", "research", "study"]):
            return "research"
        if any(w in text_lower for w in ["수학", "계산", "math", "formula"]):
            return "mathematics"
        if any(w in text_lower for w in ["분석", "데이터", "analysis", "data"]):
            return "analysis"
        if any(w in text_lower for w in ["글쓰기", "작성", "writing", "essay"]):
            return "writing"
        return default

    def _save_episode(self, episode: Episode):
        """에피소드 DB 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO episodes
                (episode_id, input_summary, process_summary, output_summary,
                 success, domain, duration, quality_score, timestamp, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                episode.episode_id,
                episode.input_summary,
                episode.process_summary,
                episode.output_summary,
                int(episode.success),
                episode.domain,
                episode.duration,
                episode.quality_score,
                episode.timestamp,
                json.dumps(episode.tags),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Episode save failed: {e}")

    def extract_patterns(self, episodes: List[Episode] = None) -> List[Pattern]:
        """
        에피소드에서 반복 패턴 발견

        유사 도메인, 유사 프로세스의 에피소드들에서 공통 패턴 추출
        """
        if episodes is None:
            with self._lock:
                episodes = list(self._episodes)

        if len(episodes) < 3:
            return []

        patterns = []

        if self.llm:
            try:
                patterns = self._llm_extract_patterns(episodes)
            except Exception as e:
                logger.debug(f"LLM pattern extraction failed: {e}")
                patterns = self._heuristic_extract_patterns(episodes)
        else:
            patterns = self._heuristic_extract_patterns(episodes)

        # 패턴 캐시 업데이트
        with self._lock:
            for p in patterns:
                if p.pattern_id in self._patterns:
                    self._patterns[p.pattern_id].frequency += 1
                else:
                    self._patterns[p.pattern_id] = p

        return patterns

    def _llm_extract_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """LLM 기반 패턴 추출"""
        # 최근 에피소드 요약 준비
        ep_summaries = []
        for ep in episodes[-20:]:  # 최근 20개만
            ep_summaries.append(
                f"[{ep.domain}] 입력: {ep.input_summary[:80]} → "
                f"성공: {ep.success} → 출력: {ep.output_summary[:60]}"
            )

        summaries_str = "\n".join(ep_summaries[:15])

        prompt = f"""다음 AI 상호작용 에피소드들에서 반복되는 패턴 3개를 찾으세요.

에피소드들:
{summaries_str}

JSON 배열로 응답:
[
  {{
    "description": "패턴 설명",
    "domains": ["관련 도메인"],
    "confidence": 0.0~1.0
  }}
]

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=400, temperature=0.4)
        patterns = []

        if response:
            import re
            match = re.search(r'\[[\s\S]*?\]', response)
            if match:
                data_list = json.loads(match.group())
                for i, data in enumerate(data_list[:5]):
                    pid = f"pat_{hashlib.md5(data.get('description','')[:50].encode()).hexdigest()[:8]}"
                    p = Pattern(
                        pattern_id=pid,
                        description=data.get("description", ""),
                        domains=data.get("domains", []),
                        confidence=float(data.get("confidence", 0.5)),
                    )
                    patterns.append(p)

        return patterns

    def _heuristic_extract_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """키워드 기반 패턴 추출 (폴백)"""
        # 도메인별 성공률 패턴
        domain_stats: Dict[str, Dict] = {}
        for ep in episodes:
            if ep.domain not in domain_stats:
                domain_stats[ep.domain] = {"success": 0, "total": 0}
            domain_stats[ep.domain]["total"] += 1
            if ep.success:
                domain_stats[ep.domain]["success"] += 1

        patterns = []
        for domain, stats in domain_stats.items():
            if stats["total"] >= 3:
                rate = stats["success"] / stats["total"]
                pid = f"pat_{domain}_{int(rate*100)}"
                patterns.append(Pattern(
                    pattern_id=pid,
                    description=f"{domain} 도메인 성공률 {rate:.0%} ({stats['total']}회)",
                    frequency=stats["total"],
                    domains=[domain],
                    confidence=rate,
                ))

        return patterns

    def crystallize_skill(self, pattern: Pattern) -> DistilledSkill:
        """
        패턴을 실행 가능한 기술로 결정화

        패턴의 통찰을 구체적 행동 가이드라인으로 변환
        """
        if self.llm:
            try:
                return self._llm_crystallize_skill(pattern)
            except Exception:
                pass

        # 폴백: 기본 기술 생성
        sid = f"skill_{pattern.pattern_id}"
        return DistilledSkill(
            skill_id=sid,
            name=f"Skill-{pattern.pattern_id[:6]}",
            description=f"패턴 기반 기술: {pattern.description[:200]}",
            domain=pattern.domains[0] if pattern.domains else "general",
            trigger_conditions=[pattern.description[:100]],
            success_rate=pattern.confidence,
        )

    def _llm_crystallize_skill(self, pattern: Pattern) -> DistilledSkill:
        """LLM 기반 기술 결정화"""
        prompt = f"""다음 패턴으로부터 실행 가능한 AI 기술을 설계하세요.

패턴: "{pattern.description}"
도메인: {', '.join(pattern.domains)}
신뢰도: {pattern.confidence:.2f}

JSON 형식:
{{
  "name": "기술 이름",
  "description": "기술 설명",
  "trigger_conditions": ["활성화 조건1", "조건2"],
  "action_template": "이 상황에서는 [구체적 행동]을 수행하라"
}}

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=300, temperature=0.5)
        if response:
            import re
            match = re.search(r'\{[\s\S]*?\}', response)
            if match:
                data = json.loads(match.group())
                sid = f"skill_{pattern.pattern_id}"
                skill = DistilledSkill(
                    skill_id=sid,
                    name=data.get("name", "Unknown Skill"),
                    description=data.get("description", ""),
                    domain=pattern.domains[0] if pattern.domains else "general",
                    trigger_conditions=data.get("trigger_conditions", [])[:3],
                    action_template=data.get("action_template", ""),
                    success_rate=pattern.confidence,
                )
                # DB 저장
                self._save_skill(skill)
                with self._lock:
                    self._skills[skill.skill_id] = skill
                return skill

        return self.crystallize_skill(pattern)  # 재귀 폴백

    def _save_skill(self, skill: DistilledSkill):
        """기술 DB 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO skills
                (skill_id, name, description, domain, trigger_conditions, action_template, success_rate, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                skill.skill_id, skill.name, skill.description, skill.domain,
                json.dumps(skill.trigger_conditions),
                skill.action_template, skill.success_rate, skill.created_at,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Skill save failed: {e}")

    def find_analogies(self, new_problem: str) -> List[Analogy]:
        """
        새 문제와 기존 경험 간의 유추 발견

        도메인을 넘나드는 유사성 탐지
        """
        if not self.llm or len(self._principles) == 0:
            return []

        try:
            principles_str = "\n".join([
                f"- [{p.applicability}] {p.title}: {p.principle[:100]}"
                for p in list(self._principles.values())[:5]
            ])

            prompt = f"""새 문제와 기존 원칙들 간의 유추를 찾으세요.

새 문제: "{new_problem[:200]}"

기존 원칙들:
{principles_str}

JSON 배열로 유추 1-2개:
[{{
  "source_domain": "출처 도메인",
  "target_domain": "적용 도메인",
  "mapping": "유추 설명",
  "confidence": 0.0~1.0,
  "transfer_rules": ["전이 규칙1", "규칙2"]
}}]

JSON만 출력:"""

            response = self.llm.generate(prompt, max_tokens=400, temperature=0.5)
            if response:
                import re
                match = re.search(r'\[[\s\S]*?\]', response)
                if match:
                    data_list = json.loads(match.group())
                    analogies = []
                    for data in data_list[:3]:
                        a = Analogy(
                            source_domain=data.get("source_domain", ""),
                            target_domain=data.get("target_domain", ""),
                            mapping=data.get("mapping", ""),
                            confidence=float(data.get("confidence", 0.5)),
                            transfer_rules=data.get("transfer_rules", []),
                        )
                        analogies.append(a)
                    return analogies
        except Exception as e:
            logger.debug(f"Analogy finding failed: {e}")

        return []

    def transfer_knowledge(self, source_domain: str, target_domain: str) -> List[Dict]:
        """도메인 간 지식 전이 규칙 생성"""
        source_skills = [
            s for s in self._skills.values()
            if s.domain == source_domain
        ]

        if not source_skills:
            return []

        transfer_rules = []
        for skill in source_skills[:3]:
            rule = {
                "from": source_domain,
                "to": target_domain,
                "skill": skill.name,
                "adaptation": f"{source_domain}의 '{skill.name}'을 {target_domain}에 적용: {skill.description[:100]}",
            }
            transfer_rules.append(rule)

        return transfer_rules

    def _auto_distill(self):
        """자동 증류 실행 (백그라운드)"""
        logger.info("ExperienceDistiller: auto-distillation started")
        try:
            with self._lock:
                episodes = list(self._episodes)

            # 패턴 추출
            patterns = self.extract_patterns(episodes)

            # 패턴에서 원칙 추출
            if self.llm and patterns:
                for pattern in patterns[:5]:
                    self.crystallize_skill(pattern)

            self._distillation_count += 1
            logger.info(
                f"ExperienceDistiller: distillation #{self._distillation_count} complete — "
                f"{len(patterns)} patterns, {len(self._skills)} skills"
            )
        except Exception as e:
            logger.warning(f"Auto-distillation failed: {e}")

    def get_wisdom_summary(self) -> Dict:
        """증류된 지혜 요약"""
        with self._lock:
            # 도메인별 에피소드 분포
            domain_dist: Dict[str, int] = {}
            for ep in self._episodes:
                domain_dist[ep.domain] = domain_dist.get(ep.domain, 0) + 1

            # 성공률
            success_count = sum(1 for ep in self._episodes if ep.success)
            success_rate = success_count / max(len(self._episodes), 1)

            return {
                "total_episodes": len(self._episodes),
                "total_patterns": len(self._patterns),
                "total_principles": len(self._principles),
                "total_skills": len(self._skills),
                "distillation_count": self._distillation_count,
                "success_rate": round(success_rate, 3),
                "domain_distribution": domain_dist,
                "recent_skills": [
                    s.to_dict() for s in sorted(
                        self._skills.values(),
                        key=lambda x: x.created_at,
                        reverse=True,
                    )[:5]
                ],
                "top_principles": [
                    p.to_dict() for p in sorted(
                        self._principles.values(),
                        key=lambda x: x.confidence,
                        reverse=True,
                    )[:3]
                ],
            }

    def get_stats(self) -> Dict:
        """통계"""
        with self._lock:
            return {
                "total_episodes": len(self._episodes),
                "total_patterns": len(self._patterns),
                "total_principles": len(self._principles),
                "total_skills": len(self._skills),
                "distillation_count": self._distillation_count,
                "episodes_since_last_distill": self._episodes_since_distill,
            }
