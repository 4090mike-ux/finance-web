"""
JARVIS 창의 엔진 — Iteration 11
아이디어 생성, 브레인스토밍, 창의적 문제 해결을 담당한다

영감:
  - SCAMPER 기법 (창의적 사고 도구)
  - TRIZ (발명 문제 해결 이론)
  - 횡적 사고 (Lateral Thinking, Edward de Bono)
  - 아날로지 추론 (Analogical Reasoning)
  - 조합 창의성 (Combinatorial Creativity)

핵심 개념:
  JARVIS는 단순 정보 검색을 넘어 새로운 것을 창조한다
  다양한 창의 기법을 적용해 참신한 아이디어를 생성하고
  아이디어를 평가·결합·구체화한다
"""

import json
import time
import logging
import threading
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 열거형 / 데이터 클래스
# ════════════════════════════════════════════════════════════════

class CreativityTechnique(Enum):
    BRAINSTORM = "brainstorm"           # 자유 브레인스토밍
    SCAMPER = "scamper"                 # 대체/결합/적용/변형/제거/재배열
    ANALOGICAL = "analogical"           # 아날로지 사고
    REVERSE = "reverse"                 # 역발상
    SIX_HATS = "six_hats"              # 드 보노의 6가지 생각 모자
    RANDOM_STIMULUS = "random_stimulus" # 무작위 자극
    MORPHOLOGICAL = "morphological"     # 형태학적 분석
    CHALLENGE = "challenge"             # 가정 도전


@dataclass
class CreativeIdea:
    """생성된 아이디어"""
    idea_id: str
    title: str
    description: str
    technique_used: str
    originality_score: float            # 참신도 (0~1)
    feasibility_score: float            # 실현 가능성 (0~1)
    impact_score: float                 # 영향력 (0~1)
    tags: List[str] = field(default_factory=list)
    parent_ideas: List[str] = field(default_factory=list)  # 결합된 아이디어
    timestamp: float = field(default_factory=time.time)

    @property
    def combined_score(self) -> float:
        return (self.originality_score * 0.35 +
                self.feasibility_score * 0.35 +
                self.impact_score * 0.30)

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "title": self.title,
            "description": self.description,
            "technique_used": self.technique_used,
            "scores": {
                "originality": self.originality_score,
                "feasibility": self.feasibility_score,
                "impact": self.impact_score,
                "combined": self.combined_score,
            },
            "tags": self.tags,
            "parent_ideas": self.parent_ideas,
        }


@dataclass
class CreativeSession:
    """창의 세션 — 하나의 문제에 대한 탐색"""
    session_id: str
    problem: str
    ideas: List[CreativeIdea] = field(default_factory=list)
    best_ideas: List[CreativeIdea] = field(default_factory=list)
    synthesis: str = ""                 # 최종 종합 결과
    techniques_applied: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_idea(self, idea: CreativeIdea):
        self.ideas.append(idea)

    def get_top_ideas(self, n: int = 5) -> List[CreativeIdea]:
        return sorted(self.ideas, key=lambda x: x.combined_score, reverse=True)[:n]

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "problem": self.problem,
            "total_ideas": len(self.ideas),
            "best_ideas": [i.to_dict() for i in self.get_top_ideas(5)],
            "techniques_applied": self.techniques_applied,
            "synthesis": self.synthesis,
        }


# ════════════════════════════════════════════════════════════════
# 창의 엔진
# ════════════════════════════════════════════════════════════════

class CreativeEngine:
    """
    JARVIS 창의 엔진
    다양한 창의 기법으로 아이디어를 생성하고 평가한다
    """

    # 무작위 자극용 단어 풀
    RANDOM_WORDS = [
        "거울", "나비", "폭풍", "씨앗", "파도", "연결", "투명", "소용돌이",
        "도서관", "바이러스", "결정체", "네트워크", "경계", "흐름", "층위",
        "공명", "변형", "촉매", "균형", "진화"
    ]

    # 아날로지 도메인
    ANALOGY_DOMAINS = [
        "자연계 (생태계, 진화, 동물 행동)",
        "건축과 도시 설계",
        "음악과 화음 이론",
        "군사 전략",
        "인체와 의학",
        "스포츠와 게임",
        "요리와 발효",
        "천문학과 우주",
        "전기 회로",
        "물의 흐름과 수문학",
    ]

    def __init__(self, llm_manager, event_callback: Optional[Callable] = None):
        self.llm = llm_manager
        self._event_cb = event_callback
        self._sessions: Dict[str, CreativeSession] = {}
        self._idea_bank: List[CreativeIdea] = []      # 전체 아이디어 은행
        self._lock = threading.Lock()
        self._stats = {
            "sessions_created": 0,
            "ideas_generated": 0,
            "syntheses_made": 0,
        }
        logger.info("CreativeEngine initialized — SCAMPER, 6-Hats, Analogical ready")

    # ── 핵심 API ─────────────────────────────────────────────────

    def start_session(self, problem: str, techniques: Optional[List[str]] = None) -> CreativeSession:
        """
        창의 세션 시작
        problem: 해결할 문제 또는 탐색할 주제
        techniques: 사용할 기법 목록 (None이면 자동 선택)
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        session = CreativeSession(session_id=session_id, problem=problem)

        with self._lock:
            self._sessions[session_id] = session
            self._stats["sessions_created"] += 1

        # 기법 자동 선택
        if techniques is None:
            techniques = self._select_techniques(problem)

        logger.info(f"Creative session {session_id}: '{problem[:50]}' with {techniques}")
        return session

    def brainstorm(self, problem: str, n_ideas: int = 10,
                   technique: CreativityTechnique = CreativityTechnique.BRAINSTORM) -> List[CreativeIdea]:
        """브레인스토밍 실행"""
        handler = {
            CreativityTechnique.BRAINSTORM: self._brainstorm_free,
            CreativityTechnique.SCAMPER: self._brainstorm_scamper,
            CreativityTechnique.ANALOGICAL: self._brainstorm_analogical,
            CreativityTechnique.REVERSE: self._brainstorm_reverse,
            CreativityTechnique.SIX_HATS: self._brainstorm_six_hats,
            CreativityTechnique.RANDOM_STIMULUS: self._brainstorm_random_stimulus,
            CreativityTechnique.MORPHOLOGICAL: self._brainstorm_morphological,
            CreativityTechnique.CHALLENGE: self._brainstorm_challenge,
        }.get(technique, self._brainstorm_free)

        ideas = handler(problem, n_ideas)

        with self._lock:
            self._stats["ideas_generated"] += len(ideas)
            self._idea_bank.extend(ideas)

        self._emit_event("ideas_generated", {
            "technique": technique.value,
            "count": len(ideas),
            "problem": problem[:60],
        })
        return ideas

    def full_creative_session(self, problem: str) -> CreativeSession:
        """
        전체 창의 세션: 여러 기법을 순차 적용 후 종합
        가장 다양하고 풍부한 아이디어 세트를 생성한다
        """
        session = self.start_session(problem)

        # 1단계: 자유 브레인스토밍
        free_ideas = self.brainstorm(problem, n_ideas=6, technique=CreativityTechnique.BRAINSTORM)
        for idea in free_ideas:
            session.add_idea(idea)
        session.techniques_applied.append("brainstorm")

        # 2단계: 역발상
        reverse_ideas = self.brainstorm(problem, n_ideas=4, technique=CreativityTechnique.REVERSE)
        for idea in reverse_ideas:
            session.add_idea(idea)
        session.techniques_applied.append("reverse")

        # 3단계: 아날로지
        analogy_ideas = self.brainstorm(problem, n_ideas=4, technique=CreativityTechnique.ANALOGICAL)
        for idea in analogy_ideas:
            session.add_idea(idea)
        session.techniques_applied.append("analogical")

        # 4단계: SCAMPER
        scamper_ideas = self.brainstorm(problem, n_ideas=4, technique=CreativityTechnique.SCAMPER)
        for idea in scamper_ideas:
            session.add_idea(idea)
        session.techniques_applied.append("scamper")

        # 5단계: 아이디어 결합 + 종합
        top_ideas = session.get_top_ideas(5)
        session.synthesis = self._synthesize(problem, top_ideas)
        session.best_ideas = top_ideas

        with self._lock:
            self._stats["syntheses_made"] += 1

        logger.info(f"Full creative session complete: {len(session.ideas)} ideas, synthesis ready")
        return session

    def evaluate_idea(self, idea_description: str, problem: str) -> Dict:
        """아이디어 평가 (참신도/실현가능성/영향력)"""
        prompt = f"""다음 아이디어를 3가지 기준으로 평가하세요:

문제: {problem}
아이디어: {idea_description}

평가 기준:
1. 참신도 (originality): 얼마나 독창적인가? (0.0~1.0)
2. 실현가능성 (feasibility): 현실에서 구현 가능한가? (0.0~1.0)
3. 영향력 (impact): 얼마나 큰 효과를 낼 수 있는가? (0.0~1.0)

JSON 형식으로만 응답:
{{"originality": 0.8, "feasibility": 0.6, "impact": 0.7, "reasoning": "간단한 이유"}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=200)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            parsed = self._parse_json(response)
            return {
                "originality": float(parsed.get("originality", 0.5)),
                "feasibility": float(parsed.get("feasibility", 0.5)),
                "impact": float(parsed.get("impact", 0.5)),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as e:
            return {"originality": 0.5, "feasibility": 0.5, "impact": 0.5, "reasoning": str(e)}

    def combine_ideas(self, idea1: CreativeIdea, idea2: CreativeIdea, problem: str) -> CreativeIdea:
        """두 아이디어를 결합하여 새로운 아이디어 생성"""
        prompt = f"""두 아이디어를 창의적으로 결합하여 더 나은 새 아이디어를 만드세요.

문제: {problem}
아이디어 A: {idea1.title} — {idea1.description}
아이디어 B: {idea2.title} — {idea2.description}

결합된 새 아이디어:
제목: (20자 이내)
설명: (2-3문장)

JSON으로만 응답: {{"title": "...", "description": "..."}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=300)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            parsed = self._parse_json(response)
            title = parsed.get("title", f"{idea1.title} + {idea2.title}")
            description = parsed.get("description", "두 아이디어의 결합")
        except Exception:
            title = f"{idea1.title} × {idea2.title}"
            description = f"{idea1.description} + {idea2.description}"

        scores = self.evaluate_idea(description, problem)
        import uuid
        return CreativeIdea(
            idea_id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            technique_used="combination",
            originality_score=min(1.0, (idea1.originality_score + idea2.originality_score) / 2 + 0.1),
            feasibility_score=(idea1.feasibility_score + idea2.feasibility_score) / 2,
            impact_score=max(idea1.impact_score, idea2.impact_score),
            parent_ideas=[idea1.idea_id, idea2.idea_id],
        )

    # ── 창의 기법 구현 ────────────────────────────────────────────

    def _brainstorm_free(self, problem: str, n: int) -> List[CreativeIdea]:
        """자유 브레인스토밍 — 제약 없이 다양한 아이디어"""
        prompt = f"""'{problem}'에 대한 창의적인 아이디어 {n}개를 생성하세요.

규칙:
- 판단하지 말고 자유롭게
- 터무니없어 보여도 포함
- 다양한 관점에서 접근
- 각 아이디어는 고유해야 함

JSON 배열로만 응답:
[
  {{"title": "아이디어 제목", "description": "1-2문장 설명"}},
  ...
]"""
        return self._generate_ideas_from_prompt(prompt, "brainstorm", n)

    def _brainstorm_scamper(self, problem: str, n: int) -> List[CreativeIdea]:
        """SCAMPER 기법 — 기존 해결책을 변형"""
        scamper_ops = [
            ("대체(Substitute)", "기존 방법/재료/사람을 무엇으로 대체하면?"),
            ("결합(Combine)", "무엇을 결합하거나 합치면?"),
            ("적용(Adapt)", "다른 분야의 아이디어를 어떻게 적용하면?"),
            ("변형(Modify/Magnify)", "더 크게, 더 작게, 더 빠르게 만들면?"),
            ("전용(Put to other use)", "다른 용도로 사용하면?"),
            ("제거(Eliminate)", "불필요한 요소를 제거하면?"),
            ("재배열(Rearrange/Reverse)", "순서를 바꾸거나 뒤집으면?"),
        ]

        selected = random.sample(scamper_ops, min(n, len(scamper_ops)))
        prompt = f"""SCAMPER 기법으로 '{problem}'에 대한 아이디어를 생성하세요.

각 SCAMPER 조작에 대해 아이디어를 1개씩:
{chr(10).join([f"- {op}: {q}" for op, q in selected])}

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "scamper_op": "..."}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "scamper", len(selected))

    def _brainstorm_analogical(self, problem: str, n: int) -> List[CreativeIdea]:
        """아날로지 추론 — 다른 도메인에서 영감"""
        domains = random.sample(self.ANALOGY_DOMAINS, min(n, len(self.ANALOGY_DOMAINS)))
        domains_str = "\n".join([f"- {d}" for d in domains])

        prompt = f"""아날로지 사고로 '{problem}'에 대한 아이디어를 생성하세요.

다음 도메인에서 영감을 받으세요:
{domains_str}

각 도메인에서 1개씩 아이디어를 만드세요.
도메인의 원리/패턴을 문제에 적용하세요.

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "domain_inspiration": "..."}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "analogical", len(domains))

    def _brainstorm_reverse(self, problem: str, n: int) -> List[CreativeIdea]:
        """역발상 — 문제를 반대로 생각"""
        prompt = f"""역발상 기법으로 '{problem}'에 대한 아이디어를 생성하세요.

단계:
1. 문제를 반대로 뒤집어보세요 (어떻게 하면 더 나빠질까?)
2. 그 반대 아이디어에서 통찰을 찾으세요
3. {n}개의 역발상 기반 창의 아이디어를 만드세요

예: "교통 체증 해결" → 역발상: "체증을 늘리려면?" → "자동차 크기 제한 없애기" → 아이디어: "소형 특화 차선"

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "reverse_insight": "..."}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "reverse", n)

    def _brainstorm_six_hats(self, problem: str, n: int) -> List[CreativeIdea]:
        """드 보노의 6가지 생각 모자"""
        hats = [
            ("흰 모자 (사실)", "객관적인 데이터와 사실에 기반한 접근"),
            ("빨간 모자 (감정)", "직관과 감정에 따른 접근"),
            ("검은 모자 (비판)", "위험과 문제점 중심의 접근"),
            ("노란 모자 (낙관)", "긍정적인 가능성 중심의 접근"),
            ("초록 모자 (창의)", "창의적이고 혁신적인 접근"),
            ("파란 모자 (과정)", "전체 과정과 조율 중심의 접근"),
        ]
        selected_hats = random.sample(hats, min(n, len(hats)))

        prompt = f"""6가지 생각 모자 기법으로 '{problem}'을 분석하세요.

각 관점에서 1개의 구체적인 아이디어/해결책을 제시하세요:
{chr(10).join([f"- {hat}: {desc}" for hat, desc in selected_hats])}

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "hat": "모자색"}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "six_hats", len(selected_hats))

    def _brainstorm_random_stimulus(self, problem: str, n: int) -> List[CreativeIdea]:
        """무작위 자극 — 무관한 단어에서 연결고리 찾기"""
        stimuli = random.sample(self.RANDOM_WORDS, min(n, len(self.RANDOM_WORDS)))

        prompt = f"""무작위 자극 기법으로 '{problem}'에 대한 아이디어를 생성하세요.

각 단어를 자극으로 사용하여 연결고리를 찾고 아이디어를 만드세요:
{', '.join(stimuli)}

각 단어에 대해 1개 아이디어:

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "stimulus_word": "..."}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "random_stimulus", len(stimuli))

    def _brainstorm_morphological(self, problem: str, n: int) -> List[CreativeIdea]:
        """형태학적 분석 — 문제를 차원으로 분해 후 조합"""
        prompt = f"""형태학적 분석으로 '{problem}'을 분해하고 아이디어를 생성하세요.

단계:
1. 문제를 3-4개 핵심 차원/요소로 분해하세요
2. 각 차원의 가능한 변형/옵션을 나열하세요
3. 흥미로운 조합으로 {n}개 아이디어를 만드세요

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "dimensions_used": ["차원1옵션", "차원2옵션"]}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "morphological", n)

    def _brainstorm_challenge(self, problem: str, n: int) -> List[CreativeIdea]:
        """가정 도전 — 당연한 가정을 의심하기"""
        prompt = f"""가정 도전 기법으로 '{problem}'의 숨겨진 가정을 찾고 도전하세요.

단계:
1. 이 문제에서 당연히 받아들이는 가정들을 나열하세요
2. 각 가정을 반박하거나 무시하면 어떤 새로운 가능성이 열리는지 탐색하세요
3. {n}개의 가정-파괴 아이디어를 만드세요

JSON 배열로만 응답:
[{{"title": "...", "description": "...", "challenged_assumption": "어떤 가정을 깼나"}}, ...]"""
        return self._generate_ideas_from_prompt(prompt, "challenge", n)

    # ── 종합 ─────────────────────────────────────────────────────

    def _synthesize(self, problem: str, top_ideas: List[CreativeIdea]) -> str:
        """최상위 아이디어들을 종합하여 최종 권장안 생성"""
        ideas_text = "\n".join([
            f"{i+1}. [{idea.title}] {idea.description}"
            for i, idea in enumerate(top_ideas)
        ])

        prompt = f"""다음 창의적 아이디어들을 종합하여 최적의 해결책을 제시하세요.

문제: {problem}

상위 아이디어들:
{ideas_text}

종합:
1. 가장 유망한 핵심 방향 (2-3문장)
2. 즉시 실행 가능한 구체적 첫 단계
3. 중장기적 발전 방향

한국어로 간결하게 작성하세요."""

        try:
            response = self.llm.generate(prompt, max_tokens=400)
            if isinstance(response, dict):
                response = response.get("content", "")
            return response
        except Exception as e:
            return f"종합 생성 실패: {e}"

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _generate_ideas_from_prompt(self, prompt: str, technique: str, n: int) -> List[CreativeIdea]:
        """LLM에서 아이디어 목록 생성"""
        try:
            response = self.llm.generate(prompt, max_tokens=1200)
            if isinstance(response, dict):
                response = response.get("content", "[]")

            # JSON 파싱
            ideas_raw = self._parse_json_list(response)
        except Exception as e:
            logger.warning(f"아이디어 생성 실패 ({technique}): {e}")
            ideas_raw = []

        ideas = []
        import uuid
        for i, raw in enumerate(ideas_raw[:n]):
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", f"아이디어 {i+1}"))
            description = str(raw.get("description", "설명 없음"))

            # 빠른 점수 추정 (LLM 재호출 최소화)
            originality = random.uniform(0.5, 0.9)
            feasibility = random.uniform(0.4, 0.85)
            impact = random.uniform(0.5, 0.9)

            idea = CreativeIdea(
                idea_id=str(uuid.uuid4())[:8],
                title=title[:80],
                description=description[:300],
                technique_used=technique,
                originality_score=round(originality, 2),
                feasibility_score=round(feasibility, 2),
                impact_score=round(impact, 2),
            )
            ideas.append(idea)

        return ideas

    def _select_techniques(self, problem: str) -> List[str]:
        """문제 특성에 따라 적합한 기법 선택"""
        techniques = ["brainstorm", "reverse", "analogical"]
        if any(kw in problem for kw in ["개선", "혁신", "새로운", "다른"]):
            techniques.append("scamper")
        if any(kw in problem for kw in ["분석", "이해", "관점"]):
            techniques.append("six_hats")
        return techniques

    def _parse_json(self, text: str) -> Dict:
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _parse_json_list(self, text: str) -> List:
        # 코드블록 제거
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # JSON 배열 찾기
        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass
        return []

    def _emit_event(self, event_type: str, data: Dict):
        if self._event_cb:
            try:
                self._event_cb({
                    "type": f"creative_{event_type}",
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    # ── 세션 관리 ────────────────────────────────────────────────

    def get_session(self, session_id: str) -> Optional[CreativeSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict]:
        return [
            {
                "session_id": s.session_id,
                "problem": s.problem[:60],
                "idea_count": len(s.ideas),
                "techniques": s.techniques_applied,
            }
            for s in self._sessions.values()
        ]

    def get_idea_bank_stats(self) -> Dict:
        if not self._idea_bank:
            return {"total": 0, "avg_score": 0.0, "techniques": {}}
        techniques = {}
        for idea in self._idea_bank:
            techniques[idea.technique_used] = techniques.get(idea.technique_used, 0) + 1
        avg_score = sum(i.combined_score for i in self._idea_bank) / len(self._idea_bank)
        return {
            "total": len(self._idea_bank),
            "avg_score": round(avg_score, 3),
            "techniques": techniques,
            "top_idea": self._idea_bank[0].title if self._idea_bank else None,
        }

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                "active_sessions": len(self._sessions),
                "idea_bank_size": len(self._idea_bank),
            }
