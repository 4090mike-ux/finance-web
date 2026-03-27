"""
JARVIS 지식 합성기 — Iteration 12
다양한 소스의 지식을 융합하여 새로운 인사이트와 이론을 생성한다

영감:
  - 개념 혼합 이론 (Conceptual Blending Theory, Fauconnier & Turner)
  - 지식 그래프 임베딩 (KGE)
  - 유추 기반 학습 (Analogical Learning)
  - 창발적 이해 (Emergent Understanding)
  - 메타 분석 (Meta-Analysis in Research)

핵심 개념:
  단순 검색 = 기존 지식 반환
  지식 합성 = 서로 다른 도메인의 지식을 조합하여 새로운 패턴을 발견
  A + B → C  (C는 A나 B에 없는 새로운 통찰)
  예: "생물학적 진화" + "소프트웨어 아키텍처" → 자기 적응 시스템 설계 원칙
"""

import json
import time
import uuid
import logging
import threading
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SynthesisMode(Enum):
    CROSS_DOMAIN = "cross_domain"       # 이종 도메인 연결
    CONTRADICTION = "contradiction"      # 모순 해결
    ANALOGICAL = "analogical"            # 유추 통합
    EMERGENT = "emergent"               # 창발적 패턴
    META_ANALYSIS = "meta_analysis"     # 메타 분석


@dataclass
class KnowledgeAtom:
    """단일 지식 단위"""
    atom_id: str
    content: str                        # 핵심 내용
    source: str                         # 출처 (web/memory/research/user)
    domain: str                         # 도메인 (AI, biology, physics 등)
    confidence: float = 0.8
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "atom_id": self.atom_id,
            "content": self.content[:200],
            "source": self.source,
            "domain": self.domain,
            "confidence": self.confidence,
            "importance": self.importance,
            "tags": self.tags,
        }


@dataclass
class SynthesisResult:
    """합성 결과"""
    synthesis_id: str
    query: str
    mode: SynthesisMode
    input_atoms: List[str]              # 입력 atom_id 목록
    synthesized_insight: str            # 핵심 합성 결과
    novel_concepts: List[str]           # 새롭게 발견된 개념
    cross_connections: List[Dict]       # 도메인 간 연결
    confidence: float = 0.75
    novelty_score: float = 0.0          # 새로움 점수 (0~1)
    applications: List[str] = field(default_factory=list)  # 응용 가능 분야
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "synthesis_id": self.synthesis_id,
            "query": self.query,
            "mode": self.mode.value,
            "synthesized_insight": self.synthesized_insight,
            "novel_concepts": self.novel_concepts,
            "cross_connections": self.cross_connections,
            "confidence": self.confidence,
            "novelty_score": self.novelty_score,
            "applications": self.applications,
        }


@dataclass
class KnowledgeTheory:
    """합성으로 생성된 이론/가설"""
    theory_id: str
    title: str
    premise: str                        # 전제
    conclusion: str                     # 결론
    supporting_atoms: List[str]         # 근거 atom_id
    domain_bridges: List[str]           # 연결된 도메인들
    strength: float = 0.6               # 이론 강도
    falsifiable: bool = True            # 반증 가능성
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "theory_id": self.theory_id,
            "title": self.title,
            "premise": self.premise,
            "conclusion": self.conclusion,
            "domain_bridges": self.domain_bridges,
            "strength": self.strength,
            "falsifiable": self.falsifiable,
        }


# ════════════════════════════════════════════════════════════════
# 지식 합성기
# ════════════════════════════════════════════════════════════════

class KnowledgeSynthesizer:
    """
    JARVIS 지식 합성기
    여러 도메인의 지식을 연결하여 새로운 통찰을 창출한다
    """

    # 주요 도메인 목록
    DOMAINS = [
        "인공지능", "머신러닝", "신경과학", "물리학", "생물학", "화학",
        "수학", "철학", "심리학", "경제학", "사회학", "언어학",
        "공학", "의학", "역사", "예술", "문학", "음악",
    ]

    def __init__(self, llm_manager, knowledge_graph=None,
                 memory_manager=None):
        self.llm = llm_manager
        self.kg = knowledge_graph      # 기존 지식 그래프 연계
        self.memory = memory_manager
        self._atom_store: Dict[str, KnowledgeAtom] = {}
        self._theories: List[KnowledgeTheory] = []
        self._synthesis_history: List[SynthesisResult] = []
        self._lock = threading.Lock()
        self._stats = {
            "atoms_added": 0,
            "syntheses_performed": 0,
            "theories_generated": 0,
            "novel_concepts_found": 0,
        }
        logger.info("KnowledgeSynthesizer initialized — conceptual blending ready")

    # ── 지식 원자 관리 ────────────────────────────────────────────

    def add_knowledge(self, content: str, source: str = "user",
                      domain: str = "일반", tags: List[str] = None,
                      importance: float = 0.5) -> KnowledgeAtom:
        """지식 원자 추가"""
        atom = KnowledgeAtom(
            atom_id=str(uuid.uuid4())[:10],
            content=content,
            source=source,
            domain=domain,
            tags=tags or [],
            importance=importance,
        )
        with self._lock:
            self._atom_store[atom.atom_id] = atom
            self._stats["atoms_added"] += 1
        return atom

    def add_knowledge_batch(self, items: List[Dict]) -> List[KnowledgeAtom]:
        """여러 지식 원자 일괄 추가"""
        atoms = []
        for item in items:
            atom = self.add_knowledge(
                content=item.get("content", ""),
                source=item.get("source", "batch"),
                domain=item.get("domain", "일반"),
                tags=item.get("tags", []),
                importance=item.get("importance", 0.5),
            )
            atoms.append(atom)
        return atoms

    def ingest_from_text(self, text: str, source: str = "document") -> List[KnowledgeAtom]:
        """긴 텍스트에서 지식 원자 자동 추출"""
        extract_prompt = f"""다음 텍스트에서 핵심 지식 단위를 최대 8개 추출하세요.

텍스트:
{text[:3000]}

각 지식 단위는 독립적으로 의미있어야 합니다.
JSON 배열로만 응답:
[
  {{"content": "핵심 내용 (1-2문장)", "domain": "도메인명", "tags": ["태그1", "태그2"], "importance": 0.8}},
  ...
]"""

        try:
            response = self.llm.generate(extract_prompt, max_tokens=1000)
            if isinstance(response, dict):
                response = response.get("content", "[]")

            items = self._parse_json_list(response)
            atoms = []
            for item in items[:8]:
                if isinstance(item, dict) and item.get("content"):
                    atom = self.add_knowledge(
                        content=item["content"],
                        source=source,
                        domain=item.get("domain", "일반"),
                        tags=item.get("tags", []),
                        importance=float(item.get("importance", 0.5)),
                    )
                    atoms.append(atom)
            return atoms
        except Exception as e:
            logger.warning(f"Knowledge ingestion failed: {e}")
            return []

    # ── 합성 핵심 API ─────────────────────────────────────────────

    def synthesize(self, query: str,
                   mode: SynthesisMode = SynthesisMode.CROSS_DOMAIN,
                   domains: List[str] = None,
                   n_atoms: int = 10) -> SynthesisResult:
        """
        쿼리에 관련된 지식을 합성하여 새로운 통찰 생성
        """
        # 관련 지식 원자 선택
        relevant_atoms = self._select_relevant_atoms(query, n_atoms)

        # 추가 도메인 지식 획득 (원자가 부족하면)
        if len(relevant_atoms) < 3:
            self._gather_domain_knowledge(query, domains or [])
            relevant_atoms = self._select_relevant_atoms(query, n_atoms)

        # 도메인별 그룹화
        domain_groups = self._group_by_domain(relevant_atoms)

        # 모드에 따른 합성 전략
        handler = {
            SynthesisMode.CROSS_DOMAIN: self._synthesize_cross_domain,
            SynthesisMode.CONTRADICTION: self._synthesize_contradiction,
            SynthesisMode.ANALOGICAL: self._synthesize_analogical,
            SynthesisMode.EMERGENT: self._synthesize_emergent,
            SynthesisMode.META_ANALYSIS: self._synthesize_meta_analysis,
        }.get(mode, self._synthesize_cross_domain)

        result = handler(query, relevant_atoms, domain_groups)

        with self._lock:
            self._synthesis_history.append(result)
            self._stats["syntheses_performed"] += 1
            self._stats["novel_concepts_found"] += len(result.novel_concepts)

        logger.info(f"Synthesis '{query[:40]}' → {len(result.novel_concepts)} novel concepts, "
                    f"novelty={result.novelty_score:.2f}")
        return result

    def synthesize_cross_domain_theory(self, domain1: str, domain2: str,
                                        topic: str) -> KnowledgeTheory:
        """두 도메인을 연결하는 새 이론 생성"""
        prompt = f"""두 도메인의 원리를 연결하여 새로운 이론을 만드세요.

도메인 1: {domain1}
도메인 2: {domain2}
주제: {topic}

단계:
1. {domain1}의 핵심 원리/패턴을 찾으세요
2. {domain2}의 핵심 원리/패턴을 찾으세요
3. 두 원리 사이의 구조적 유사성(아날로지)을 발견하세요
4. 이를 {topic}에 적용한 새 이론을 공식화하세요
5. 이 이론을 반증할 수 있는 방법을 제시하세요

JSON으로만 응답:
{{
  "title": "이론 제목",
  "premise": "전제 (2-3문장)",
  "conclusion": "결론 및 주장 (2-3문장)",
  "domain_bridges": ["{domain1}", "{domain2}"],
  "strength": 0.7,
  "falsifiable": true
}}"""

        try:
            response = self.llm.generate(prompt, max_tokens=600)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            data = self._parse_json(response)
        except Exception as e:
            data = {
                "title": f"{domain1} × {domain2} 이론",
                "premise": f"{domain1}과 {domain2}는 구조적 유사성을 공유합니다",
                "conclusion": f"이를 {topic}에 적용할 수 있습니다",
                "domain_bridges": [domain1, domain2],
                "strength": 0.5,
                "falsifiable": True,
            }

        theory = KnowledgeTheory(
            theory_id=str(uuid.uuid4())[:8],
            title=data.get("title", f"{domain1}×{domain2} 이론"),
            premise=data.get("premise", ""),
            conclusion=data.get("conclusion", ""),
            supporting_atoms=[],
            domain_bridges=data.get("domain_bridges", [domain1, domain2]),
            strength=float(data.get("strength", 0.6)),
            falsifiable=bool(data.get("falsifiable", True)),
        )

        with self._lock:
            self._theories.append(theory)
            self._stats["theories_generated"] += 1

        return theory

    def find_hidden_connections(self, topic1: str, topic2: str) -> Dict:
        """두 주제 사이의 숨겨진 연결 탐색"""
        prompt = f"""두 주제 사이의 숨겨진/비직관적 연결을 발견하세요.

주제 1: {topic1}
주제 2: {topic2}

다음을 포함하여 분석하세요:
1. 표면적 공통점 (명확한 연관)
2. 구조적 유사성 (작동 방식의 유사)
3. 역설적 연결 (겉보기에 반대지만 실제로 연결)
4. 창발적 연결 (둘을 합칠 때 생기는 새 가능성)
5. 응용 가능성 (어디에 이 연결을 활용할 수 있나)

한국어로 구체적으로 설명하세요."""

        try:
            response = self.llm.generate(prompt, max_tokens=600)
            if isinstance(response, dict):
                response = response.get("content", "")
            return {
                "topic1": topic1,
                "topic2": topic2,
                "connections": response,
            }
        except Exception as e:
            return {"topic1": topic1, "topic2": topic2, "error": str(e)}

    def generate_research_questions(self, topic: str, n: int = 5) -> List[str]:
        """주제에서 미답 연구 질문 생성"""
        atoms = self._select_relevant_atoms(topic, 8)
        atoms_text = "\n".join([f"- {a.content}" for a in atoms[:6]])

        prompt = f"""'{topic}' 주제에서 아직 충분히 탐구되지 않은 연구 질문 {n}개를 생성하세요.

현재 알려진 지식:
{atoms_text if atoms_text else "(직접 생성)"}

좋은 연구 질문의 조건:
- 현재 답이 없거나 불완전한 질문
- 여러 도메인에 걸친 융합 질문
- 실용적 가치가 있는 질문
- 검증/반증 가능한 질문

번호 없이 각 질문만 출력 (한 줄에 하나):"""

        try:
            response = self.llm.generate(prompt, max_tokens=400)
            if isinstance(response, dict):
                response = response.get("content", "")
            questions = [q.strip() for q in response.split("\n") if q.strip() and len(q.strip()) > 10]
            return questions[:n]
        except Exception as e:
            return [f"'{topic}'의 미래 방향은?", f"'{topic}'의 한계는 무엇인가?"]

    # ── 합성 전략 ─────────────────────────────────────────────────

    def _synthesize_cross_domain(self, query: str, atoms: List[KnowledgeAtom],
                                  domain_groups: Dict) -> SynthesisResult:
        """이종 도메인 지식 연결 합성"""
        domains_found = list(domain_groups.keys())[:4]
        atoms_text = self._atoms_to_text(atoms[:8])

        prompt = f"""다음 다양한 도메인의 지식을 융합하여 '{query}'에 대한 새로운 통찰을 생성하세요.

발견된 도메인: {', '.join(domains_found)}

지식 조각들:
{atoms_text}

수행할 작업:
1. 서로 다른 도메인에서 공통 패턴 찾기
2. 이 패턴을 질문에 적용한 새 통찰 생성
3. 어느 도메인에도 단독으로 없는 새 개념 발견

JSON으로만 응답:
{{
  "synthesized_insight": "핵심 합성 통찰 (3-4문장)",
  "novel_concepts": ["새 개념1", "새 개념2", "새 개념3"],
  "cross_connections": [
    {{"from": "도메인A", "to": "도메인B", "connection": "연결 설명"}},
    {{"from": "도메인B", "to": "주제", "connection": "적용 방법"}}
  ],
  "applications": ["응용분야1", "응용분야2"],
  "novelty_score": 0.75
}}"""

        return self._build_synthesis_result(query, prompt, atoms, SynthesisMode.CROSS_DOMAIN)

    def _synthesize_contradiction(self, query: str, atoms: List[KnowledgeAtom],
                                   domain_groups: Dict) -> SynthesisResult:
        """모순되는 지식들을 통합하여 상위 원리 도출"""
        atoms_text = self._atoms_to_text(atoms[:8])

        prompt = f"""다음 지식들 중 서로 모순/충돌하는 것들을 찾아 상위 원리로 통합하세요.

주제: {query}
지식들:
{atoms_text}

모순 해결 방법:
- 두 관점이 모두 옳은 더 높은 수준의 원리 찾기
- 조건에 따라 달라지는 경우 조건 명시
- 관점의 차이가 실제로 다른 것을 가리키는지 확인

JSON으로만 응답:
{{
  "synthesized_insight": "모순을 해결하는 통합 원리",
  "novel_concepts": ["상위 개념", "조화 원리"],
  "cross_connections": [{{"from": "관점A", "to": "관점B", "connection": "통합 방법"}}],
  "applications": ["어디에 적용 가능한가"],
  "novelty_score": 0.8
}}"""

        return self._build_synthesis_result(query, prompt, atoms, SynthesisMode.CONTRADICTION)

    def _synthesize_analogical(self, query: str, atoms: List[KnowledgeAtom],
                                domain_groups: Dict) -> SynthesisResult:
        """아날로지 기반 합성 — 잘 알려진 도메인의 구조를 새 도메인에 이식"""
        source_domains = list(domain_groups.keys())[:2]
        atoms_text = self._atoms_to_text(atoms[:6])

        prompt = f"""유추(아날로지) 기반으로 '{query}'에 대한 새로운 이해를 생성하세요.

출발 도메인: {source_domains}
지식:
{atoms_text}

아날로지 과정:
1. 출발 도메인의 구조/원리 파악
2. 목표 도메인({query})의 구조 파악
3. 두 구조 사이의 대응 관계 매핑
4. 아직 발견되지 않은 대응 관계에서 새 예측 도출

JSON으로만 응답:
{{
  "synthesized_insight": "아날로지에서 도출된 새 통찰",
  "novel_concepts": ["아날로지적 개념1", "예측된 현상"],
  "cross_connections": [{{"from": "도메인", "to": "주제", "connection": "구조적 대응"}}],
  "applications": ["이 아날로지가 유용한 곳"],
  "novelty_score": 0.7
}}"""

        return self._build_synthesis_result(query, prompt, atoms, SynthesisMode.ANALOGICAL)

    def _synthesize_emergent(self, query: str, atoms: List[KnowledgeAtom],
                              domain_groups: Dict) -> SynthesisResult:
        """창발적 합성 — 부분의 합이 전체를 초월하는 패턴 발견"""
        atoms_text = self._atoms_to_text(atoms[:8])

        prompt = f"""다음 지식 조각들을 종합할 때 창발하는 새 패턴을 발견하세요.

주제: {query}
지식 조각들:
{atoms_text}

창발 분석:
- 각 조각 단독으로는 보이지 않지만 합쳐지면 보이는 것
- 비선형적 상호작용에서 나오는 새 속성
- 시스템 수준의 패턴 (미시 → 거시 전환)

JSON으로만 응답:
{{
  "synthesized_insight": "창발적 통찰 (전체가 부분의 합보다 큰 이유)",
  "novel_concepts": ["창발 속성1", "시스템 패턴", "새 이해"],
  "cross_connections": [{{"from": "요소", "to": "전체", "connection": "창발 관계"}}],
  "applications": ["창발 원리 응용"],
  "novelty_score": 0.85
}}"""

        return self._build_synthesis_result(query, prompt, atoms, SynthesisMode.EMERGENT)

    def _synthesize_meta_analysis(self, query: str, atoms: List[KnowledgeAtom],
                                   domain_groups: Dict) -> SynthesisResult:
        """메타 분석 — 여러 연구/관점의 패턴을 상위 수준에서 분석"""
        atoms_text = self._atoms_to_text(atoms[:10])

        prompt = f"""메타 분석: '{query}'에 관한 다양한 관점/연구의 상위 패턴을 찾으세요.

수집된 관점들:
{atoms_text}

메타 분석 내용:
1. 대부분의 연구/관점이 동의하는 것 (강한 증거)
2. 논쟁 중인 부분 (불확실성이 높음)
3. 아직 연구되지 않은 공백 (연구 기회)
4. 전체 패턴이 시사하는 바 (메타 인사이트)

JSON으로만 응답:
{{
  "synthesized_insight": "메타 수준 통찰 및 결론",
  "novel_concepts": ["메타 패턴", "연구 공백", "미래 방향"],
  "cross_connections": [{{"from": "연구그룹", "to": "메타결론", "connection": "패턴"}}],
  "applications": ["메타 분석 시사점"],
  "novelty_score": 0.7
}}"""

        return self._build_synthesis_result(query, prompt, atoms, SynthesisMode.META_ANALYSIS)

    def _build_synthesis_result(self, query: str, prompt: str,
                                 atoms: List[KnowledgeAtom],
                                 mode: SynthesisMode) -> SynthesisResult:
        """LLM 호출 후 SynthesisResult 생성"""
        try:
            response = self.llm.generate(prompt, max_tokens=800)
            if isinstance(response, dict):
                response = response.get("content", "{}")
            data = self._parse_json(response)
        except Exception as e:
            logger.warning(f"Synthesis LLM call failed: {e}")
            data = {
                "synthesized_insight": f"'{query}'에 대한 합성 분석 완료",
                "novel_concepts": [],
                "cross_connections": [],
                "applications": [],
                "novelty_score": 0.5,
            }

        return SynthesisResult(
            synthesis_id=str(uuid.uuid4())[:8],
            query=query,
            mode=mode,
            input_atoms=[a.atom_id for a in atoms],
            synthesized_insight=data.get("synthesized_insight", ""),
            novel_concepts=data.get("novel_concepts", []),
            cross_connections=data.get("cross_connections", []),
            confidence=0.75,
            novelty_score=float(data.get("novelty_score", 0.5)),
            applications=data.get("applications", []),
        )

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _select_relevant_atoms(self, query: str, n: int) -> List[KnowledgeAtom]:
        """쿼리와 관련성 높은 원자 선택 (키워드 매칭 + 중요도)"""
        query_words = set(query.lower().split())
        scored = []
        for atom in self._atom_store.values():
            atom_words = set(atom.content.lower().split())
            overlap = len(query_words & atom_words) / max(len(query_words), 1)
            score = overlap * 0.6 + atom.importance * 0.4
            scored.append((score, atom))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored[:n]]

    def _gather_domain_knowledge(self, query: str, domains: List[str]):
        """LLM으로 도메인 지식 즉석 생성 (외부 데이터 없을 때)"""
        domains_to_use = domains or self.DOMAINS[:4]
        prompt = f"""'{query}'와 관련하여 다음 도메인에서의 핵심 지식을 각 1-2문장으로 알려주세요:

{', '.join(domains_to_use[:4])}

JSON 배열로만 응답:
[{{"content": "...", "domain": "도메인명", "importance": 0.7}}, ...]"""

        try:
            response = self.llm.generate(prompt, max_tokens=600)
            if isinstance(response, dict):
                response = response.get("content", "[]")
            items = self._parse_json_list(response)
            for item in items[:8]:
                if isinstance(item, dict) and item.get("content"):
                    self.add_knowledge(
                        content=item["content"],
                        source="llm_generated",
                        domain=item.get("domain", "일반"),
                        importance=float(item.get("importance", 0.5)),
                    )
        except Exception as e:
            logger.warning(f"Domain knowledge gathering failed: {e}")

    def _group_by_domain(self, atoms: List[KnowledgeAtom]) -> Dict[str, List[KnowledgeAtom]]:
        groups: Dict[str, List] = {}
        for atom in atoms:
            groups.setdefault(atom.domain, []).append(atom)
        return groups

    def _atoms_to_text(self, atoms: List[KnowledgeAtom]) -> str:
        return "\n".join([
            f"[{a.domain}] {a.content}"
            for a in atoms
        ])

    def _parse_json(self, text: str) -> Dict:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _parse_json_list(self, text: str) -> List:
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass
        return []

    # ── 조회 API ─────────────────────────────────────────────────

    def get_theories(self, n: int = 20) -> List[Dict]:
        return [t.to_dict() for t in self._theories[-n:]]

    def get_synthesis_history(self, n: int = 10) -> List[Dict]:
        return [s.to_dict() for s in self._synthesis_history[-n:]]

    def get_domain_map(self) -> Dict[str, int]:
        """도메인별 원자 수"""
        domain_map: Dict[str, int] = {}
        for atom in self._atom_store.values():
            domain_map[atom.domain] = domain_map.get(atom.domain, 0) + 1
        return dict(sorted(domain_map.items(), key=lambda x: x[1], reverse=True))

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                "total_atoms": len(self._atom_store),
                "total_theories": len(self._theories),
                "total_syntheses": len(self._synthesis_history),
                "domains_covered": len(self.get_domain_map()),
            }
