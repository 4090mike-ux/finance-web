"""
JARVIS Hypothesis Engine — Iteration 8
과학적 방법론에 기반한 가설 생성, 검증, 이론 구축

Peirce의 삼단 추론:
  귀추(Abduction):  관찰 → 최선의 설명 가설
  연역(Deduction):  가설 → 예측
  귀납(Induction):  실험 → 가설 갱신

Bayesian Belief Update:
  P(H|E) = P(E|H) × P(H) / P(E)
"""

import json
import time
import logging
import re
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    PROPOSED  = "proposed"
    TESTING   = "testing"
    SUPPORTED = "supported"
    REFUTED   = "refuted"
    UNCERTAIN = "uncertain"


@dataclass
class Observation:
    id: str
    content: str
    source: str = "conversation"
    timestamp: float = field(default_factory=time.time)
    reliability: float = 0.8


@dataclass
class Hypothesis:
    id: str
    claim: str
    rationale: str
    prior_probability: float = 0.5
    posterior_probability: float = 0.5
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    domain: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Theory:
    id: str
    name: str
    description: str
    supporting_hypotheses: List[str]
    confidence: float = 0.6
    domain: str = "general"
    key_principles: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class HypothesisEngine:
    """
    가설 생성·검증 엔진

    1. observe()         — 관찰 누적
    2. generate_hypotheses() — 귀추 추론으로 가설 생성
    3. test_hypothesis() — Bayesian 업데이트로 검증
    4. synthesize_theory()   — 지지된 가설들에서 이론 합성
    """

    def __init__(self, llm_manager=None, knowledge_graph=None,
                 data_dir: str = "data/jarvis"):
        self.llm = llm_manager
        self.kg  = knowledge_graph
        self.observations: List[Observation] = []
        self.hypotheses:   Dict[str, Hypothesis] = {}
        self.theories:     List[Theory] = []
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"HypothesisEngine: {len(self.hypotheses)} hypotheses, "
                    f"{len(self.theories)} theories")

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        f = self.data_dir / "hypothesis_engine.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for o in data.get("observations", []):
                self.observations.append(Observation(**o))
            for h in data.get("hypotheses", {}).values():
                h["status"] = HypothesisStatus(h.get("status", "proposed"))
                self.hypotheses[h["id"]] = Hypothesis(**h)
            for t in data.get("theories", []):
                self.theories.append(Theory(**t))
        except Exception as ex:
            logger.warning(f"HypothesisEngine load: {ex}")

    def _save(self):
        f = self.data_dir / "hypothesis_engine.json"
        f.write_text(json.dumps({
            "observations": [vars(o) for o in self.observations[-300:]],
            "hypotheses":   {k: {**vars(v), "status": v.status.value}
                             for k, v in self.hypotheses.items()},
            "theories":     [vars(t) for t in self.theories],
            "updated_at":   time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── LLM ──────────────────────────────────────────────────────────

    def _get_client(self):
        if self.llm:
            return (getattr(self.llm, "anthropic_client", None)
                    or getattr(self.llm, "_client", None))
        return None

    def _llm(self, prompt: str, max_tokens: int = 1500) -> str:
        client = self._get_client()
        if not client:
            return ""
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"LLM: {e}")
            return ""

    # ── Core Methods ─────────────────────────────────────────────────

    def observe(self, content: str, source: str = "conversation",
                reliability: float = 0.8) -> Observation:
        """관찰 추가"""
        obs = Observation(
            id=hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:8],
            content=content, source=source, reliability=reliability,
        )
        self.observations.append(obs)
        if len(self.observations) > 500:
            self.observations = self.observations[-500:]
        return obs

    def generate_hypotheses(self, topic: str = "",
                            max_hyp: int = 3) -> List[Hypothesis]:
        """귀추 추론: 관찰 → 최선의 가설 생성"""
        obs_text = "\n".join(
            f"- [{o.source}] {o.content[:180]}"
            for o in self.observations[-20:]
        ) or "(관찰 없음)"

        kg_ctx = ""
        if self.kg and topic:
            try:
                res = self.kg.semantic_search(topic, limit=5)
                kg_ctx = "\n".join(
                    f"- {r.get('name','')}: {r.get('description','')[:100]}"
                    for r in res
                )
            except Exception:
                pass

        prompt = f"""귀추 추론(Abductive Reasoning)으로 가설을 생성하세요.

주제: {topic or "최근 관찰 패턴 분석"}

최근 관찰:
{obs_text}

관련 지식:
{kg_ctx or "(없음)"}

JSON 배열로 최대 {max_hyp}개 가설 반환:
[
  {{
    "claim": "가설 주장",
    "rationale": "이 가설이 관찰을 가장 잘 설명하는 이유",
    "prior_probability": 0.5,
    "predictions": ["검증 가능한 예측1", "예측2"],
    "domain": "AI|science|technology|economics|general"
  }}
]
JSON만 반환."""

        text = self._llm(prompt, 2000)
        new_hyps: List[Hypothesis] = []
        try:
            m = re.search(r'\[.*\]', text, re.DOTALL)
            items = json.loads(m.group()) if m else []
            for item in items[:max_hyp]:
                hid = hashlib.md5(item["claim"].encode()).hexdigest()[:8]
                if hid in self.hypotheses:
                    new_hyps.append(self.hypotheses[hid])
                    continue
                h = Hypothesis(
                    id=hid, claim=item["claim"],
                    rationale=item.get("rationale", ""),
                    prior_probability=float(item.get("prior_probability", 0.5)),
                    posterior_probability=float(item.get("prior_probability", 0.5)),
                    predictions=item.get("predictions", []),
                    domain=item.get("domain", "general"),
                )
                self.hypotheses[hid] = h
                new_hyps.append(h)
        except Exception:
            pass
        self._save()
        return new_hyps

    def test_hypothesis(self, hyp_id: str, evidence: str) -> Hypothesis:
        """연역+귀납: Bayesian 업데이트로 가설 검증"""
        hyp = self.hypotheses.get(hyp_id)
        if not hyp:
            raise ValueError(f"Hypothesis {hyp_id} not found")

        hyp.status = HypothesisStatus.TESTING

        prompt = f"""가설을 새 증거로 평가하세요 (Bayesian Reasoning).

가설: {hyp.claim}
근거: {hyp.rationale}
예측: {', '.join(hyp.predictions[:3])}
현재 확률: {hyp.posterior_probability:.2f}

새 증거: {evidence}

JSON으로 응답:
{{
  "supports": true,
  "likelihood_ratio": 2.5,
  "updated_probability": 0.65,
  "explanation": "왜 이 증거가 가설을 지지/반박하는가"
}}"""

        text = self._llm(prompt, 600)
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            result = json.loads(m.group()) if m else {}
        except Exception:
            result = {}

        supports = result.get("supports", True)
        lr = float(result.get("likelihood_ratio", 1.5))
        prior = hyp.posterior_probability
        explanation = result.get("explanation", "")[:200]

        # Bayesian update
        if supports:
            posterior = (prior * lr) / (prior * lr + (1 - prior))
            hyp.evidence_for.append(f"{evidence[:80]}: {explanation}")
        else:
            posterior = prior / (prior + (1 - prior) * lr)
            hyp.evidence_against.append(f"{evidence[:80]}: {explanation}")

        hyp.posterior_probability = min(0.99, max(0.01, posterior))
        hyp.updated_at = time.time()

        if hyp.posterior_probability > 0.75:
            hyp.status = HypothesisStatus.SUPPORTED
        elif hyp.posterior_probability < 0.25:
            hyp.status = HypothesisStatus.REFUTED
        else:
            hyp.status = HypothesisStatus.UNCERTAIN

        self._save()
        return hyp

    def synthesize_theory(self, domain: str = "") -> Optional[Theory]:
        """지지된 가설들을 통합 이론으로 합성"""
        supported = [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.SUPPORTED
            and (not domain or h.domain == domain)
        ]
        if len(supported) < 2:
            return None

        hyp_text = "\n".join(
            f"- [{h.domain}] {h.claim} (확률: {h.posterior_probability:.2f})"
            for h in supported[:10]
        )

        prompt = f"""지지된 가설들에서 통합 이론을 합성하세요.

지지된 가설:
{hyp_text}

JSON으로 응답:
{{
  "name": "이론 이름",
  "description": "이론 설명 (3-5 문장)",
  "confidence": 0.75,
  "key_principles": ["원리1", "원리2", "원리3"]
}}"""

        text = self._llm(prompt, 800)
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            data = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

        if not data.get("name"):
            return None

        theory = Theory(
            id=hashlib.md5(data["name"].encode()).hexdigest()[:8],
            name=data["name"],
            description=data["description"],
            supporting_hypotheses=[h.id for h in supported[:10]],
            confidence=float(data.get("confidence", 0.6)),
            domain=domain or "general",
            key_principles=data.get("key_principles", []),
        )
        self.theories.append(theory)
        self._save()
        return theory

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        counts: Dict[str, int] = {}
        for h in self.hypotheses.values():
            counts[h.status.value] = counts.get(h.status.value, 0) + 1
        return {
            "available": True,
            "total_observations": len(self.observations),
            "total_hypotheses":   len(self.hypotheses),
            "total_theories":     len(self.theories),
            "hypothesis_status":  counts,
            "supported_count":    counts.get("supported", 0),
        }
