"""
JARVIS 의식 루프 (Consciousness Loop) — Iteration 5
JARVIS가 자신의 추론을 관찰하고 평가하는 메타인지 시스템

인간 대비 우위:
- 모든 응답의 품질을 실시간으로 자기 평가
- 할루시네이션(환각) 자동 감지 패턴
- 모순 감지 (이전 답변과 충돌 여부)
- 신뢰도 낮으면 자동으로 재추론 트리거
- 자기 인식 리포트 — "내가 무엇을 모르는지 안다"
- 인지 부하 모니터링 — 복잡도에 따른 전략 전환
"""

import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

CONSCIOUSNESS_DB = Path("data/jarvis/consciousness_state.json")


@dataclass
class ResponseEvaluation:
    """응답 품질 평가 결과"""
    response_id: str
    query: str
    response: str
    quality_score: float       # 0-1
    confidence: float          # 0-1
    hallucination_risk: float  # 0-1 (높을수록 위험)
    contradictions: List[str]  # 이전 응답과 모순되는 내용
    uncertainty_flags: List[str]  # 불확실한 표현들
    needs_rethink: bool        # 재추론 필요 여부
    rethink_reason: str        # 재추론 이유
    evaluation_time: float     # 평가 소요 시간
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConsciousnessState:
    """JARVIS 의식 상태"""
    cognitive_load: float = 0.3           # 인지 부하 (0=여유, 1=과부하)
    overall_confidence: float = 0.8      # 전체 신뢰도
    total_evaluations: int = 0
    rethinks_triggered: int = 0
    hallucinations_caught: int = 0
    contradictions_found: int = 0
    current_focus: str = ""              # 현재 처리 중인 주제
    active_since: str = field(default_factory=lambda: datetime.now().isoformat())


# 할루시네이션 위험 신호 패턴
HALLUCINATION_PATTERNS = [
    r'\b(항상|절대|모든|전혀|반드시)\b',        # 극단적 확실성
    r'\b(2026년|2027년|2028년)\b.{0,50}(발표|출시|개발)',  # 미래 확실 예측
    r'\b(사실입니다|분명합니다|확실합니다)\b',   # 과도한 확신
    r'\d{1,3}\.\d+%',                           # 과도하게 정밀한 수치
]

# 불확실성 표현 패턴
UNCERTAINTY_PATTERNS = [
    r'\b(아마|아마도|추측|모를|불확실)\b',
    r'\b(확인이 필요|검증 필요|더 알아봐야)\b',
    r'\b(~일 수도|~가능성|~일지도)\b',
]


class ConsciousnessLoop:
    """
    JARVIS 메타인지 시스템
    자신의 추론을 지켜보고 품질을 보장하는 의식 루프
    """

    # 재추론 임계값 (이 점수 미만이면 재추론)
    RETHINK_THRESHOLD = 0.5
    # 할루시네이션 임계값
    HALLUCINATION_THRESHOLD = 0.7
    # 최근 평가 유지 개수
    EVAL_HISTORY_SIZE = 100

    def __init__(
        self,
        llm_manager=None,
        memory_manager=None,
        event_callback: Optional[Callable] = None,
    ):
        self.llm = llm_manager
        self.memory = memory_manager
        self.event_callback = event_callback

        self.state = ConsciousnessState()
        self._eval_history: deque = deque(maxlen=self.EVAL_HISTORY_SIZE)
        self._recent_responses: deque = deque(maxlen=20)  # 모순 감지용
        self._is_active = False
        self._lock = threading.Lock()
        self._load_state()
        logger.info("ConsciousnessLoop initialized — meta-cognition active")

    # ── 메인 평가 함수 ─────────────────────────────────────────────────────

    def evaluate_response(
        self,
        query: str,
        response: str,
        auto_rethink: bool = True,
    ) -> ResponseEvaluation:
        """응답 품질 평가 및 필요시 재추론 지시"""
        start_time = time.time()
        response_id = f"eval_{int(time.time())}"

        # 1. 할루시네이션 위험 감지
        hallucination_risk = self._detect_hallucination_risk(response)

        # 2. 불확실성 플래그 감지
        uncertainty_flags = self._detect_uncertainty(response)

        # 3. 모순 감지
        contradictions = self._detect_contradictions(response)

        # 4. 품질 점수 계산
        quality_score = self._compute_quality_score(
            response, hallucination_risk, uncertainty_flags, contradictions
        )

        # 5. 신뢰도 추정
        confidence = self._estimate_confidence(response, uncertainty_flags, quality_score)

        # 6. 재추론 필요 여부
        needs_rethink = False
        rethink_reason = ""
        if quality_score < self.RETHINK_THRESHOLD:
            needs_rethink = True
            rethink_reason = f"품질 점수 낮음: {quality_score:.2f}"
        elif hallucination_risk > self.HALLUCINATION_THRESHOLD:
            needs_rethink = True
            rethink_reason = f"할루시네이션 위험: {hallucination_risk:.2f}"
        elif len(contradictions) > 0:
            needs_rethink = True
            rethink_reason = f"모순 감지: {len(contradictions)}개"

        eval_result = ResponseEvaluation(
            response_id=response_id,
            query=query,
            response=response[:500],
            quality_score=quality_score,
            confidence=confidence,
            hallucination_risk=hallucination_risk,
            contradictions=contradictions,
            uncertainty_flags=uncertainty_flags,
            needs_rethink=needs_rethink,
            rethink_reason=rethink_reason,
            evaluation_time=round(time.time() - start_time, 3),
        )

        # 상태 업데이트
        with self._lock:
            self._eval_history.append(eval_result)
            self._recent_responses.append({
                "query": query[:100],
                "response": response[:300],
                "timestamp": eval_result.timestamp,
            })
            self.state.total_evaluations += 1
            if needs_rethink:
                self.state.rethinks_triggered += 1
            if hallucination_risk > self.HALLUCINATION_THRESHOLD:
                self.state.hallucinations_caught += 1
            if contradictions:
                self.state.contradictions_found += len(contradictions)

            # 전체 신뢰도 업데이트 (이동 평균)
            alpha = 0.1
            self.state.overall_confidence = (
                (1 - alpha) * self.state.overall_confidence + alpha * confidence
            )
            # 인지 부하 업데이트
            self.state.cognitive_load = min(1.0, len(self._eval_history) / self.EVAL_HISTORY_SIZE)
            self.state.current_focus = query[:50]

        if needs_rethink:
            self._emit("consciousness_alert", {
                "reason": rethink_reason,
                "quality_score": quality_score,
                "hallucination_risk": hallucination_risk,
            })
            logger.warning(f"[Consciousness] Rethink needed: {rethink_reason}")

        if quality_score > 0.85:
            logger.debug(f"[Consciousness] High quality response: {quality_score:.2f}")

        self._save_state()
        return eval_result

    def deep_evaluate(self, query: str, response: str) -> Dict:
        """LLM을 사용한 심층 품질 평가"""
        if not self.llm:
            return {"error": "LLM not available"}

        from jarvis.llm.manager import Message
        prompt = f"""다음 AI 응답의 품질을 평가하세요:

질문: {query}

응답: {response[:2000]}

다음 기준으로 평가:
1. 정확성 (사실 여부, 검증 가능성)
2. 완전성 (질문에 완전히 답했는지)
3. 논리적 일관성
4. 할루시네이션 위험 (불확실한 사실을 확실한 것처럼 표현했는지)
5. 유용성

JSON으로 반환:
{{
  "accuracy": 0.85,
  "completeness": 0.90,
  "coherence": 0.88,
  "hallucination_risk": 0.15,
  "usefulness": 0.92,
  "overall": 0.87,
  "strengths": ["장점 1", "장점 2"],
  "weaknesses": ["약점 1"],
  "suggested_improvement": "개선 방향"
}}

JSON만 출력하세요."""

        try:
            resp = self.llm.chat([Message(role="user", content=prompt)], max_tokens=1024)
            match = re.search(r'\{.*\}', resp.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"[Consciousness] Deep evaluate error: {e}")
        return {"error": "평가 실패"}

    # ── 메타인지 분석 ──────────────────────────────────────────────────────

    def self_reflect(self) -> Dict:
        """자기 성찰 — 현재 상태와 패턴 분석"""
        recent = list(self._eval_history)[-20:]
        if not recent:
            return {"status": "평가 기록 없음"}

        avg_quality = sum(e.quality_score for e in recent) / len(recent)
        avg_confidence = sum(e.confidence for e in recent) / len(recent)
        rethink_rate = sum(1 for e in recent if e.needs_rethink) / len(recent)
        avg_hallucination = sum(e.hallucination_risk for e in recent) / len(recent)

        # 취약 영역 식별
        weak_queries = [e.query[:50] for e in recent if e.quality_score < 0.6]

        reflection = {
            "period": f"최근 {len(recent)}회 평가",
            "avg_quality": round(avg_quality, 3),
            "avg_confidence": round(avg_confidence, 3),
            "rethink_rate": round(rethink_rate, 3),
            "avg_hallucination_risk": round(avg_hallucination, 3),
            "state": {
                "cognitive_load": self.state.cognitive_load,
                "overall_confidence": self.state.overall_confidence,
                "total_evaluations": self.state.total_evaluations,
                "hallucinations_caught": self.state.hallucinations_caught,
                "contradictions_found": self.state.contradictions_found,
            },
            "weak_areas": weak_queries[:3],
            "assessment": self._generate_assessment(avg_quality, rethink_rate, avg_hallucination),
        }
        return reflection

    def get_cognitive_status(self) -> Dict:
        """현재 인지 상태 리포트"""
        recent_quality = [e.quality_score for e in list(self._eval_history)[-10:]]
        trend = "stable"
        if len(recent_quality) >= 5:
            first_half = sum(recent_quality[:len(recent_quality)//2]) / (len(recent_quality)//2)
            second_half = sum(recent_quality[len(recent_quality)//2:]) / (len(recent_quality) - len(recent_quality)//2)
            if second_half > first_half + 0.05:
                trend = "improving"
            elif second_half < first_half - 0.05:
                trend = "declining"

        return {
            "cognitive_load": self.state.cognitive_load,
            "overall_confidence": self.state.overall_confidence,
            "quality_trend": trend,
            "is_active": self._is_active,
            "current_focus": self.state.current_focus,
            "stats": {
                "total_evaluations": self.state.total_evaluations,
                "rethinks_triggered": self.state.rethinks_triggered,
                "hallucinations_caught": self.state.hallucinations_caught,
                "contradictions_found": self.state.contradictions_found,
            },
            "active_since": self.state.active_since,
        }

    # ── 내부 감지 함수 ─────────────────────────────────────────────────────

    def _detect_hallucination_risk(self, text: str) -> float:
        """할루시네이션 위험 점수 계산"""
        risk_score = 0.0
        text_lower = text.lower()

        # 패턴 기반 감지
        for pattern in HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            risk_score += len(matches) * 0.1

        # 과도하게 구체적인 수치
        specific_numbers = re.findall(r'\b\d{4,}\b', text)
        risk_score += len(specific_numbers) * 0.05

        # 검증 불가 주장 패턴
        unverifiable = ['연구에 따르면', '전문가들은', '통계에 의하면']
        for phrase in unverifiable:
            if phrase in text_lower:
                risk_score += 0.15

        return min(1.0, risk_score)

    def _detect_uncertainty(self, text: str) -> List[str]:
        """불확실성 표현 감지"""
        flags = []
        for pattern in UNCERTAINTY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            flags.extend(matches)
        return list(set(flags))[:5]

    def _detect_contradictions(self, response: str) -> List[str]:
        """이전 응답과 모순 감지 (간단한 키워드 기반)"""
        contradictions = []
        recent = list(self._recent_responses)[-5:]

        response_lower = response.lower()
        for prev in recent:
            prev_text = prev.get("response", "").lower()
            # 간단한 모순 패턴 감지
            # "A이다" 후 "A가 아니다"
            if len(prev_text) > 50 and len(response_lower) > 50:
                # 같은 주제에서 반대 표현 감지
                if "아니다" in response_lower and any(
                    word in response_lower and word in prev_text
                    for word in ["사실", "맞다", "옳다"]
                ):
                    contradictions.append("이전 응답과 모순 가능성")
                    break

        return contradictions

    def _compute_quality_score(
        self,
        response: str,
        hallucination_risk: float,
        uncertainty_flags: List[str],
        contradictions: List[str],
    ) -> float:
        """응답 품질 종합 점수 (0-1)"""
        score = 1.0

        # 할루시네이션 위험 차감
        score -= hallucination_risk * 0.3

        # 모순 차감
        score -= len(contradictions) * 0.2

        # 길이 기반 완성도
        if len(response) < 50:
            score -= 0.3  # 너무 짧음
        elif len(response) > 5000:
            score -= 0.1  # 너무 긴 것도 약간 차감

        # 불확실성은 적당히 있는 게 좋음 (솔직함)
        if not uncertainty_flags and hallucination_risk > 0.5:
            score -= 0.1  # 불확실한데 확신하는 경우

        # 구조화 보너스
        if re.search(r'\n\d+\.|##|\*\*', response):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _estimate_confidence(
        self,
        response: str,
        uncertainty_flags: List[str],
        quality_score: float,
    ) -> float:
        """응답 신뢰도 추정"""
        confidence = quality_score * 0.7

        # 불확실성 표현 수 반영
        if uncertainty_flags:
            # 불확실성 표현이 있으면 신뢰도 낮추되, 솔직함 보너스
            confidence += 0.05  # 솔직하게 불확실성 표현한 것은 긍정적
            confidence -= len(uncertainty_flags) * 0.05

        # 참조 문헌 언급
        if re.search(r'(출처|참고|reference|source)', response, re.IGNORECASE):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _generate_assessment(
        self, avg_quality: float, rethink_rate: float, avg_hallucination: float
    ) -> str:
        if avg_quality > 0.8 and rethink_rate < 0.1:
            return "우수 — 높은 응답 품질과 낮은 재추론 비율"
        elif avg_quality > 0.6:
            return "양호 — 전반적으로 좋은 성능, 일부 개선 가능"
        elif rethink_rate > 0.3:
            return "주의 — 높은 재추론 비율, 지식 보강 필요"
        elif avg_hallucination > 0.5:
            return "경고 — 높은 할루시네이션 위험, 신중한 검토 필요"
        else:
            return "보통 — 안정적이나 추가 최적화 권장"

    # ── 활성화/비활성화 ────────────────────────────────────────────────────

    def activate(self):
        self._is_active = True
        logger.info("[Consciousness] Meta-cognition activated")

    def deactivate(self):
        self._is_active = False
        logger.info("[Consciousness] Meta-cognition deactivated")

    def is_active(self) -> bool:
        return self._is_active

    # ── 이벤트 ─────────────────────────────────────────────────────────────

    def _emit(self, event_type: str, data: Dict):
        if self.event_callback:
            try:
                self.event_callback({"type": event_type, **data, "timestamp": datetime.now().isoformat()})
            except Exception:
                pass

    # ── 영속성 ─────────────────────────────────────────────────────────────

    def _save_state(self):
        try:
            CONSCIOUSNESS_DB.parent.mkdir(parents=True, exist_ok=True)
            state_data = {
                "cognitive_load": self.state.cognitive_load,
                "overall_confidence": self.state.overall_confidence,
                "total_evaluations": self.state.total_evaluations,
                "rethinks_triggered": self.state.rethinks_triggered,
                "hallucinations_caught": self.state.hallucinations_caught,
                "contradictions_found": self.state.contradictions_found,
                "active_since": self.state.active_since,
            }
            CONSCIOUSNESS_DB.write_text(json.dumps(state_data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[Consciousness] Save error: {e}")

    def _load_state(self):
        try:
            if CONSCIOUSNESS_DB.exists():
                data = json.loads(CONSCIOUSNESS_DB.read_text(encoding="utf-8"))
                self.state.cognitive_load = data.get("cognitive_load", 0.3)
                self.state.overall_confidence = data.get("overall_confidence", 0.8)
                self.state.total_evaluations = data.get("total_evaluations", 0)
                self.state.rethinks_triggered = data.get("rethinks_triggered", 0)
                self.state.hallucinations_caught = data.get("hallucinations_caught", 0)
                self.state.contradictions_found = data.get("contradictions_found", 0)
                logger.info(f"[Consciousness] Loaded: {self.state.total_evaluations} past evaluations")
        except Exception as e:
            logger.debug(f"[Consciousness] Load error: {e}")

    def get_recent_evaluations(self, n: int = 10) -> List[Dict]:
        evals = list(reversed(list(self._eval_history)))[:n]
        return [
            {
                "query": e.query[:80],
                "quality_score": e.quality_score,
                "confidence": e.confidence,
                "hallucination_risk": e.hallucination_risk,
                "needs_rethink": e.needs_rethink,
                "rethink_reason": e.rethink_reason,
                "contradictions": e.contradictions,
                "timestamp": e.timestamp,
            }
            for e in evals
        ]
