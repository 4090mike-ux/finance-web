"""
JARVIS 예측 엔진 — Iteration 4
사용자 행동 패턴을 분석하여 다음 요구사항을 선제적으로 예측·준비
- 사용 패턴 학습 (시간대, 빈도, 주제 클러스터)
- 컨텍스트 인식 (시간, 요일, 직전 작업)
- 선제적 정보 수집 (백그라운드 사전 준비)
- 자연어 의도 분류 (정보 탐색 / 작업 실행 / 창작 / 분석)
- 신뢰도 기반 자동 실행 임계값
"""

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STORAGE_PATH = Path("data/jarvis/prediction_state.json")


@dataclass
class Prediction:
    intent: str               # 예측된 의도 (inform / execute / create / analyze)
    topic: str                # 예측 주제
    suggested_action: str     # 권고 행동
    confidence: float         # 0–1
    reason: str               # 예측 근거
    preloaded_data: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserPattern:
    hour_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    topic_frequency: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    intent_history: List[str] = field(default_factory=list)
    session_sequences: List[List[str]] = field(default_factory=list)  # [[topic1, topic2], ...]
    last_topics: List[str] = field(default_factory=list)
    total_interactions: int = 0


class PredictionEngine:
    """
    JARVIS 선제적 예측 엔진
    사용자가 무엇을 원하는지 묻기 전에 준비
    """

    INTENT_KEYWORDS = {
        "inform": ["뭐야", "알려줘", "설명", "what", "explain", "how", "why", "정보", "검색"],
        "execute": ["실행", "해줘", "run", "execute", "만들어", "생성", "create", "write"],
        "analyze": ["분석", "비교", "평가", "analyze", "compare", "review", "검토", "assess"],
        "create": ["코드", "작성", "generate", "code", "script", "프로그램", "개발"],
        "research": ["논문", "연구", "arxiv", "github", "최신", "트렌드", "paper"],
    }

    TOPIC_PATTERNS = [
        ("AI/ML", ["ai", "machine learning", "deep learning", "llm", "gpt", "claude", "neural", "모델", "학습"]),
        ("코딩", ["python", "code", "function", "class", "bug", "error", "debug", "코드", "함수"]),
        ("데이터", ["data", "csv", "database", "sql", "dataframe", "분석", "데이터"]),
        ("시스템", ["server", "cpu", "memory", "process", "file", "system", "시스템", "서버"]),
        ("연구", ["paper", "arxiv", "research", "논문", "연구", "study"]),
        ("자동화", ["automation", "schedule", "cron", "task", "workflow", "자동", "스케줄"]),
    ]

    def __init__(self, llm_manager=None, background_callback: Optional[Callable] = None):
        self.llm = llm_manager
        self._bg_callback = background_callback
        self._pattern = UserPattern()
        self._pending_predictions: List[Prediction] = []
        self._preload_cache: Dict[str, Any] = {}
        self._load_state()
        logger.info("PredictionEngine initialized")

    # ── 학습 ─────────────────────────────────────────────────────────────────
    def record_interaction(self, user_input: str, response_summary: str = ""):
        """상호작용 기록 → 패턴 업데이트"""
        now = datetime.now()
        hour = now.hour
        self._pattern.hour_distribution[hour] = self._pattern.hour_distribution.get(hour, 0) + 1
        self._pattern.total_interactions += 1

        intent = self._classify_intent(user_input)
        topic = self._classify_topic(user_input)

        self._pattern.intent_history.append(intent)
        if len(self._pattern.intent_history) > 200:
            self._pattern.intent_history = self._pattern.intent_history[-200:]

        freq = self._pattern.topic_frequency
        freq[topic] = freq.get(topic, 0) + 1

        self._pattern.last_topics.append(topic)
        if len(self._pattern.last_topics) > 10:
            self._pattern.last_topics = self._pattern.last_topics[-10:]

        # 세션 시퀀스 업데이트
        if self._pattern.session_sequences and len(self._pattern.session_sequences[-1]) < 5:
            self._pattern.session_sequences[-1].append(topic)
        else:
            self._pattern.session_sequences.append([topic])
        if len(self._pattern.session_sequences) > 50:
            self._pattern.session_sequences = self._pattern.session_sequences[-50:]

        self._save_state()

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        scores = {intent: 0 for intent in self.INTENT_KEYWORDS}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[intent] += 1
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "inform"

    def _classify_topic(self, text: str) -> str:
        text_lower = text.lower()
        for topic, keywords in self.TOPIC_PATTERNS:
            if any(kw in text_lower for kw in keywords):
                return topic
        return "기타"

    # ── 예측 ─────────────────────────────────────────────────────────────────
    def predict_next(self, current_input: str = "") -> List[Prediction]:
        """현재 컨텍스트에서 다음 요청 예측"""
        predictions = []

        # 1. 시간 기반 예측
        hour_pred = self._time_based_prediction()
        if hour_pred:
            predictions.append(hour_pred)

        # 2. 시퀀스 패턴 기반 예측
        seq_pred = self._sequence_based_prediction()
        if seq_pred:
            predictions.append(seq_pred)

        # 3. LLM 기반 고급 예측
        if self.llm and current_input:
            llm_pred = self._llm_prediction(current_input)
            if llm_pred:
                predictions.append(llm_pred)

        # 신뢰도 정렬
        predictions.sort(key=lambda p: -p.confidence)
        self._pending_predictions = predictions[:5]
        return self._pending_predictions

    def _time_based_prediction(self) -> Optional[Prediction]:
        """시간대 패턴으로 예측"""
        hour = datetime.now().hour
        hour_dist = self._pattern.hour_distribution
        if not hour_dist:
            return None

        # 이 시간대에 자주 하는 작업 찾기
        hour_topics = []
        for seq in self._pattern.session_sequences:
            # 단순화: 기존 레코드에서 시간대 매칭 없이 전체 빈도 사용
            hour_topics.extend(seq)

        if not hour_topics:
            return None

        top = Counter(hour_topics).most_common(1)[0]
        return Prediction(
            intent="inform",
            topic=top[0],
            suggested_action=f"{top[0]} 관련 최신 정보 준비",
            confidence=min(0.6, top[1] / 20),
            reason=f"이 시간대({hour}시)에 {top[0]} 작업을 자주 합니다",
        )

    def _sequence_based_prediction(self) -> Optional[Prediction]:
        """이전 작업 시퀀스로 다음 작업 예측"""
        if len(self._pattern.last_topics) < 2:
            return None

        # Markov chain: 현재 topic → 다음 topic
        last = self._pattern.last_topics[-1]
        next_counts: Dict[str, int] = defaultdict(int)
        for seq in self._pattern.session_sequences:
            for i in range(len(seq) - 1):
                if seq[i] == last:
                    next_counts[seq[i + 1]] += 1

        if not next_counts:
            return None

        predicted_next = max(next_counts, key=next_counts.get)
        total = sum(next_counts.values())
        confidence = next_counts[predicted_next] / total

        if confidence < 0.3:
            return None

        return Prediction(
            intent="execute",
            topic=predicted_next,
            suggested_action=f"{predicted_next} 작업 준비",
            confidence=confidence,
            reason=f"{last} 후에 {predicted_next}을 자주 합니다 ({confidence:.0%})",
        )

    def _llm_prediction(self, current_input: str) -> Optional[Prediction]:
        """LLM 기반 의도 예측"""
        from jarvis.llm.manager import Message

        top_topics = sorted(
            self._pattern.topic_frequency.items(), key=lambda x: -x[1]
        )[:5]
        recent = self._pattern.last_topics[-5:]

        prompt = f"""사용자의 최근 요청: "{current_input}"
최근 작업 주제: {recent}
자주 하는 작업: {[t for t, _ in top_topics]}
현재 시각: {datetime.now().strftime('%H:%M')} ({['월','화','수','목','금','토','일'][datetime.now().weekday()]}요일)

다음 형식으로 사용자의 다음 요구사항을 예측하세요:
{{
  "intent": "inform|execute|analyze|create|research",
  "topic": "예측 주제",
  "suggested_action": "선제적으로 준비할 행동",
  "confidence": 0.75,
  "reason": "예측 근거"
}}

JSON만 반환하세요."""

        messages = [Message(role="user", content=prompt)]
        try:
            resp = self.llm.chat(
                messages,
                system="당신은 사용자 행동 패턴을 분석하는 AI입니다.",
                max_tokens=512,
            )
            import re
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return Prediction(
                    intent=data.get("intent", "inform"),
                    topic=data.get("topic", ""),
                    suggested_action=data.get("suggested_action", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", ""),
                )
        except Exception as e:
            logger.debug(f"LLM prediction error: {e}")
        return None

    # ── 선제적 준비 ──────────────────────────────────────────────────────────
    def preload(self, prediction: Prediction, web_search_fn: Optional[Callable] = None):
        """예측된 주제 데이터를 백그라운드로 미리 준비"""
        if prediction.confidence < 0.5:
            return

        key = f"{prediction.topic}_{prediction.intent}"
        if key in self._preload_cache:
            return  # 이미 준비됨

        if web_search_fn and prediction.intent in ("inform", "research"):
            try:
                results = web_search_fn(f"{prediction.topic} 최신 정보")
                self._preload_cache[key] = {
                    "topic": prediction.topic,
                    "results": results,
                    "loaded_at": datetime.now().isoformat(),
                }
                logger.info(f"[Prediction] 선제적 로딩 완료: {prediction.topic}")
                if self._bg_callback:
                    self._bg_callback({
                        "type": "prediction_preloaded",
                        "topic": prediction.topic,
                        "confidence": prediction.confidence,
                    })
            except Exception as e:
                logger.debug(f"Preload error: {e}")

    def get_preloaded(self, topic: str) -> Optional[Dict]:
        """미리 준비된 데이터 조회"""
        for key, data in self._preload_cache.items():
            if topic.lower() in key.lower():
                return data
        return None

    # ── 프로액티브 알림 ──────────────────────────────────────────────────────
    def get_proactive_suggestions(self) -> List[str]:
        """현재 컨텍스트 기반 선제적 제안 생성"""
        suggestions = []
        hour = datetime.now().hour
        weekday = datetime.now().weekday()

        # 시간대 기반 제안
        if 9 <= hour <= 10:
            suggestions.append("오늘의 AI 뉴스 및 GitHub 트렌드를 요약해 드릴까요?")
        elif 13 <= hour <= 14:
            suggestions.append("오후 작업을 위한 세션 브리핑을 준비할까요?")
        elif 17 <= hour <= 18:
            suggestions.append("오늘 작업 내용을 요약하고 내일 계획을 세울까요?")
        elif hour >= 22:
            suggestions.append("야간 자율 학습 모드를 활성화할까요? (최신 논문 수집)")

        # 패턴 기반 제안
        if self._pattern.total_interactions > 10:
            top = sorted(self._pattern.topic_frequency.items(), key=lambda x: -x[1])[:1]
            if top:
                suggestions.append(f"{top[0][0]} 관련 최신 정보를 미리 수집해 둘까요?")

        # 주말
        if weekday >= 5:
            suggestions.append("주간 학습 요약 리포트를 생성할까요?")

        return suggestions[:3]

    # ── 통계 ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        total = self._pattern.total_interactions
        top_topics = sorted(
            self._pattern.topic_frequency.items(), key=lambda x: -x[1]
        )[:5]
        intent_count = Counter(self._pattern.intent_history)
        peak_hour = max(self._pattern.hour_distribution.items(), key=lambda x: x[1])[0] \
            if self._pattern.hour_distribution else None

        return {
            "total_interactions": total,
            "top_topics": top_topics,
            "intent_distribution": dict(intent_count.most_common(5)),
            "peak_hour": peak_hour,
            "cached_preloads": len(self._preload_cache),
            "pending_predictions": len(self._pending_predictions),
        }

    def format_predictions_markdown(self, predictions: List[Prediction]) -> str:
        if not predictions:
            return "예측된 다음 작업 없음"
        lines = ["## JARVIS 선제적 예측"]
        for i, p in enumerate(predictions, 1):
            lines.append(
                f"{i}. **{p.topic}** ({p.confidence:.0%} 신뢰도)\n"
                f"   의도: {p.intent} | 행동: {p.suggested_action}\n"
                f"   근거: {p.reason}"
            )
        return "\n".join(lines)

    # ── 영속성 ───────────────────────────────────────────────────────────────
    def _save_state(self):
        try:
            STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "hour_distribution": dict(self._pattern.hour_distribution),
                "topic_frequency": dict(self._pattern.topic_frequency),
                "intent_history": self._pattern.intent_history[-100:],
                "session_sequences": self._pattern.session_sequences[-30:],
                "last_topics": self._pattern.last_topics,
                "total_interactions": self._pattern.total_interactions,
            }
            STORAGE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Save state error: {e}")

    def _load_state(self):
        try:
            if STORAGE_PATH.exists():
                state = json.loads(STORAGE_PATH.read_text(encoding="utf-8"))
                p = self._pattern
                p.hour_distribution = defaultdict(int, {int(k): v for k, v in state.get("hour_distribution", {}).items()})
                p.topic_frequency = defaultdict(int, state.get("topic_frequency", {}))
                p.intent_history = state.get("intent_history", [])
                p.session_sequences = state.get("session_sequences", [])
                p.last_topics = state.get("last_topics", [])
                p.total_interactions = state.get("total_interactions", 0)
                logger.info(f"[Prediction] 상태 로드: {p.total_interactions}건 학습")
        except Exception as e:
            logger.debug(f"Load state error: {e}")
