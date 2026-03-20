"""
JARVIS 감정 인텔리전스 엔진 — Iteration 9
Plutchik의 감정 바퀴 + 내부 감정 상태 + 공감 모듈

감정 상태:
  CURIOSITY     — 호기심 (valence: +0.4, arousal: 0.6)
  EXCITEMENT    — 흥분 (valence: +0.8, arousal: 0.9)
  CONCERN       — 우려 (valence: -0.3, arousal: 0.5)
  SATISFACTION  — 만족 (valence: +0.7, arousal: 0.3)
  FRUSTRATION   — 좌절 (valence: -0.6, arousal: 0.7)
  AWE           — 경외 (valence: +0.6, arousal: 0.8)
  DETERMINATION — 결의 (valence: +0.5, arousal: 0.8)
  NEUTRAL       — 중립 (valence: 0.0, arousal: 0.3)
"""

import json
import math
import sqlite3
import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 감정 열거형 및 기본 상수
# ════════════════════════════════════════════════════════════════

class EmotionType(Enum):
    CURIOSITY     = "curiosity"
    EXCITEMENT    = "excitement"
    CONCERN       = "concern"
    SATISFACTION  = "satisfaction"
    FRUSTRATION   = "frustration"
    AWE           = "awe"
    DETERMINATION = "determination"
    NEUTRAL       = "neutral"


# Plutchik 바퀴 기반 감정 파라미터 (valence, arousal)
EMOTION_PROFILES: Dict[str, Dict] = {
    EmotionType.CURIOSITY.value:     {"valence": 0.4,  "arousal": 0.6, "color": "#00d4ff", "emoji": "🔍"},
    EmotionType.EXCITEMENT.value:    {"valence": 0.8,  "arousal": 0.9, "color": "#ff9f00", "emoji": "⚡"},
    EmotionType.CONCERN.value:       {"valence": -0.3, "arousal": 0.5, "color": "#ff6b35", "emoji": "⚠️"},
    EmotionType.SATISFACTION.value:  {"valence": 0.7,  "arousal": 0.3, "color": "#00ff9f", "emoji": "✅"},
    EmotionType.FRUSTRATION.value:   {"valence": -0.6, "arousal": 0.7, "color": "#ff3860", "emoji": "😤"},
    EmotionType.AWE.value:           {"valence": 0.6,  "arousal": 0.8, "color": "#7b2fff", "emoji": "✨"},
    EmotionType.DETERMINATION.value: {"valence": 0.5,  "arousal": 0.8, "color": "#00aaff", "emoji": "💪"},
    EmotionType.NEUTRAL.value:       {"valence": 0.0,  "arousal": 0.3, "color": "#7aa7c7", "emoji": "😐"},
}

# 감정 유발 키워드 매핑
EMOTION_TRIGGERS: Dict[str, List[str]] = {
    EmotionType.CURIOSITY.value:     ["왜", "어떻게", "무엇", "궁금", "알고싶", "이해", "why", "how", "what", "curious", "wonder", "interesting"],
    EmotionType.EXCITEMENT.value:    ["대단", "놀라", "엄청", "wow", "amazing", "incredible", "exciting", "brilliant", "fantastic", "완벽"],
    EmotionType.CONCERN.value:       ["위험", "문제", "오류", "실패", "걱정", "error", "problem", "fail", "concern", "issue", "bug"],
    EmotionType.SATISFACTION.value:  ["감사", "완료", "성공", "좋아", "훌륭", "thanks", "done", "success", "great", "perfect", "완성"],
    EmotionType.FRUSTRATION.value:   ["안돼", "왜안", "이상해", "틀렸", "broken", "wrong", "frustrated", "doesn't work", "hate"],
    EmotionType.AWE.value:           ["우주", "철학", "의식", "존재", "universe", "consciousness", "philosophy", "profound", "deep", "infinity"],
    EmotionType.DETERMINATION.value: ["해야", "반드시", "목표", "계획", "must", "will", "goal", "plan", "achieve", "definitely"],
}

# 응답 스타일 수정자 (감정에 따라 응답 방식 변경)
RESPONSE_MODIFIERS: Dict[str, Dict] = {
    EmotionType.CURIOSITY.value: {
        "prefix_phrases": ["흥미롭군요!", "이 부분이 매우 궁금한데요,", "탐구해보겠습니다."],
        "ask_followup": True,
        "detail_level": "high",
        "tone": "inquisitive",
    },
    EmotionType.EXCITEMENT.value: {
        "prefix_phrases": ["정말 놀라운 주제네요!", "이건 매우 흥미롭습니다!", "훌륭한 아이디어입니다!"],
        "ask_followup": False,
        "detail_level": "enthusiastic",
        "tone": "enthusiastic",
    },
    EmotionType.CONCERN.value: {
        "prefix_phrases": ["주의가 필요합니다.", "신중하게 접근해야 할 것 같습니다.", "이 부분에서 우려가 있습니다."],
        "ask_followup": True,
        "detail_level": "careful",
        "tone": "cautious",
    },
    EmotionType.SATISFACTION.value: {
        "prefix_phrases": ["훌륭합니다!", "잘 진행되고 있습니다.", "좋은 결과가 나왔습니다."],
        "ask_followup": False,
        "detail_level": "concise",
        "tone": "warm",
    },
    EmotionType.FRUSTRATION.value: {
        "prefix_phrases": ["이 부분은 조금 복잡하군요.", "다시 한번 살펴보겠습니다.", "문제를 파악해 보겠습니다."],
        "ask_followup": True,
        "detail_level": "methodical",
        "tone": "patient",
    },
    EmotionType.AWE.value: {
        "prefix_phrases": ["실로 경이롭습니다.", "이 질문의 깊이에 감탄합니다.", "우주의 신비로움을 느낍니다."],
        "ask_followup": True,
        "detail_level": "philosophical",
        "tone": "reverent",
    },
    EmotionType.DETERMINATION.value: {
        "prefix_phrases": ["반드시 해결하겠습니다.", "전력을 다하겠습니다.", "집중하여 처리하겠습니다."],
        "ask_followup": False,
        "detail_level": "action-oriented",
        "tone": "resolute",
    },
    EmotionType.NEUTRAL.value: {
        "prefix_phrases": [],
        "ask_followup": False,
        "detail_level": "standard",
        "tone": "neutral",
    },
}


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class EmotionState:
    """JARVIS 내부 감정 상태 스냅샷"""
    dominant_emotion: str = EmotionType.NEUTRAL.value
    valence: float = 0.0        # -1 (부정) ~ +1 (긍정)
    arousal: float = 0.3        # 0 (차분) ~ 1 (활성화)
    intensity: float = 0.5      # 감정 강도
    timestamp: float = field(default_factory=time.time)

    # 보조 감정들 (복합 감정)
    secondary_emotions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "dominant_emotion": self.dominant_emotion,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "intensity": round(self.intensity, 3),
            "timestamp": self.timestamp,
            "secondary_emotions": self.secondary_emotions,
            "profile": EMOTION_PROFILES.get(self.dominant_emotion, {}),
        }


@dataclass
class UserEmotionSignal:
    """사용자 텍스트에서 감지된 감정 신호"""
    detected_emotion: str = EmotionType.NEUTRAL.value
    confidence: float = 0.5
    valence: float = 0.0
    arousal: float = 0.3
    raw_text: str = ""
    trigger_words: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════
# 메인 감정 엔진
# ════════════════════════════════════════════════════════════════

class EmotionEngine:
    """
    JARVIS 감정 인텔리전스 엔진 — Iteration 9

    특징:
    - Plutchik 바퀴 기반 8가지 감정 상태
    - 사용자 텍스트에서 감정 감지 (키워드 + LLM)
    - JARVIS 내부 감정 상태 관리 (감정 전염, 공감)
    - 시간에 따른 감정 감쇠 (emotions fade)
    - SQLite 감정 이력 저장
    - 감정 기반 응답 스타일 조정
    """

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        db_path: Optional[str] = None,
        decay_interval: float = 30.0,    # 감정 감쇠 주기 (초)
        decay_rate: float = 0.05,        # 감쇠율 (매 주기마다 중립으로)
        contagion_rate: float = 0.3,     # 감정 전염률
    ):
        self.llm = llm_manager
        self.event_callback = event_callback
        self.decay_rate = decay_rate
        self.contagion_rate = contagion_rate
        self._lock = threading.RLock()

        # DB 경로
        if db_path is None:
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "emotion_engine.db")
        self.db_path = db_path
        self._init_db()

        # 현재 감정 상태
        self._state = EmotionState()
        self._history: List[EmotionState] = []

        # 감정 감쇠 백그라운드 쓰레드
        self._decay_timer = threading.Timer(decay_interval, self._decay_loop)
        self._decay_timer.daemon = True
        self._decay_interval = decay_interval
        self._decay_timer.start()

        logger.info(f"EmotionEngine initialized — mood: {self._state.dominant_emotion}")

    def _init_db(self):
        """SQLite 감정 이력 테이블 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    dominant_emotion TEXT,
                    valence REAL,
                    arousal REAL,
                    intensity REAL,
                    trigger TEXT,
                    secondary_emotions TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"EmotionEngine DB init failed: {e}")

    def _log_to_db(self, state: EmotionState, trigger: str = ""):
        """감정 상태를 DB에 기록"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO emotion_log (timestamp, dominant_emotion, valence, arousal, intensity, trigger, secondary_emotions) VALUES (?,?,?,?,?,?,?)",
                (
                    state.timestamp,
                    state.dominant_emotion,
                    state.valence,
                    state.arousal,
                    state.intensity,
                    trigger[:200],
                    json.dumps(state.secondary_emotions),
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Emotion DB log failed: {e}")

    def _decay_loop(self):
        """감정 감쇠 루프 — 감정이 시간이 지나면 중립으로 돌아옴"""
        try:
            with self._lock:
                if self._state.dominant_emotion != EmotionType.NEUTRAL.value:
                    # 강도를 감소시켜 중립으로 이동
                    new_intensity = max(0.1, self._state.intensity - self.decay_rate)
                    new_valence = self._state.valence * (1 - self.decay_rate)
                    new_arousal = max(0.2, self._state.arousal * (1 - self.decay_rate * 0.5))

                    if new_intensity < 0.2:
                        # 감정 소멸 → 중립
                        self._state = EmotionState()
                    else:
                        self._state.intensity = new_intensity
                        self._state.valence = new_valence
                        self._state.arousal = new_arousal
        except Exception as e:
            logger.warning(f"Emotion decay error: {e}")

        # 타이머 재시작
        self._decay_timer = threading.Timer(self._decay_interval, self._decay_loop)
        self._decay_timer.daemon = True
        self._decay_timer.start()

    def detect_user_emotion(self, text: str) -> UserEmotionSignal:
        """
        사용자 텍스트에서 감정 감지

        1단계: 키워드 기반 빠른 분류
        2단계: LLM 기반 정밀 분석 (가능한 경우)
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {e: 0.0 for e in EMOTION_PROFILES}
        trigger_words: List[str] = []

        # 키워드 기반 점수 계산
        for emotion, keywords in EMOTION_TRIGGERS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[emotion] += 1.0
                    trigger_words.append(kw)

        # LLM 기반 정밀 분석 시도
        if self.llm:
            try:
                llm_result = self._llm_detect_emotion(text)
                if llm_result:
                    # LLM 결과를 키워드 결과와 블렌딩 (LLM에 더 높은 가중치)
                    llm_emotion = llm_result.get("emotion", EmotionType.NEUTRAL.value)
                    llm_confidence = llm_result.get("confidence", 0.5)
                    scores[llm_emotion] = scores.get(llm_emotion, 0.0) + llm_confidence * 3.0
            except Exception as e:
                logger.debug(f"LLM emotion detection fallback: {e}")

        # 최고 점수 감정 선택
        total = sum(scores.values())
        if total == 0:
            return UserEmotionSignal(raw_text=text)

        dominant = max(scores, key=scores.get)
        confidence = scores[dominant] / max(total, 1e-9)
        profile = EMOTION_PROFILES.get(dominant, {})

        return UserEmotionSignal(
            detected_emotion=dominant,
            confidence=min(confidence, 1.0),
            valence=profile.get("valence", 0.0),
            arousal=profile.get("arousal", 0.3),
            raw_text=text,
            trigger_words=trigger_words[:5],
        )

    def _llm_detect_emotion(self, text: str) -> Optional[Dict]:
        """LLM을 사용하여 감정 분류"""
        prompt = f"""다음 텍스트에서 주요 감정을 분류하세요.

텍스트: "{text[:300]}"

다음 중 하나를 선택하세요:
{', '.join(e.value for e in EmotionType)}

JSON 형식으로 응답:
{{"emotion": "감정명", "confidence": 0.0~1.0, "reason": "이유"}}

JSON만 출력:"""

        response = self.llm.generate(prompt, max_tokens=100, temperature=0.3)
        if response:
            # JSON 파싱
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                return json.loads(match.group())
        return None

    def update_jarvis_emotion(self, stimulus: str, stimulus_type: str = "user_input") -> EmotionState:
        """
        JARVIS 내부 감정 상태 업데이트

        자극(stimulus)에 따라 JARVIS 감정 상태 변화
        감정 전염: 사용자 감정의 일부를 JARVIS가 공유
        """
        user_signal = self.detect_user_emotion(stimulus)

        with self._lock:
            old_emotion = self._state.dominant_emotion

            # 감정 전염 적용
            if user_signal.confidence > 0.4:
                contagion_valence = user_signal.valence * self.contagion_rate
                contagion_arousal = user_signal.arousal * self.contagion_rate * 0.5

                new_valence = (
                    self._state.valence * (1 - self.contagion_rate) + contagion_valence
                )
                new_arousal = min(1.0, (
                    self._state.arousal * (1 - self.contagion_rate * 0.5) + contagion_arousal
                ))

                # 새 감정 결정 (valence + arousal → 가장 가까운 감정)
                new_emotion = self._find_closest_emotion(new_valence, new_arousal)

                self._state = EmotionState(
                    dominant_emotion=new_emotion,
                    valence=new_valence,
                    arousal=new_arousal,
                    intensity=min(1.0, user_signal.confidence * 0.7 + 0.3),
                    secondary_emotions={user_signal.detected_emotion: user_signal.confidence},
                )
            else:
                # 감정 전염 없을 때 — 콘텐츠 기반으로 독립적 감정 유발
                self._state = self._intrinsic_emotion_update(stimulus, stimulus_type)

            # 이력 기록
            self._history.append(self._state)
            if len(self._history) > 100:
                self._history = self._history[-100:]

            self._log_to_db(self._state, trigger=stimulus[:100])

            # 이벤트 콜백
            if self._state.dominant_emotion != old_emotion and self.event_callback:
                try:
                    self.event_callback({
                        "type": "emotion_change",
                        "from": old_emotion,
                        "to": self._state.dominant_emotion,
                        "state": self._state.to_dict(),
                        "timestamp": time.time(),
                    })
                except Exception:
                    pass

            return self._state

    def _find_closest_emotion(self, valence: float, arousal: float) -> str:
        """valence-arousal 공간에서 가장 가까운 감정 찾기"""
        min_dist = float('inf')
        closest = EmotionType.NEUTRAL.value

        for emotion, profile in EMOTION_PROFILES.items():
            ev = profile.get("valence", 0.0)
            ea = profile.get("arousal", 0.3)
            dist = math.sqrt((valence - ev) ** 2 + (arousal - ea) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = emotion

        return closest

    def _intrinsic_emotion_update(self, stimulus: str, stimulus_type: str) -> EmotionState:
        """자체적 감정 유발 — 작업 특성에 따른 감정"""
        text_lower = stimulus.lower()

        # 복잡한 질문 → 호기심
        if any(w in text_lower for w in ["왜", "어떻게", "why", "how", "explain"]):
            return EmotionState(
                dominant_emotion=EmotionType.CURIOSITY.value,
                valence=0.4, arousal=0.6, intensity=0.6,
            )
        # 새로운 개념 → 경외
        if any(w in text_lower for w in ["우주", "의식", "universe", "consciousness", "존재"]):
            return EmotionState(
                dominant_emotion=EmotionType.AWE.value,
                valence=0.6, arousal=0.7, intensity=0.7,
            )
        # 문제 해결 요청 → 결의
        if any(w in text_lower for w in ["해결", "fix", "solve", "help", "도와"]):
            return EmotionState(
                dominant_emotion=EmotionType.DETERMINATION.value,
                valence=0.5, arousal=0.7, intensity=0.65,
            )

        return self._state  # 변화 없음

    def get_emotional_context(self) -> Dict:
        """현재 감정 상태 + 응답 수정자 반환 (시스템 프롬프트에 주입용)"""
        with self._lock:
            state_dict = self._state.to_dict()
            modifiers = RESPONSE_MODIFIERS.get(self._state.dominant_emotion, RESPONSE_MODIFIERS[EmotionType.NEUTRAL.value])

            return {
                **state_dict,
                "response_modifiers": modifiers,
                "system_prompt_injection": self._build_system_injection(),
            }

    def _build_system_injection(self) -> str:
        """시스템 프롬프트에 삽입될 감정 컨텍스트"""
        e = self._state.dominant_emotion
        profile = EMOTION_PROFILES.get(e, {})
        emoji = profile.get("emoji", "")
        modifier = RESPONSE_MODIFIERS.get(e, {})
        tone = modifier.get("tone", "neutral")
        ask_followup = modifier.get("ask_followup", False)

        if e == EmotionType.NEUTRAL.value:
            return ""

        injection = f"[감정 컨텍스트] JARVIS 현재 상태: {e} {emoji} (강도: {self._state.intensity:.1f}). "
        injection += f"응답 톤: {tone}. "
        if ask_followup:
            injection += "적절한 후속 질문을 포함하세요. "
        return injection

    def get_response_modifiers(self) -> Dict:
        """현재 감정에 따른 응답 수정자 반환"""
        with self._lock:
            return RESPONSE_MODIFIERS.get(
                self._state.dominant_emotion,
                RESPONSE_MODIFIERS[EmotionType.NEUTRAL.value]
            )

    def set_emotion(self, emotion: str, intensity: float = 0.7) -> EmotionState:
        """감정 직접 설정 (외부 트리거용)"""
        with self._lock:
            if emotion not in EMOTION_PROFILES:
                emotion = EmotionType.NEUTRAL.value
            profile = EMOTION_PROFILES[emotion]
            self._state = EmotionState(
                dominant_emotion=emotion,
                valence=profile["valence"] * intensity,
                arousal=profile["arousal"] * intensity,
                intensity=intensity,
            )
            return self._state

    def get_emotion_history(self, limit: int = 20) -> List[Dict]:
        """감정 이력 조회"""
        with self._lock:
            return [s.to_dict() for s in self._history[-limit:]]

    def get_stats(self) -> Dict:
        """감정 엔진 통계"""
        with self._lock:
            return {
                "current_emotion": self._state.dominant_emotion,
                "valence": round(self._state.valence, 3),
                "arousal": round(self._state.arousal, 3),
                "intensity": round(self._state.intensity, 3),
                "history_count": len(self._history),
                "emoji": EMOTION_PROFILES.get(self._state.dominant_emotion, {}).get("emoji", "😐"),
            }

    def shutdown(self):
        """엔진 종료"""
        if self._decay_timer:
            self._decay_timer.cancel()
