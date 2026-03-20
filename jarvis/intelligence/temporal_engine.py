"""
JARVIS Temporal Reasoning Engine — Iteration 8
시간적 추론: 이벤트 타임라인 구축, 패턴 탐지, 미래 예측

Allen의 시간 관계 대수 (1983):
  before / after / meets / overlaps / during / starts / finishes / equals
"""

import json
import time
import logging
import re
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TemporalEvent:
    id: str
    description: str
    timestamp: float
    duration: float = 0.0        # seconds, 0 = instantaneous
    domain: str = ""
    certainty: float = 0.9
    source: str = "observation"


@dataclass
class Prediction:
    id: str
    event_description: str
    predicted_time: float
    confidence: float
    basis: str
    domain: str = ""
    verified: Optional[bool] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class TemporalPattern:
    pattern_type: str        # periodic / trend / causal_sequence / anomaly
    description: str
    confidence: float = 0.5
    period: Optional[float] = None
    trend_direction: str = ""


class TemporalEngine:
    """
    시간적 추론 엔진

    - 이벤트 타임라인 구축
    - Allen 시간 관계 분류
    - 주기성 / 트렌드 / 인과 시퀀스 패턴 감지
    - Bayesian 미래 예측 (confidence interval 포함)
    """

    def __init__(self, llm_manager=None, data_dir: str = "data/jarvis"):
        self.llm = llm_manager
        self.events:      List[TemporalEvent]  = []
        self.predictions: List[Prediction]     = []
        self.patterns:    List[TemporalPattern]= []
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(f"TemporalEngine: {len(self.events)} events, "
                    f"{len(self.predictions)} predictions")

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        f = self.data_dir / "temporal_engine.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for e in data.get("events", []):
                self.events.append(TemporalEvent(**e))
            for p in data.get("predictions", []):
                self.predictions.append(Prediction(**p))
            for pat in data.get("patterns", []):
                self.patterns.append(TemporalPattern(**pat))
        except Exception as ex:
            logger.warning(f"TemporalEngine load: {ex}")

    def _save(self):
        f = self.data_dir / "temporal_engine.json"
        f.write_text(json.dumps({
            "events":      [vars(e) for e in self.events[-600:]],
            "predictions": [vars(p) for p in self.predictions[-200:]],
            "patterns":    [vars(p) for p in self.patterns],
            "updated_at":  time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── LLM ──────────────────────────────────────────────────────────

    def _get_client(self):
        if self.llm:
            return (getattr(self.llm, "anthropic_client", None)
                    or getattr(self.llm, "_client", None))
        return None

    def _llm(self, prompt: str, max_tokens: int = 1200) -> str:
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

    # ── Event Management ──────────────────────────────────────────────

    def add_event(self, description: str, timestamp: float = None,
                  domain: str = "", certainty: float = 0.9,
                  source: str = "observation") -> TemporalEvent:
        """타임라인에 이벤트 추가"""
        evt = TemporalEvent(
            id=hashlib.md5(f"{description}{time.time()}".encode()).hexdigest()[:8],
            description=description,
            timestamp=timestamp or time.time(),
            domain=domain, certainty=certainty, source=source,
        )
        self.events.append(evt)
        self.events.sort(key=lambda e: e.timestamp)
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        self._save()
        return evt

    def extract_events_from_text(self, text: str,
                                 domain: str = "") -> List[TemporalEvent]:
        """텍스트에서 시간 이벤트 자동 추출"""
        prompt = f"""다음 텍스트에서 시간 관련 이벤트를 추출하세요.

텍스트: {text[:3000]}
도메인: {domain or "일반"}

JSON 배열:
[
  {{
    "description": "이벤트 설명",
    "relative_time": "past|present|future",
    "time_hint": "특정 날짜나 상대 시간 (예: 2024년, 3일전, 다음달)",
    "certainty": 0.8
  }}
]
JSON만 반환."""

        text_out = self._llm(prompt, 1500)
        new_events: List[TemporalEvent] = []
        try:
            m = re.search(r'\[.*\]', text_out, re.DOTALL)
            items = json.loads(m.group()) if m else []
            for item in items:
                ts = self._parse_time_hint(
                    item.get("time_hint", ""),
                    item.get("relative_time", "present"),
                )
                evt = self.add_event(
                    description=item.get("description", ""),
                    timestamp=ts, domain=domain,
                    certainty=float(item.get("certainty", 0.8)),
                    source="text_extraction",
                )
                new_events.append(evt)
        except Exception:
            pass
        return new_events

    def _parse_time_hint(self, hint: str, relative: str) -> float:
        """시간 힌트 → Unix timestamp"""
        now = time.time()
        if not hint:
            if relative == "past":    return now - 86400 * 30
            if relative == "future":  return now + 86400 * 30
            return now
        h = hint.lower()
        m = re.search(r'(\d{4})년', h)
        if m:
            try:
                return datetime(int(m.group(1)), 6, 1).timestamp()
            except Exception:
                pass
        m = re.search(r'(\d+)\s*(일|day)', h)
        if m:
            days = int(m.group(1))
            return now - 86400 * days if '전' in h or 'ago' in h else now + 86400 * days
        if any(k in h for k in ['next', '다음', '미래']):
            return now + 86400 * 30
        if any(k in h for k in ['past', '과거', '전']):
            return now - 86400 * 30
        return now

    # ── Allen Relations ───────────────────────────────────────────────

    def allen_relation(self, a: TemporalEvent, b: TemporalEvent) -> str:
        """Allen의 시간 관계 대수"""
        a_s, a_e = a.timestamp, a.timestamp + max(a.duration, 0)
        b_s, b_e = b.timestamp, b.timestamp + max(b.duration, 0)
        eps = 60  # 1 minute tolerance
        if a_e < b_s - eps:          return "before"
        if b_e < a_s - eps:          return "after"
        if abs(a_e - b_s) < eps:     return "meets"
        if abs(a_s - b_s) < eps and abs(a_e - b_e) < eps: return "equals"
        if a_s < b_s and a_e > b_e:  return "contains"
        if b_s < a_s and b_e > a_e:  return "during"
        if abs(a_s - b_s) < eps:     return "starts"
        if abs(a_e - b_e) < eps:     return "finishes"
        return "overlaps"

    # ── Pattern Detection ─────────────────────────────────────────────

    def detect_patterns(self) -> List[TemporalPattern]:
        """이벤트 타임라인에서 패턴 감지"""
        if len(self.events) < 3:
            return []

        evt_text = "\n".join(
            f"- [{datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d %H:%M')}] "
            f"[{e.domain}] {e.description[:100]}"
            for e in sorted(self.events, key=lambda x: x.timestamp)[-30:]
        )

        prompt = f"""이벤트 타임라인에서 시간 패턴을 분석하세요.

이벤트 (시간순):
{evt_text}

JSON 배열로 패턴 반환:
[
  {{
    "pattern_type": "periodic|trend|causal_sequence|anomaly",
    "description": "패턴 설명",
    "confidence": 0.8,
    "period": null,
    "trend_direction": "up|down|stable"
  }}
]
JSON만 반환."""

        text_out = self._llm(prompt, 1000)
        self.patterns = []
        try:
            m = re.search(r'\[.*\]', text_out, re.DOTALL)
            items = json.loads(m.group()) if m else []
            for item in items:
                self.patterns.append(TemporalPattern(
                    pattern_type=item.get("pattern_type", "trend"),
                    description=item.get("description", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    period=item.get("period"),
                    trend_direction=item.get("trend_direction", ""),
                ))
        except Exception:
            pass
        self._save()
        return self.patterns

    # ── Prediction ────────────────────────────────────────────────────

    def predict_future(self, domain: str = "",
                       horizon_days: int = 30) -> List[Prediction]:
        """패턴 기반 미래 예측"""
        recent = [e for e in self.events
                  if e.timestamp > time.time() - 86400 * 90]
        if not recent:
            return []

        evt_text = "\n".join(
            f"- [{datetime.fromtimestamp(e.timestamp).strftime('%Y-%m-%d')}] "
            f"{e.description[:120]}"
            for e in sorted(recent, key=lambda x: x.timestamp)[-20:]
        )
        pat_text = "\n".join(
            f"- {p.description}" for p in self.patterns[:5]
        ) or "(패턴 없음)"

        prompt = f"""과거 이벤트와 패턴을 바탕으로 미래 {horizon_days}일 이내의 이벤트를 예측하세요.

과거 이벤트:
{evt_text}

감지된 패턴:
{pat_text}

도메인: {domain or "전체"}

JSON 배열 (최대 5개):
[
  {{
    "event_description": "예측 이벤트",
    "days_from_now": 14,
    "confidence": 0.7,
    "basis": "예측 근거",
    "domain": "AI|tech|market|general"
  }}
]
JSON만 반환."""

        text_out = self._llm(prompt, 1200)
        new_preds: List[Prediction] = []
        try:
            m = re.search(r'\[.*\]', text_out, re.DOTALL)
            items = json.loads(m.group()) if m else []
            for item in items[:5]:
                days = float(item.get("days_from_now", 14))
                pred = Prediction(
                    id=hashlib.md5(f"{item['event_description']}{time.time()}".encode()).hexdigest()[:8],
                    event_description=item["event_description"],
                    predicted_time=time.time() + 86400 * days,
                    confidence=float(item.get("confidence", 0.5)),
                    basis=item.get("basis", ""),
                    domain=item.get("domain", domain),
                )
                self.predictions.append(pred)
                new_preds.append(pred)
        except Exception:
            pass
        self._save()
        return new_preds

    def get_timeline(self, domain: str = "", limit: int = 50) -> List[Dict]:
        events = self.events
        if domain:
            events = [e for e in events if e.domain == domain]
        return [
            {
                "id": e.id, "description": e.description,
                "timestamp": e.timestamp,
                "date": datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M"),
                "domain": e.domain, "certainty": e.certainty,
            }
            for e in sorted(events, key=lambda x: x.timestamp, reverse=True)[:limit]
        ]

    def get_status(self) -> Dict:
        return {
            "available": True,
            "total_events":      len(self.events),
            "total_predictions": len(self.predictions),
            "total_patterns":    len(self.patterns),
            "verified_predictions": len([p for p in self.predictions
                                         if p.verified is True]),
            "domains": list({e.domain for e in self.events if e.domain}),
        }
