"""
JARVIS 자율 루프 엔진 — Iteration 3
아무 명령 없이도 스스로 생각하고, 학습하고, 행동하는 시스템
- 주기적 지식 업데이트 (ArXiv, GitHub 트렌드, 뉴스)
- 자율 목표 생성 및 실행
- 시스템 자가 진단 및 개선 제안
- 사용자 패턴 학습 및 선제 작업
- 이상 탐지 및 알림
"""

import time
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AutonomousEventType(str, Enum):
    KNOWLEDGE_UPDATE = "knowledge_update"
    SELF_DIAGNOSIS = "self_diagnosis"
    TREND_ALERT = "trend_alert"
    ANOMALY_DETECTED = "anomaly_detected"
    PROACTIVE_SUGGESTION = "proactive_suggestion"
    GOAL_COMPLETED = "goal_completed"
    LEARNING_CYCLE = "learning_cycle"


@dataclass
class AutonomousEvent:
    type: AutonomousEventType
    title: str
    content: str
    priority: int = 5  # 1=critical, 10=low
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict = field(default_factory=dict)


class AutonomousLoop:
    """
    JARVIS 자율 지능 루프
    사용자와의 대화 없이 독립적으로 작동하는 에이전트 루프
    """

    # 각 작업의 실행 간격 (초)
    SCHEDULES = {
        "knowledge_update": 3600,      # 1시간 - ArXiv/GitHub 트렌드
        "memory_consolidate": 7200,    # 2시간 - 메모리 압축/정리
        "self_diagnosis": 1800,        # 30분 - 시스템 자가 진단
        "trend_monitor": 900,          # 15분 - AI/기술 트렌드 모니터링
        "skill_discovery": 14400,      # 4시간 - 새 스킬 자동 발견
        "anomaly_check": 300,          # 5분 - 시스템 이상 탐지
    }

    def __init__(
        self,
        jarvis_engine,
        memory_manager,
        web_intelligence,
        skill_library=None,
        event_callback: Callable = None,
    ):
        self.jarvis = jarvis_engine
        self.memory = memory_manager
        self.web = web_intelligence
        self.skills = skill_library
        self.event_callback = event_callback

        self.is_running = False
        self._threads: List[threading.Thread] = []
        self._last_run: Dict[str, float] = {}
        self._event_log: List[AutonomousEvent] = []
        self._stats = {
            "total_events": 0,
            "knowledge_updates": 0,
            "skills_discovered": 0,
            "anomalies_detected": 0,
            "started_at": None,
        }
        logger.info("AutonomousLoop initialized")

    # ── 시작/중지 ─────────────────────────────────────────────────────────
    def start(self):
        """자율 루프 시작"""
        if self.is_running:
            return
        self.is_running = True
        self._stats["started_at"] = datetime.now().isoformat()

        # 각 작업을 별도 스레드로
        tasks = {
            "knowledge_update": self._task_knowledge_update,
            "self_diagnosis": self._task_self_diagnosis,
            "trend_monitor": self._task_trend_monitor,
            "anomaly_check": self._task_anomaly_check,
            "skill_discovery": self._task_skill_discovery,
            "memory_consolidate": self._task_memory_consolidate,
        }

        for name, func in tasks.items():
            t = threading.Thread(
                target=self._run_scheduled,
                args=(name, func, self.SCHEDULES[name]),
                daemon=True,
                name=f"JARVIS-Auto-{name}",
            )
            self._threads.append(t)
            t.start()

        logger.info(f"Autonomous Loop started — {len(tasks)} background tasks active")
        self._emit(AutonomousEvent(
            type=AutonomousEventType.LEARNING_CYCLE,
            title="자율 루프 시작",
            content="JARVIS 자율 지능 루프가 활성화되었습니다. 백그라운드에서 지식 업데이트, 트렌드 모니터링, 자가 진단을 수행합니다.",
            priority=3,
        ))

    def stop(self):
        """자율 루프 중지"""
        self.is_running = False
        logger.info("Autonomous Loop stopped")

    # ── 스케줄러 ──────────────────────────────────────────────────────────
    def _run_scheduled(self, name: str, func: Callable, interval: int):
        """주기적으로 작업 실행"""
        # 첫 실행은 약간 지연 (시스템 안정화)
        time.sleep(30)
        while self.is_running:
            try:
                logger.info(f"[AutoLoop] Running: {name}")
                func()
                self._last_run[name] = time.time()
            except Exception as e:
                logger.error(f"[AutoLoop] {name} error: {e}")
            time.sleep(interval)

    # ── 지식 업데이트 ─────────────────────────────────────────────────────
    def _task_knowledge_update(self):
        """최신 ArXiv 논문 + GitHub 트렌드 수집 및 기억 저장"""
        updates = []

        # ArXiv 최신 논문
        try:
            papers = self.web.search_arxiv("large language model agent autonomous 2025", max_results=3)
            for p in papers:
                if isinstance(p, dict) and p.get("title"):
                    content = f"논문: {p['title']}\n저자: {', '.join(p.get('authors', [])[:3])}\n요약: {p.get('abstract', '')[:500]}"
                    self.memory.add_knowledge("arxiv_paper", content, p["title"])
                    updates.append(f"📄 {p['title'][:60]}")
            self._stats["knowledge_updates"] += len(papers)
        except Exception as e:
            logger.debug(f"ArXiv update error: {e}")

        # GitHub 트렌딩
        try:
            repos = self.web.search_github("AI agent LLM 2025", max_results=3)
            for r in repos:
                if isinstance(r, dict) and r.get("name"):
                    content = f"GitHub: {r['name']}\n설명: {r.get('description', '')}\n스타: {r.get('stars', 0)}"
                    self.memory.add_knowledge("github_trend", content, r["name"])
                    updates.append(f"⭐ {r['name']}")
        except Exception as e:
            logger.debug(f"GitHub update error: {e}")

        if updates:
            self._emit(AutonomousEvent(
                type=AutonomousEventType.KNOWLEDGE_UPDATE,
                title="지식 베이스 업데이트",
                content=f"새로운 정보 {len(updates)}개 저장:\n" + "\n".join(updates[:5]),
                priority=7,
                data={"updates": updates},
            ))

    # ── 자가 진단 ─────────────────────────────────────────────────────────
    def _task_self_diagnosis(self):
        """JARVIS 시스템 자가 진단"""
        import psutil
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        issues = []
        suggestions = []

        if cpu > 90:
            issues.append(f"CPU 과부하: {cpu:.1f}%")
            suggestions.append("CPU 사용량이 높습니다. 실행 중인 프로세스를 확인하세요.")
        if mem > 85:
            issues.append(f"메모리 부족: {mem:.1f}%")
            suggestions.append("메모리 사용량이 높습니다. 불필요한 프로세스를 종료하세요.")
        if disk > 90:
            issues.append(f"디스크 공간 부족: {disk:.1f}%")
            suggestions.append("디스크 공간이 부족합니다. 불필요한 파일을 정리하세요.")

        # 메모리 통계 확인
        try:
            mem_stats = self.memory.get_stats()
            if mem_stats.get("total_messages", 0) > 5000:
                suggestions.append("대화 기록이 많습니다. /memory 명령으로 정리를 권장합니다.")
        except Exception:
            pass

        status = "정상" if not issues else f"주의 필요: {len(issues)}개 문제"
        priority = 5 if not issues else 2

        self._emit(AutonomousEvent(
            type=AutonomousEventType.SELF_DIAGNOSIS,
            title=f"시스템 자가 진단 — {status}",
            content="\n".join([
                f"CPU: {cpu:.1f}% | 메모리: {mem:.1f}% | 디스크: {disk:.1f}%",
                *([f"⚠ {i}" for i in issues] or ["✅ 모든 시스템 정상"]),
                *([f"💡 {s}" for s in suggestions[:3]]),
            ]),
            priority=priority,
            data={"cpu": cpu, "mem": mem, "disk": disk, "issues": issues},
        ))

    # ── 트렌드 모니터 ─────────────────────────────────────────────────────
    def _task_trend_monitor(self):
        """AI/기술 최신 트렌드 모니터링"""
        try:
            news = self.web.search_news("AI LLM agent breakthrough 2025", max_results=3)
            if news and isinstance(news[0], dict) and not news[0].get("error"):
                top = news[0]
                self._emit(AutonomousEvent(
                    type=AutonomousEventType.TREND_ALERT,
                    title="AI 트렌드 알림",
                    content=f"📰 {top.get('title', '')}\n{top.get('snippet', '')[:300]}\n출처: {top.get('source', '')}",
                    priority=6,
                    data={"news": news[:3]},
                ))
        except Exception as e:
            logger.debug(f"Trend monitor error: {e}")

    # ── 스킬 자동 발견 ────────────────────────────────────────────────────
    def _task_skill_discovery(self):
        """자주 요청되는 패턴에서 새 스킬 자동 생성"""
        if not self.skills:
            return
        try:
            # 최근 대화에서 반복 패턴 분석
            history = self.memory.get_conversation_history(limit=50)
            if not history:
                return

            texts = [h.get("content", "") for h in history if h.get("role") == "user"]
            combined = " ".join(texts[:20])

            # 반복되는 계산/변환 작업 감지
            import re
            if re.search(r'(변환|계산|파싱|추출|포맷)', combined):
                # LLM 없이도 기본 스킬로 처리 가능한지 확인
                discovered = []
                keyword_map = {
                    "날짜": ("날짜 형식 변환", "datetime"),
                    "json": ("JSON 파싱 및 포맷팅", "parsing"),
                    "base64": ("Base64 인코딩/디코딩", "encoding"),
                    "hash": ("해시 계산 (MD5/SHA)", "security"),
                }
                for kw, (desc, cat) in keyword_map.items():
                    if kw in combined.lower() and not self.skills.search_skills(kw):
                        discovered.append((desc, cat))

                if discovered and self.skills.llm:
                    desc, cat = discovered[0]
                    result = self.skills.create_skill(desc, category=cat)
                    if result.get("success"):
                        self._stats["skills_discovered"] += 1
                        self._emit(AutonomousEvent(
                            type=AutonomousEventType.LEARNING_CYCLE,
                            title="새 스킬 자동 생성",
                            content=f"💡 사용 패턴 분석으로 새 스킬 생성: **{result.get('skill_name')}**\n{desc}",
                            priority=6,
                            data={"skill": result.get("skill_name")},
                        ))
        except Exception as e:
            logger.debug(f"Skill discovery error: {e}")

    # ── 이상 탐지 ─────────────────────────────────────────────────────────
    def _task_anomaly_check(self):
        """시스템 이상 패턴 탐지"""
        try:
            import psutil
            # 비정상적인 CPU 스파이크 감지
            cpu_samples = [psutil.cpu_percent(interval=0.1) for _ in range(3)]
            avg_cpu = sum(cpu_samples) / len(cpu_samples)

            if avg_cpu > 95:
                # 고CPU 프로세스 찾기
                procs = sorted(
                    psutil.process_iter(["pid", "name", "cpu_percent"]),
                    key=lambda p: p.info.get("cpu_percent", 0),
                    reverse=True,
                )[:3]
                proc_info = ", ".join([f"{p.info['name']}({p.info['cpu_percent']:.0f}%)" for p in procs[:3]])
                self._stats["anomalies_detected"] += 1
                self._emit(AutonomousEvent(
                    type=AutonomousEventType.ANOMALY_DETECTED,
                    title="CPU 이상 탐지",
                    content=f"⚠ CPU 사용률이 비정상적으로 높습니다: {avg_cpu:.1f}%\n원인 프로세스: {proc_info}",
                    priority=1,
                    data={"cpu": avg_cpu},
                ))
        except Exception as e:
            logger.debug(f"Anomaly check error: {e}")

    # ── 메모리 통합 ───────────────────────────────────────────────────────
    def _task_memory_consolidate(self):
        """오래된 메모리를 요약하고 압축"""
        try:
            stats = self.memory.get_stats()
            total = stats.get("total_messages", 0)

            if total > 200:
                # 오래된 대화 요약하여 지식으로 저장
                history = self.memory.get_conversation_history(limit=100)
                if history and len(history) > 50:
                    # 최근 50개 이전은 요약
                    old_history = history[:-50]
                    summary_content = f"대화 {len(old_history)}턴 요약 (자동 통합)\n"
                    topics = set()
                    for h in old_history:
                        content = h.get("content", "")[:100]
                        words = content.split()[:5]
                        if words:
                            topics.add(" ".join(words))
                    summary_content += "주요 주제: " + ", ".join(list(topics)[:5])
                    self.memory.add_knowledge("memory_consolidation", summary_content, "자동 메모리 통합")
                    self._emit(AutonomousEvent(
                        type=AutonomousEventType.LEARNING_CYCLE,
                        title="메모리 통합 완료",
                        content=f"🧠 {len(old_history)}개 대화 압축 완료. 지식 베이스에 저장.",
                        priority=8,
                    ))
        except Exception as e:
            logger.debug(f"Memory consolidate error: {e}")

    # ── 이벤트 관리 ───────────────────────────────────────────────────────
    def _emit(self, event: AutonomousEvent):
        """이벤트 발행"""
        self._event_log.append(event)
        if len(self._event_log) > 100:
            self._event_log = self._event_log[-100:]
        self._stats["total_events"] += 1

        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_recent_events(self, n: int = 20, event_type: str = None) -> List[Dict]:
        events = list(reversed(self._event_log[-n:]))
        if event_type:
            events = [e for e in events if e.type.value == event_type]
        return [
            {
                "type": e.type.value,
                "title": e.title,
                "content": e.content,
                "priority": e.priority,
                "timestamp": e.timestamp,
                "data": e.data,
            }
            for e in events
        ]

    def get_status(self) -> Dict:
        return {
            "is_running": self.is_running,
            "stats": self._stats,
            "last_runs": {
                k: datetime.fromtimestamp(v).isoformat() if v else None
                for k, v in self._last_run.items()
            },
            "schedules": self.SCHEDULES,
            "recent_events": self.get_recent_events(5),
        }

    def trigger_now(self, task_name: str) -> Dict:
        """특정 작업 즉시 실행"""
        task_map = {
            "knowledge_update": self._task_knowledge_update,
            "self_diagnosis": self._task_self_diagnosis,
            "trend_monitor": self._task_trend_monitor,
            "skill_discovery": self._task_skill_discovery,
            "anomaly_check": self._task_anomaly_check,
            "memory_consolidate": self._task_memory_consolidate,
        }
        func = task_map.get(task_name)
        if not func:
            return {"error": f"Unknown task: {task_name}"}
        try:
            func()
            return {"success": True, "task": task_name, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"error": str(e)}
