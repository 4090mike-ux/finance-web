"""
JARVIS 강화학습 자율최적화 엔진 — Iteration 9
RL 기반 모듈 선택 최적화 (신경망 없는 표 기반 Q-학습)

원리:
  - 상태 공간: (task_type, complexity, context_richness, user_satisfaction)
  - 행동 공간: 어떤 추론 모듈을 활성화할지
  - 보상: 사용자 피드백 (명시적 평점 OR 암묵적 신호)
  - Q-테이블: SQLite에 상태해시 → 행동가치 저장
  - ε-탐욕: 탐색(exploration) vs 활용(exploitation)
  - UCB: Upper Confidence Bound for 모듈 선택
  - 경험 재플레이: 마지막 1000개 전환 저장, 10스텝마다 배치 업데이트
"""

import json
import math
import random
import sqlite3
import threading
import time
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 상수 정의
# ════════════════════════════════════════════════════════════════

# 행동 공간 — 활성화할 수 있는 추론 모듈
AVAILABLE_ACTIONS = [
    "direct",           # 직접 LLM 응답
    "tot",              # Tree of Thoughts
    "debate",           # 다중 에이전트 토론
    "hypothesis",       # 가설 생성
    "causal",           # 인과 추론
    "research",         # 심층 연구
    "simulation",       # 현실 시뮬레이션
    "knowledge_graph",  # 지식 그래프 활용
    "temporal",         # 시간 추론
    "swarm",            # 에이전트 스웜
]

# 작업 유형 분류
TASK_TYPES = ["question", "analysis", "coding", "creative", "research", "planning", "unknown"]

# 복잡도 레벨
COMPLEXITY_LEVELS = ["low", "medium", "high"]

# 만족도 레벨
SATISFACTION_LEVELS = ["low", "medium", "high"]


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class RLState:
    """RL 상태 표현"""
    task_type: str = "unknown"
    complexity: str = "medium"
    context_richness: str = "medium"   # 컨텍스트 풍부도
    user_satisfaction: str = "medium"  # 이전 상호작용 만족도

    def to_hash(self) -> str:
        """상태를 해시로 변환 (Q-테이블 키)"""
        state_str = f"{self.task_type}|{self.complexity}|{self.context_richness}|{self.user_satisfaction}"
        return hashlib.md5(state_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "context_richness": self.context_richness,
            "user_satisfaction": self.user_satisfaction,
        }


@dataclass
class Transition:
    """RL 전환 (s, a, r, s') — 경험 재플레이용"""
    state_hash: str
    action: str
    reward: float
    next_state_hash: str
    timestamp: float = field(default_factory=time.time)
    info: str = ""


# ════════════════════════════════════════════════════════════════
# Q-테이블 (SQLite 기반 영속성)
# ════════════════════════════════════════════════════════════════

class QTable:
    """SQLite 기반 Q-값 테이블"""

    def __init__(self, db_path: str, default_value: float = 0.0):
        self.db_path = db_path
        self.default_value = default_value
        self._cache: Dict[str, Dict[str, float]] = {}  # 인메모리 캐시
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS q_values (
                state_hash TEXT,
                action TEXT,
                q_value REAL DEFAULT 0.0,
                visit_count INTEGER DEFAULT 0,
                last_updated REAL,
                PRIMARY KEY (state_hash, action)
            )
        """)
        conn.commit()
        # 캐시 로드
        rows = conn.execute("SELECT state_hash, action, q_value, visit_count FROM q_values").fetchall()
        conn.close()
        for state_hash, action, q_value, visits in rows:
            if state_hash not in self._cache:
                self._cache[state_hash] = {}
            self._cache[state_hash][action] = q_value

    def get(self, state_hash: str, action: str) -> float:
        with self._lock:
            return self._cache.get(state_hash, {}).get(action, self.default_value)

    def get_all_actions(self, state_hash: str) -> Dict[str, float]:
        with self._lock:
            return dict(self._cache.get(state_hash, {action: self.default_value for action in AVAILABLE_ACTIONS}))

    def update(self, state_hash: str, action: str, new_value: float):
        with self._lock:
            if state_hash not in self._cache:
                self._cache[state_hash] = {}
            self._cache[state_hash][action] = new_value

            # DB 비동기 저장 (성능 최적화)
            threading.Thread(target=self._db_update, args=(state_hash, action, new_value), daemon=True).start()

    def _db_update(self, state_hash: str, action: str, q_value: float):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO q_values (state_hash, action, q_value, visit_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(state_hash, action) DO UPDATE SET
                    q_value=excluded.q_value,
                    visit_count=visit_count+1,
                    last_updated=excluded.last_updated
            """, (state_hash, action, q_value, time.time()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Q-table DB update failed: {e}")

    def get_visit_count(self, state_hash: str, action: str) -> int:
        """UCB에 사용할 방문 횟수"""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT visit_count FROM q_values WHERE state_hash=? AND action=?",
                (state_hash, action)
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0


# ════════════════════════════════════════════════════════════════
# 메인 RL 옵티마이저
# ════════════════════════════════════════════════════════════════

class RLOptimizer:
    """
    JARVIS 강화학습 자율최적화 엔진 — Iteration 9

    기능:
    - ε-탐욕 정책으로 모듈 선택
    - UCB (Upper Confidence Bound) 탐색 전략
    - 경험 재플레이 (Experience Replay)
    - 배치 Q-학습 업데이트
    - 작업 유형별 최적 전략 추적
    """

    # 하이퍼파라미터
    ALPHA = 0.1          # 학습률
    GAMMA = 0.9          # 할인 인수
    EPSILON_START = 0.3  # 초기 탐색률
    EPSILON_MIN = 0.05   # 최소 탐색률
    EPSILON_DECAY = 0.995
    UCB_CONFIDENCE = 1.5  # UCB 탐색 상수
    REPLAY_BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    UPDATE_EVERY = 10    # 몇 스텝마다 배치 업데이트

    def __init__(self, llm_manager=None, db_path: Optional[str] = None):
        self.llm = llm_manager
        self._lock = threading.RLock()

        # DB 경로
        if db_path is None:
            from pathlib import Path
            db_dir = Path(__file__).parent.parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "rl_optimizer.db")
        self.db_path = db_path

        # Q-테이블
        self.q_table = QTable(db_path)

        # 경험 재플레이 버퍼
        self.replay_buffer: deque = deque(maxlen=self.REPLAY_BUFFER_SIZE)

        # 상태 추적
        self._current_state: Optional[RLState] = None
        self._last_action: Optional[str] = None
        self._current_state_hash: Optional[str] = None

        # 통계
        self.total_steps = 0
        self.total_reward = 0.0
        self.epsilon = self.EPSILON_START
        self.reward_history: deque = deque(maxlen=200)

        # 상태별 행동 빈도 추적
        self.action_counts: Dict[str, int] = {a: 0 for a in AVAILABLE_ACTIONS}

        # 작업 유형별 최적 전략
        self.best_strategy_per_type: Dict[str, str] = {}

        self._init_stats_db()
        logger.info(f"RLOptimizer initialized — ε={self.epsilon:.2f}, actions={len(AVAILABLE_ACTIONS)}")

    def _init_stats_db(self):
        """통계 테이블 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    state_hash TEXT,
                    action TEXT,
                    reward REAL,
                    next_state_hash TEXT,
                    info TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rl_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # 이전 통계 복원
            rows = conn.execute("SELECT key, value FROM rl_stats").fetchall()
            conn.commit()
            conn.close()

            for key, value in rows:
                if key == "total_steps":
                    self.total_steps = int(float(value))
                elif key == "total_reward":
                    self.total_reward = float(value)
                elif key == "epsilon":
                    self.epsilon = max(self.EPSILON_MIN, float(value))
        except Exception as e:
            logger.warning(f"RL stats DB init failed: {e}")

    def classify_state(self, user_input: str, context: Dict = None) -> RLState:
        """사용자 입력을 RL 상태로 분류"""
        context = context or {}
        text_lower = user_input.lower()

        # 작업 유형 분류
        if any(w in text_lower for w in ["왜", "어떻게", "what", "why", "how", "설명"]):
            task_type = "question"
        elif any(w in text_lower for w in ["코드", "code", "프로그램", "함수", "function"]):
            task_type = "coding"
        elif any(w in text_lower for w in ["분석", "analyze", "evaluate", "평가"]):
            task_type = "analysis"
        elif any(w in text_lower for w in ["계획", "plan", "전략", "strategy"]):
            task_type = "planning"
        elif any(w in text_lower for w in ["연구", "research", "조사", "탐구"]):
            task_type = "research"
        elif any(w in text_lower for w in ["써", "작성", "create", "만들어", "generate"]):
            task_type = "creative"
        else:
            task_type = "unknown"

        # 복잡도 추정
        word_count = len(user_input.split())
        if word_count < 10:
            complexity = "low"
        elif word_count < 30:
            complexity = "medium"
        else:
            complexity = "high"

        # 컨텍스트 풍부도
        context_size = len(str(context))
        if context_size < 100:
            context_richness = "low"
        elif context_size < 500:
            context_richness = "medium"
        else:
            context_richness = "high"

        return RLState(
            task_type=task_type,
            complexity=complexity,
            context_richness=context_richness,
            user_satisfaction=context.get("last_satisfaction", "medium"),
        )

    def select_action(self, state: RLState) -> str:
        """
        ε-탐욕 + UCB 정책으로 행동(모듈) 선택

        탐색 단계: 무작위 행동
        활용 단계: UCB로 최적 행동
        """
        with self._lock:
            self._current_state = state
            self._current_state_hash = state.to_hash()

            # ε-탐욕 탐색
            if random.random() < self.epsilon:
                action = random.choice(AVAILABLE_ACTIONS)
                logger.debug(f"RL 탐색: {action} (ε={self.epsilon:.3f})")
            else:
                # UCB 기반 행동 선택
                action = self._ucb_select(self._current_state_hash)
                logger.debug(f"RL 활용: {action} (UCB)")

            self._last_action = action
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            return action

    def _ucb_select(self, state_hash: str) -> str:
        """UCB (Upper Confidence Bound) 행동 선택"""
        q_values = self.q_table.get_all_actions(state_hash)
        total_visits = sum(self.action_counts.values()) + 1

        best_action = None
        best_ucb = float('-inf')

        for action in AVAILABLE_ACTIONS:
            q = q_values.get(action, 0.0)
            n = max(self.q_table.get_visit_count(state_hash, action), 1)
            # UCB 공식: Q(s,a) + C * sqrt(ln(N) / n(s,a))
            ucb = q + self.UCB_CONFIDENCE * math.sqrt(math.log(total_visits) / n)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action

        return best_action or "direct"

    def observe_reward(self, reward: float, info: str = "") -> None:
        """
        보상 관찰 및 Q-값 업데이트

        reward: -1.0 ~ +1.0
        - +1.0: 매우 좋은 응답 (명시적 긍정 피드백)
        - +0.5: 암묵적 긍정 (후속 질문)
        - 0.0: 중립
        - -0.5: 암묵적 부정 (짧은 반응)
        - -1.0: 명시적 불만
        """
        if self._current_state_hash is None or self._last_action is None:
            return

        with self._lock:
            # 전환 저장 (경험 재플레이)
            transition = Transition(
                state_hash=self._current_state_hash,
                action=self._last_action,
                reward=reward,
                next_state_hash=self._current_state_hash,  # 단순화: 같은 상태
                info=info[:200],
            )
            self.replay_buffer.append(transition)

            # 즉시 Q-값 업데이트 (단일 전환)
            self._update_q(transition)

            # 통계 업데이트
            self.total_steps += 1
            self.total_reward += reward
            self.reward_history.append(reward)

            # ε 감쇠
            self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)

            # 배치 업데이트 (매 N스텝)
            if self.total_steps % self.UPDATE_EVERY == 0:
                self._batch_update()

            # 작업 유형별 최적 전략 갱신
            if self._current_state and reward > 0:
                task_type = self._current_state.task_type
                self.best_strategy_per_type[task_type] = self._last_action

            # 주기적 통계 저장
            if self.total_steps % 50 == 0:
                self._save_stats()

    def _update_q(self, transition: Transition):
        """단일 전환에 대한 Q-값 업데이트"""
        old_q = self.q_table.get(transition.state_hash, transition.action)
        # Q-학습 업데이트: Q(s,a) += α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        next_q_values = self.q_table.get_all_actions(transition.next_state_hash)
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        new_q = old_q + self.ALPHA * (
            transition.reward + self.GAMMA * max_next_q - old_q
        )
        self.q_table.update(transition.state_hash, transition.action, new_q)

    def _batch_update(self):
        """경험 재플레이 배치 업데이트"""
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        batch = random.sample(list(self.replay_buffer), min(self.BATCH_SIZE, len(self.replay_buffer)))
        for transition in batch:
            self._update_q(transition)

    def _save_stats(self):
        """통계를 DB에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            stats = [
                ("total_steps", str(self.total_steps)),
                ("total_reward", str(self.total_reward)),
                ("epsilon", str(self.epsilon)),
            ]
            for key, value in stats:
                conn.execute(
                    "INSERT OR REPLACE INTO rl_stats (key, value) VALUES (?,?)",
                    (key, value)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"RL stats save failed: {e}")

    def infer_reward_from_response(self, user_input: str, response_length: int, has_error: bool) -> float:
        """
        암묵적 보상 추론 (명시적 피드백 없을 때)

        - 짧은 에러 응답 → 부정적
        - 긴 상세 응답 → 약간 긍정적
        """
        if has_error:
            return -0.5
        if response_length > 500:
            return 0.3
        if response_length > 200:
            return 0.1
        return 0.0

    def get_best_strategy(self, task_type: str) -> Dict:
        """특정 작업 유형의 최적 전략 조회"""
        best_action = self.best_strategy_per_type.get(task_type, "direct")

        # Q-값 기반으로도 확인
        dummy_state = RLState(task_type=task_type, complexity="medium")
        state_hash = dummy_state.to_hash()
        q_values = self.q_table.get_all_actions(state_hash)

        if q_values:
            q_best = max(q_values, key=q_values.get)
            if q_values[q_best] > 0:
                best_action = q_best

        return {
            "task_type": task_type,
            "recommended_action": best_action,
            "q_values": {k: round(v, 3) for k, v in sorted(q_values.items(), key=lambda x: -x[1])[:5]},
        }

    def get_stats(self) -> Dict:
        """RL 옵티마이저 통계"""
        with self._lock:
            avg_reward = (
                sum(self.reward_history) / len(self.reward_history)
                if self.reward_history else 0.0
            )
            return {
                "total_steps": self.total_steps,
                "total_reward": round(self.total_reward, 2),
                "avg_reward": round(avg_reward, 3),
                "exploration_rate": round(self.epsilon, 4),
                "replay_buffer_size": len(self.replay_buffer),
                "action_distribution": dict(self.action_counts),
                "best_strategies": dict(self.best_strategy_per_type),
                "available_actions": AVAILABLE_ACTIONS,
            }
