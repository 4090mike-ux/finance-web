"""
JARVIS 자율 프로그래머 — Iteration 10
JARVIS가 스스로 새 Python 모듈을 작성하고 테스트하고 배포한다

영감:
  - 자기 증식 알고리즘 (Gödel Machine)
  - 자율 코드 생성 (AutoGPT, Devin)
  - 역량 격차 탐지 (Capability Gap Analysis)
  - 진화적 알고리즘 (Evolutionary Algorithms)

핵심 개념:
  JARVIS는 자신의 역량 격차를 식별하고 → 새 코드를 작성하고
  → 샌드박스에서 테스트하고 → 통과 시 통합한다
  실패 시 최대 3번 재시도 (iterate_until_passing)
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
import importlib
import subprocess
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class CapabilityGap:
    """식별된 역량 격차 — JARVIS가 수행할 수 없는 것"""
    name: str                           # 격차 이름 (예: "image_resizer")
    description: str                    # 상세 설명
    priority: float = 0.5              # 우선순위 (0~1)
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    detected_at: float = field(default_factory=time.time)
    task_context: str = ""              # 어떤 작업에서 발견되었는가

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TestResult:
    """모듈 테스트 결과"""
    passed: bool
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    error_type: str = ""
    test_count: int = 0
    pass_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GeneratedModule:
    """자동 생성된 Python 모듈"""
    module_id: str
    gap_name: str
    module_name: str                    # Python 모듈 이름 (예: "image_resizer")
    code: str                           # 생성된 Python 코드
    tests: str                          # 생성된 테스트 코드
    docs: str                           # 생성된 문서
    file_path: str = ""                 # 저장된 파일 경로
    integrated: bool = False            # importlib으로 통합되었는가
    test_result: Optional[TestResult] = None
    attempts: int = 1                   # 생성 시도 횟수
    created_at: float = field(default_factory=time.time)
    version: str = "1.0.0"

    def to_dict(self) -> Dict:
        d = {
            "module_id": self.module_id,
            "gap_name": self.gap_name,
            "module_name": self.module_name,
            "code": self.code[:500] + "..." if len(self.code) > 500 else self.code,
            "docs": self.docs,
            "file_path": self.file_path,
            "integrated": self.integrated,
            "test_result": self.test_result.to_dict() if self.test_result else None,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "version": self.version,
        }
        return d


# ════════════════════════════════════════════════════════════════
# 자율 프로그래머
# ════════════════════════════════════════════════════════════════

class AutonomousProgrammer:
    """
    JARVIS 자율 프로그래머

    역량 격차를 탐지하고, LLM으로 코드를 생성하고,
    샌드박스 subprocess에서 테스트하고,
    importlib으로 런타임 통합한다.
    """

    # 생성된 모듈을 저장할 디렉토리
    GENERATED_DIR = Path(__file__).parent.parent / "generated"

    def __init__(
        self,
        llm_manager=None,
        event_callback: Optional[Callable] = None,
        db_path: str = "",
    ):
        """
        Args:
            llm_manager: LLMManager 인스턴스 (없으면 LLM 없이 동작)
            event_callback: 이벤트 발생 시 호출되는 콜백 함수
            db_path: SQLite DB 경로 (기본값: data/autonomous_programmer.db)
        """
        self.llm = llm_manager
        self.event_callback = event_callback
        self._lock = threading.Lock()

        # 통계
        self._modules_created: List[GeneratedModule] = []
        self._stats = {
            "modules_created": 0,
            "modules_integrated": 0,
            "modules_failed": 0,
            "total_attempts": 0,
        }

        # generated 디렉토리 생성
        self.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        # __init__.py 생성 (패키지로 인식)
        init_file = self.GENERATED_DIR / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""JARVIS 자동 생성 모듈 패키지"""\n')

        # DB 초기화
        if not db_path:
            data_dir = Path(__file__).parent.parent.parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "autonomous_programmer.db")
        self.db_path = db_path
        self._init_db()

        # 백그라운드 격차 분석 (30분 간격)
        self._bg_thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(f"AutonomousProgrammer 초기화 완료 — generated dir: {self.GENERATED_DIR}")

    # ── DB 초기화 ──────────────────────────────────────────────

    def _init_db(self):
        """SQLite 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generated_modules (
                    module_id TEXT PRIMARY KEY,
                    gap_name TEXT,
                    module_name TEXT,
                    code TEXT,
                    tests TEXT,
                    docs TEXT,
                    file_path TEXT,
                    integrated INTEGER DEFAULT 0,
                    test_passed INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 1,
                    created_at REAL,
                    version TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS capability_gaps (
                    gap_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    priority REAL,
                    detected_at REAL,
                    resolved INTEGER DEFAULT 0
                )
            """)
            conn.commit()

        # 기존 모듈 로드
        self._load_existing_modules()

    def _load_existing_modules(self):
        """DB에서 기존 생성 모듈 로드"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT module_id, gap_name, module_name, code, tests, docs, "
                    "file_path, integrated, test_passed, attempts, created_at, version "
                    "FROM generated_modules ORDER BY created_at DESC"
                ).fetchall()

            for row in rows:
                m = GeneratedModule(
                    module_id=row[0], gap_name=row[1], module_name=row[2],
                    code=row[3], tests=row[4], docs=row[5],
                    file_path=row[6], integrated=bool(row[7]),
                    test_result=TestResult(passed=bool(row[8])),
                    attempts=row[9], created_at=row[10], version=row[11] or "1.0.0",
                )
                self._modules_created.append(m)

            # 통계 업데이트
            self._stats["modules_created"] = len(self._modules_created)
            self._stats["modules_integrated"] = sum(1 for m in self._modules_created if m.integrated)
        except Exception as e:
            logger.warning(f"기존 모듈 로드 실패: {e}")

    # ── 역량 격차 식별 ──────────────────────────────────────────

    def identify_gaps(self, task_history: List[str]) -> List[CapabilityGap]:
        """
        대화 기록을 분석하여 역량 격차를 식별한다

        Args:
            task_history: 최근 작업/대화 목록

        Returns:
            식별된 CapabilityGap 목록
        """
        if not task_history:
            return []

        gaps = []

        # LLM 기반 격차 분석
        if self.llm:
            try:
                history_text = "\n".join(f"- {t}" for t in task_history[-20:])
                prompt = f"""다음 JARVIS 작업 기록을 분석하여 현재 시스템에 없는 유용한 Python 모듈/기능을 최대 3개 제안하라.

작업 기록:
{history_text}

각 제안에 대해 JSON 배열로 응답하라:
[
  {{"name": "모듈명_snake_case", "description": "기능 설명 (1-2문장)", "priority": 0.0-1.0}},
  ...
]

중요: 실제로 없는 기능만, 구현 가능한 것만 제안."""

                resp = self.llm.complete(prompt, max_tokens=500)
                import re
                json_match = re.search(r'\[.*?\]', resp, re.DOTALL)
                if json_match:
                    gap_dicts = json.loads(json_match.group())
                    for g in gap_dicts[:3]:
                        gap = CapabilityGap(
                            name=g.get("name", "unknown_module"),
                            description=g.get("description", ""),
                            priority=float(g.get("priority", 0.5)),
                        )
                        gaps.append(gap)
                        logger.info(f"역량 격차 발견: {gap.name} (우선순위: {gap.priority:.2f})")
            except Exception as e:
                logger.warning(f"LLM 기반 격차 분석 실패: {e}")

        # LLM 없이 간단한 휴리스틱 기반 격차 감지
        if not gaps:
            keywords_gaps = {
                "image": CapabilityGap("image_processor", "이미지 처리 유틸리티 (리사이즈, 필터)", 0.7),
                "chart": CapabilityGap("chart_generator", "간단한 ASCII 차트 생성기", 0.6),
                "csv": CapabilityGap("csv_analyzer", "CSV 파일 통계 분석기", 0.65),
                "email": CapabilityGap("email_formatter", "이메일 텍스트 포맷터", 0.5),
                "json": CapabilityGap("json_validator", "JSON 스키마 검증기", 0.6),
            }
            combined = " ".join(task_history).lower()
            for kw, gap in keywords_gaps.items():
                if kw in combined:
                    gaps.append(gap)
                    break

        return gaps

    # ── 모듈 작성 ──────────────────────────────────────────────

    def write_module(self, gap: CapabilityGap) -> GeneratedModule:
        """
        역량 격차를 해결하는 Python 모듈을 LLM으로 생성한다

        Args:
            gap: 해결할 역량 격차

        Returns:
            생성된 모듈 (코드 + 테스트 + 문서)
        """
        module_id = str(uuid.uuid4())[:12]
        module_name = gap.name.replace("-", "_").replace(" ", "_").lower()

        logger.info(f"모듈 작성 시작: {module_name}")
        self._emit_event("writing_started", {"gap_name": gap.name, "module_name": module_name})

        code = self._generate_code(gap, module_name)
        tests = self._generate_tests(gap, module_name, code)
        docs = self._generate_docs(gap, module_name)

        module = GeneratedModule(
            module_id=module_id,
            gap_name=gap.name,
            module_name=module_name,
            code=code,
            tests=tests,
            docs=docs,
        )

        logger.info(f"모듈 작성 완료: {module_name} ({len(code)} 문자)")
        return module

    def _generate_code(self, gap: CapabilityGap, module_name: str) -> str:
        """LLM으로 실제 Python 코드 생성"""
        if self.llm:
            try:
                prompt = f"""다음 요구사항을 충족하는 완전한 Python 모듈을 작성하라.

모듈 이름: {module_name}
요구사항: {gap.description}

규칙:
1. 파일 시작에 docstring 필수
2. 외부 pip 패키지 사용 금지 (표준 라이브러리만)
3. 모든 함수에 타입 힌트 및 docstring
4. 오류 처리 필수
5. 클래스 기반 설계

Python 코드만 출력하라 (마크다운 없이):"""

                code = self.llm.complete(prompt, max_tokens=1500)
                # 마크다운 코드 블록 제거
                code = code.strip()
                if code.startswith("```python"):
                    code = code[9:]
                if code.startswith("```"):
                    code = code[3:]
                if code.endswith("```"):
                    code = code[:-3]
                return code.strip()
            except Exception as e:
                logger.warning(f"LLM 코드 생성 실패: {e}")

        # 폴백: 템플릿 기반 코드 생성
        return self._template_code(module_name, gap.description)

    def _template_code(self, module_name: str, description: str) -> str:
        """LLM 없을 때 사용하는 기본 템플릿 코드"""
        class_name = "".join(w.capitalize() for w in module_name.split("_"))
        return f'''"""
{module_name} — JARVIS 자동 생성 모듈
{description}
자동 생성: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class {class_name}:
    """
    {description}
    JARVIS 자율 프로그래머에 의해 자동 생성됨.
    """

    def __init__(self):
        """초기화"""
        self.version = "1.0.0"
        self.created_at = time.time()
        logger.info(f"{class_name} 초기화 완료")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        주요 처리 메서드

        Args:
            input_data: 처리할 입력 데이터

        Returns:
            처리 결과 딕셔너리
        """
        try:
            result = str(input_data).strip()
            return {{
                "success": True,
                "result": result,
                "length": len(result),
                "timestamp": time.time(),
            }}
        except Exception as e:
            logger.error(f"처리 실패: {{e}}")
            return {{"success": False, "error": str(e)}}

    def get_info(self) -> Dict[str, Any]:
        """모듈 정보 반환"""
        return {{
            "name": "{module_name}",
            "version": self.version,
            "description": "{description}",
            "created_at": self.created_at,
        }}


# 편의 함수
def process_{module_name}(data: Any) -> Dict[str, Any]:
    """편의 함수: {class_name} 인스턴스 생성 없이 바로 처리"""
    return {class_name}().process(data)
'''

    def _generate_tests(self, gap: CapabilityGap, module_name: str, code: str) -> str:
        """LLM으로 테스트 코드 생성"""
        if self.llm:
            try:
                prompt = f"""다음 Python 모듈에 대한 간단한 테스트 코드를 작성하라.

모듈 이름: {module_name}
요구사항: {gap.description}

모듈 코드 (앞부분):
{code[:800]}

규칙:
1. 표준 라이브러리 unittest만 사용
2. 최소 3개 테스트 케이스
3. 모듈을 현재 디렉토리에서 임포트: import {module_name}
4. 실행 가능한 완전한 코드

Python 코드만 출력하라:"""

                tests = self.llm.complete(prompt, max_tokens=800)
                tests = tests.strip()
                for marker in ["```python", "```"]:
                    tests = tests.replace(marker, "")
                return tests.strip()
            except Exception as e:
                logger.warning(f"LLM 테스트 생성 실패: {e}")

        # 폴백 테스트
        class_name = "".join(w.capitalize() for w in module_name.split("_"))
        return f'''import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import {module_name}

class Test{class_name}(unittest.TestCase):
    def setUp(self):
        """테스트 설정"""
        self.instance = {module_name}.{class_name}()

    def test_instantiation(self):
        """인스턴스 생성 테스트"""
        self.assertIsNotNone(self.instance)

    def test_process_string(self):
        """문자열 처리 테스트"""
        result = self.instance.process("test input")
        self.assertIn("success", result)

    def test_get_info(self):
        """정보 조회 테스트"""
        info = self.instance.get_info()
        self.assertIn("name", info)
        self.assertEqual(info["name"], "{module_name}")

    def test_process_empty(self):
        """빈 입력 테스트"""
        result = self.instance.process("")
        self.assertIn("success", result)

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''

    def _generate_docs(self, gap: CapabilityGap, module_name: str) -> str:
        """모듈 문서 생성"""
        return f"""# {module_name}

**목적**: {gap.description}

**생성일**: {time.strftime("%Y-%m-%d %H:%M:%S")}

**우선순위**: {gap.priority:.2f}

## 사용법

```python
from jarvis.generated.{module_name} import {module_name.title().replace('_', '')}

instance = {module_name.title().replace('_', '')}()
result = instance.process(data)
```

## 자동 생성 정보

이 모듈은 JARVIS 자율 프로그래머(Iteration 10)에 의해 자동으로 생성되었습니다.
역량 격차 ID: {gap.gap_id}
"""

    # ── 테스트 ──────────────────────────────────────────────────

    def test_module(self, module: GeneratedModule) -> TestResult:
        """
        샌드박스 subprocess에서 모듈을 테스트한다

        Args:
            module: 테스트할 생성 모듈

        Returns:
            TestResult (passed, stdout, stderr, duration)
        """
        start = time.time()
        logger.info(f"테스트 시작: {module.module_name}")

        # 1. 모듈 파일 저장
        module_path = self.GENERATED_DIR / f"{module.module_name}.py"
        test_path = self.GENERATED_DIR / f"test_{module.module_name}.py"

        try:
            module_path.write_text(module.code, encoding="utf-8")
            test_path.write_text(module.tests, encoding="utf-8")
            module.file_path = str(module_path)
        except Exception as e:
            return TestResult(passed=False, stderr=f"파일 저장 실패: {e}", duration=time.time() - start)

        # 2. 먼저 구문 검사 (컴파일)
        try:
            compile(module.code, module_path, "exec")
        except SyntaxError as e:
            return TestResult(
                passed=False,
                stderr=f"구문 오류: {e}",
                duration=time.time() - start,
                error_type="SyntaxError",
            )

        # 3. subprocess로 테스트 실행 (샌드박스, timeout=10s)
        try:
            proc = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.GENERATED_DIR),
            )
            duration = time.time() - start
            passed = proc.returncode == 0

            result = TestResult(
                passed=passed,
                stdout=proc.stdout[:2000],
                stderr=proc.stderr[:2000],
                duration=duration,
                error_type="" if passed else "TestFailure",
            )

            if passed:
                logger.info(f"테스트 통과: {module.module_name} ({duration:.2f}s)")
            else:
                logger.warning(f"테스트 실패: {module.module_name}\n{proc.stderr[:500]}")

            return result

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                stderr="테스트 시간 초과 (10s)",
                duration=10.0,
                error_type="Timeout",
            )
        except Exception as e:
            return TestResult(
                passed=False,
                stderr=f"테스트 실행 오류: {e}",
                duration=time.time() - start,
                error_type=type(e).__name__,
            )

    # ── 통합 ──────────────────────────────────────────────────

    def integrate_module(self, module: GeneratedModule) -> bool:
        """
        통과한 모듈을 importlib으로 런타임에 통합한다

        Args:
            module: 통합할 모듈

        Returns:
            bool: 통합 성공 여부
        """
        if not module.file_path or not Path(module.file_path).exists():
            logger.warning(f"통합 실패: 파일 없음 {module.file_path}")
            return False

        try:
            spec = importlib.util.spec_from_file_location(
                f"jarvis.generated.{module.module_name}",
                module.file_path,
            )
            if spec is None:
                return False

            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"jarvis.generated.{module.module_name}"] = mod
            spec.loader.exec_module(mod)

            module.integrated = True
            logger.info(f"모듈 통합 성공: jarvis.generated.{module.module_name}")
            self._emit_event("module_integrated", {"module_name": module.module_name})
            return True

        except Exception as e:
            logger.error(f"모듈 통합 실패: {module.module_name} — {e}")
            return False

    # ── 반복 시도 ──────────────────────────────────────────────

    def iterate_until_passing(
        self,
        gap: CapabilityGap,
        max_attempts: int = 3,
    ) -> Optional[GeneratedModule]:
        """
        테스트 통과할 때까지 코드를 재생성한다 (최대 max_attempts회)

        Args:
            gap: 해결할 역량 격차
            max_attempts: 최대 재시도 횟수

        Returns:
            통과한 GeneratedModule, 실패 시 None
        """
        logger.info(f"자율 프로그래밍 시작: {gap.name} (최대 {max_attempts}회 시도)")
        self._emit_event("programming_started", {"gap": gap.to_dict()})
        self._stats["total_attempts"] += 1

        last_module = None
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            logger.info(f"시도 {attempt}/{max_attempts}: {gap.name}")

            # 이전 오류를 description에 포함 (재생성 시 개선)
            if attempt > 1 and last_error:
                gap = CapabilityGap(
                    name=gap.name,
                    description=f"{gap.description}\n\n이전 오류 수정 필요: {last_error[:300]}",
                    priority=gap.priority,
                    gap_id=gap.gap_id,
                )

            module = self.write_module(gap)
            module.attempts = attempt
            last_module = module

            # 테스트
            test_result = self.test_module(module)
            module.test_result = test_result

            if test_result.passed:
                # 통합
                self.integrate_module(module)
                self._save_module(module)

                with self._lock:
                    self._modules_created.append(module)
                    self._stats["modules_created"] += 1
                    if module.integrated:
                        self._stats["modules_integrated"] += 1

                self._emit_event("programming_success", {
                    "module": module.to_dict(),
                    "attempts": attempt,
                })
                logger.info(f"자율 프로그래밍 성공: {gap.name} (시도 {attempt}회)")
                return module
            else:
                last_error = test_result.stderr or test_result.stdout
                self._emit_event("test_failed", {
                    "attempt": attempt,
                    "error": last_error[:200],
                    "module_name": module.module_name,
                })

        # 모든 시도 실패
        logger.warning(f"자율 프로그래밍 실패: {gap.name} ({max_attempts}회 시도 후 포기)")
        self._stats["modules_failed"] += 1

        if last_module:
            self._save_module(last_module, failed=True)
            with self._lock:
                self._modules_created.append(last_module)
                self._stats["modules_created"] += 1

        self._emit_event("programming_failed", {
            "gap_name": gap.name,
            "attempts": max_attempts,
        })
        return None

    # ── 조회 ──────────────────────────────────────────────────

    def get_created_modules(self) -> Dict[str, Any]:
        """생성된 모든 모듈 목록 반환"""
        with self._lock:
            modules = [m.to_dict() for m in self._modules_created]

        return {
            "total": len(modules),
            "integrated": self._stats["modules_integrated"],
            "failed": self._stats["modules_failed"],
            "modules": modules,
            "stats": self._stats.copy(),
        }

    # ── 백그라운드 격차 분석 ──────────────────────────────────

    def start_background_analysis(self, interval_minutes: int = 30):
        """
        백그라운드에서 주기적으로 격차를 분석하고 모듈을 생성한다

        Args:
            interval_minutes: 분석 간격 (분)
        """
        if self._running:
            return

        self._running = True
        self._bg_thread = threading.Thread(
            target=self._background_loop,
            args=(interval_minutes * 60,),
            daemon=True,
            name="autonomous-programmer-bg",
        )
        self._bg_thread.start()
        logger.info(f"백그라운드 격차 분석 시작 (간격: {interval_minutes}분)")

    def _background_loop(self, interval_seconds: int):
        """백그라운드 분석 루프"""
        while self._running:
            try:
                time.sleep(interval_seconds)
                if not self._running:
                    break

                # 간단한 자동 격차 탐지 및 모듈 생성
                default_gaps = [
                    CapabilityGap(
                        "text_summarizer",
                        "텍스트를 3문장으로 요약하는 유틸리티",
                        priority=0.6,
                    ),
                ]
                for gap in default_gaps:
                    # 이미 생성된 모듈인지 확인
                    existing = [m for m in self._modules_created if m.gap_name == gap.name]
                    if not existing:
                        self.iterate_until_passing(gap)
                        break  # 한 번에 하나씩만

            except Exception as e:
                logger.warning(f"백그라운드 분석 오류: {e}")

    # ── 내부 유틸 ──────────────────────────────────────────────

    def _save_module(self, module: GeneratedModule, failed: bool = False):
        """모듈 정보를 DB에 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO generated_modules
                    (module_id, gap_name, module_name, code, tests, docs,
                     file_path, integrated, test_passed, attempts, created_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    module.module_id, module.gap_name, module.module_name,
                    module.code, module.tests, module.docs,
                    module.file_path, int(module.integrated),
                    int(module.test_result.passed if module.test_result else False),
                    module.attempts, module.created_at, module.version,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"모듈 DB 저장 실패: {e}")

    def _emit_event(self, event_type: str, data: Dict):
        """이벤트 콜백 호출"""
        if self.event_callback:
            try:
                self.event_callback({
                    "type": event_type,
                    "data": data,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
