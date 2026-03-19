"""
JARVIS 코드 인텔리전스 에이전트 — Iteration 3
소프트웨어를 인간 시니어 개발자 수준으로 이해/작성/디버그/최적화하는 에이전트
- 코드 자동 생성 (요구사항 → 완전한 구현)
- 버그 자동 감지 및 수정
- 코드 최적화 및 리팩토링
- 단위 테스트 자동 생성
- 코드 문서화 자동화
- 다중 언어 지원 (Python, JavaScript, SQL, Shell)
- 실시간 코드 실행 및 결과 분석
"""

import ast
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CodeResult:
    language: str
    code: str
    explanation: str
    tests: str = ""
    documentation: str = ""
    complexity: str = "O(n)"
    warnings: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BugReport:
    severity: str  # critical, high, medium, low
    location: str
    description: str
    fix_suggestion: str
    fixed_code: Optional[str] = None


class CodeIntelligence:
    """
    JARVIS 코드 인텔리전스
    인간 시니어 개발자를 뛰어넘는 코드 이해와 생성 능력
    """

    LANGUAGE_CONFIGS = {
        "python": {"extension": ".py", "executor": "python"},
        "javascript": {"extension": ".js", "executor": "node"},
        "sql": {"extension": ".sql", "executor": None},
        "shell": {"extension": ".sh", "executor": "bash"},
        "typescript": {"extension": ".ts", "executor": "npx ts-node"},
        "rust": {"extension": ".rs", "executor": "rustc"},
    }

    def __init__(self, llm_manager, code_executor=None):
        self.llm = llm_manager
        self.executor = code_executor
        self._history: List[CodeResult] = []
        logger.info("CodeIntelligence initialized")

    # ── 코드 생성 ─────────────────────────────────────────────────────────
    def generate(
        self,
        requirement: str,
        language: str = "python",
        include_tests: bool = True,
        include_docs: bool = True,
        style: str = "production",  # prototype, production, minimal
    ) -> CodeResult:
        """자연어 요구사항으로 완전한 코드 생성"""
        from jarvis.llm.manager import Message

        style_desc = {
            "prototype": "빠른 프로토타입, 간단한 구조",
            "production": "프로덕션 품질, 에러 처리, 로깅, 타입 힌트 포함",
            "minimal": "최소한의 코드, 핵심 기능만",
        }

        system = f"""당신은 세계 최고 수준의 {language} 개발자입니다.
요구사항을 완벽하게 구현하는 코드를 작성합니다.
스타일: {style_desc.get(style, 'production')}
언어: {language}"""

        prompt = f"""다음 요구사항을 구현하세요:

{requirement}

다음 형식으로 응답하세요:

## 코드
```{language}
[구현 코드]
```

## 설명
[코드 동작 설명]

## 복잡도
시간: O(...), 공간: O(...)

## 주의사항
- [주의사항 목록]"""

        if include_tests:
            prompt += "\n\n## 테스트\n```python\n[단위 테스트 코드]\n```"
        if include_docs:
            prompt += "\n\n## 문서\n[함수/클래스 API 문서]"

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system=system, max_tokens=8192)
        result = self._parse_generation_response(response.content, language)
        self._history.append(result)
        return result

    def _parse_generation_response(self, text: str, language: str) -> CodeResult:
        """LLM 응답 파싱"""
        # 코드 블록 추출
        code_match = re.search(rf'```{language}\n(.*?)```', text, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else text

        # 테스트 추출
        test_match = re.search(r'```python\n(.*?test.*?)```', text, re.DOTALL | re.IGNORECASE)
        tests = test_match.group(1).strip() if test_match else ""

        # 설명 추출
        explanation_match = re.search(r'## 설명\n(.*?)(?=##|\Z)', text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # 복잡도 추출
        complexity_match = re.search(r'시간[:\s]*(O\([^)]+\))', text)
        complexity = complexity_match.group(1) if complexity_match else "O(n)"

        return CodeResult(
            language=language,
            code=code,
            explanation=explanation or "코드가 생성되었습니다.",
            tests=tests,
            complexity=complexity,
        )

    # ── 버그 감지 및 수정 ─────────────────────────────────────────────────
    def detect_bugs(self, code: str, language: str = "python") -> List[BugReport]:
        """코드에서 버그 자동 감지"""
        bugs = []

        # Python 정적 분석
        if language == "python":
            bugs.extend(self._python_static_analysis(code))

        # LLM 기반 심층 분석
        if self.llm:
            llm_bugs = self._llm_bug_detection(code, language)
            bugs.extend(llm_bugs)

        return bugs

    def _python_static_analysis(self, code: str) -> List[BugReport]:
        """Python AST 기반 정적 분석"""
        bugs = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [BugReport(
                severity="critical",
                location=f"Line {e.lineno}",
                description=f"문법 오류: {e.msg}",
                fix_suggestion="문법을 수정하세요.",
            )]

        # 미정의 변수 사용 패턴 감지
        for node in ast.walk(tree):
            # except 블록에서 bare except 감지
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bugs.append(BugReport(
                    severity="medium",
                    location="except 블록",
                    description="Bare except 사용 — 모든 예외를 잡습니다",
                    fix_suggestion="'except Exception as e:' 사용을 권장합니다",
                ))

            # 변경 가능한 기본 인수 감지
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        bugs.append(BugReport(
                            severity="high",
                            location=f"함수 '{node.name}'",
                            description="변경 가능한 기본 인수 사용",
                            fix_suggestion=f"def {node.name}(..., arg=None): 후 'if arg is None: arg = []' 패턴 사용",
                        ))

        return bugs

    def _llm_bug_detection(self, code: str, language: str) -> List[BugReport]:
        """LLM 기반 고급 버그 감지"""
        if not self.llm:
            return []
        from jarvis.llm.manager import Message

        prompt = f"""다음 {language} 코드의 버그, 보안 취약점, 성능 문제를 찾아주세요.

```{language}
{code[:5000]}
```

JSON 배열로 반환:
[
  {{
    "severity": "critical|high|medium|low",
    "location": "줄 번호 또는 함수명",
    "description": "문제 설명",
    "fix_suggestion": "수정 방법"
  }}
]

문제가 없으면 빈 배열 []을 반환하세요."""

        messages = [Message(role="user", content=prompt)]
        try:
            response = self.llm.chat(messages, system="당신은 보안 전문가이자 코드 리뷰어입니다.", max_tokens=2048)
            match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return [
                    BugReport(
                        severity=item.get("severity", "low"),
                        location=item.get("location", "unknown"),
                        description=item.get("description", ""),
                        fix_suggestion=item.get("fix_suggestion", ""),
                    )
                    for item in data
                ]
        except Exception as e:
            logger.debug(f"LLM bug detection error: {e}")
        return []

    def fix_bugs(self, code: str, bugs: List[BugReport], language: str = "python") -> str:
        """발견된 버그들을 자동으로 수정"""
        if not self.llm or not bugs:
            return code

        from jarvis.llm.manager import Message
        bug_list = "\n".join([f"- [{b.severity}] {b.location}: {b.description}" for b in bugs])

        prompt = f"""다음 {language} 코드의 모든 버그를 수정해주세요.

발견된 버그:
{bug_list}

원본 코드:
```{language}
{code}
```

수정된 전체 코드만 반환:
```{language}
[수정된 코드]
```"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="코드 수정 전문가입니다.", max_tokens=8192)
        match = re.search(rf'```{language}\n(.*?)```', response.content, re.DOTALL)
        if not match:
            match = re.search(r'```(?:\w+)?\n(.*?)```', response.content, re.DOTALL)
        return match.group(1).strip() if match else code

    # ── 코드 최적화 ───────────────────────────────────────────────────────
    def optimize(self, code: str, language: str = "python", goal: str = "speed") -> Dict:
        """코드 성능 최적화"""
        if not self.llm:
            return {"error": "LLM 없음"}

        from jarvis.llm.manager import Message
        goal_map = {
            "speed": "실행 속도 최적화",
            "memory": "메모리 사용 최소화",
            "readability": "가독성 향상",
            "all": "속도/메모리/가독성 종합 최적화",
        }

        prompt = f"""다음 {language} 코드를 {goal_map.get(goal, goal)} 관점에서 최적화해주세요.

원본:
```{language}
{code[:4000]}
```

응답 형식:
## 최적화된 코드
```{language}
[코드]
```

## 변경사항
- [변경 1]: [이유]
- [변경 2]: [이유]

## 성능 예상
- 기존: O(?)
- 개선: O(?)"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="성능 최적화 전문가입니다.", max_tokens=4096)

        code_match = re.search(rf'```{language}\n(.*?)```', response.content, re.DOTALL)
        optimized = code_match.group(1).strip() if code_match else code

        changes_match = re.search(r'## 변경사항\n(.*?)(?=##|\Z)', response.content, re.DOTALL)
        changes = changes_match.group(1).strip() if changes_match else ""

        return {
            "optimized_code": optimized,
            "changes": changes,
            "goal": goal,
            "original_lines": len(code.splitlines()),
            "optimized_lines": len(optimized.splitlines()),
        }

    # ── 테스트 생성 ───────────────────────────────────────────────────────
    def generate_tests(self, code: str, framework: str = "pytest") -> str:
        """단위 테스트 자동 생성"""
        if not self.llm:
            return "# LLM 없음"

        from jarvis.llm.manager import Message
        prompt = f"""다음 Python 코드에 대한 포괄적인 {framework} 단위 테스트를 작성해주세요.

코드:
```python
{code[:4000]}
```

요구사항:
- 정상 케이스 테스트
- 엣지 케이스 (None, 빈 입력, 경계값)
- 예외 케이스
- 모킹이 필요하면 unittest.mock 사용

```python
[테스트 코드만 반환]
```"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="TDD 전문가입니다.", max_tokens=4096)
        match = re.search(r'```python\n(.*?)```', response.content, re.DOTALL)
        return match.group(1).strip() if match else response.content

    # ── 코드 설명 ─────────────────────────────────────────────────────────
    def explain(self, code: str, detail_level: str = "medium", language: str = "ko") -> str:
        """코드 상세 설명"""
        if not self.llm:
            return "LLM 없음"

        from jarvis.llm.manager import Message
        level_map = {"low": "한 문단으로", "medium": "섹션별로", "high": "줄 단위로"}
        prompt = f"""다음 코드를 {level_map.get(detail_level, '섹션별로')} 설명해주세요.

```
{code[:5000]}
```

{"한국어로 설명하세요." if language == "ko" else "Explain in English."}
알고리즘, 자료구조, 복잡도, 잠재적 문제점도 포함하세요."""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, max_tokens=3000)
        return response.content

    # ── 코드 실행 및 검증 ─────────────────────────────────────────────────
    def run_and_verify(self, code: str, expected_output: str = None) -> Dict:
        """코드 실행 및 결과 검증"""
        if not self.executor:
            return {"error": "코드 실행기 없음"}

        result = self.executor.execute_python(code)
        verified = True
        if expected_output and result.get("output"):
            verified = expected_output.strip() in result["output"].strip()

        return {
            "execution": result,
            "verified": verified,
            "expected": expected_output,
        }

    # ── 문서 생성 ─────────────────────────────────────────────────────────
    def generate_docs(self, code: str, doc_format: str = "markdown") -> str:
        """코드 API 문서 자동 생성"""
        if not self.llm:
            return "LLM 없음"

        from jarvis.llm.manager import Message
        prompt = f"""다음 Python 코드의 API 문서를 {doc_format} 형식으로 생성해주세요.

```python
{code[:5000]}
```

포함 사항:
- 모든 함수/클래스 설명
- 파라미터 타입 및 설명
- 반환값
- 사용 예시
- 주의사항"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, max_tokens=4096)
        return response.content

    def get_history(self) -> List[Dict]:
        return [
            {
                "language": r.language,
                "code_preview": r.code[:100],
                "complexity": r.complexity,
                "generated_at": r.generated_at,
            }
            for r in reversed(self._history[-20:])
        ]
