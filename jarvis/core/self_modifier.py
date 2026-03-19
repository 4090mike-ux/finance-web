"""
JARVIS 자기 코드 수정 시스템 — Iteration 3
JARVIS가 자신의 Python 소스 코드를 읽고, 분석하고, 개선하는 시스템
- 코드 품질 분석
- 버그 자동 탐지 및 수정 제안
- 새 기능 자동 추가
- 리팩토링 제안
- 안전한 수정 (백업 후 적용, 문법 검증)

경고: 이 시스템은 자신의 코드를 수정합니다.
모든 수정은 백업 후 적용되며 문법 검증이 선행됩니다.
"""

import ast
import json
import time
import shutil
import logging
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

JARVIS_ROOT = Path(__file__).parent.parent.parent
BACKUP_DIR = JARVIS_ROOT / "data" / "jarvis" / "code_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CodeAnalysis:
    file_path: str
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: int
    issues: List[str]
    suggestions: List[str]
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Modification:
    file_path: str
    description: str
    old_code: str
    new_code: str
    backup_path: str = ""
    applied: bool = False
    verified: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SelfModifier:
    """
    JARVIS 자기 수정 시스템
    자신의 코드를 분석하고 LLM의 도움으로 개선
    """

    # 수정 가능한 파일 목록 (안전 목록)
    MODIFIABLE_FILES = [
        "jarvis/skills/skill_library.py",
        "jarvis/web/intelligence.py",
        "jarvis/core/reasoning.py",
        "jarvis/agents/agent_manager.py",
        "jarvis/config.py",
    ]

    # 절대 수정하면 안 되는 파일
    PROTECTED_FILES = [
        "jarvis/core/self_modifier.py",  # 자기 자신
        "jarvis_app.py",
        "jarvis/memory/memory_manager.py",
    ]

    def __init__(self, llm_manager, jarvis_root: Path = None):
        self.llm = llm_manager
        self.root = jarvis_root or JARVIS_ROOT
        self._modifications: List[Modification] = []
        self._analysis_cache: Dict[str, CodeAnalysis] = {}
        logger.info("SelfModifier initialized")

    # ── 코드 분석 ─────────────────────────────────────────────────────────
    def analyze_file(self, file_path: str) -> CodeAnalysis:
        """파일 정적 분석"""
        p = self.root / file_path
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        source = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return CodeAnalysis(
                file_path=file_path,
                lines=len(source.splitlines()),
                functions=[], classes=[], imports=[],
                complexity_score=0,
                issues=[f"SyntaxError: {e}"],
                suggestions=["문법 오류를 수정하세요."],
            )

        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend([alias.name for alias in n.names])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    imports.append(n.module)

        lines = len(source.splitlines())
        issues = []
        suggestions = []

        # 기본 품질 체크
        if lines > 500:
            issues.append(f"파일이 너무 큽니다 ({lines}줄). 모듈 분리를 고려하세요.")
        if any(len(f) > 80 for f in functions):
            issues.append("함수 이름이 너무 깁니다.")
        if len(functions) > 30:
            suggestions.append("함수가 많습니다. 클래스로 묶는 것을 고려하세요.")

        # TODO/FIXME 체크
        for i, line in enumerate(source.splitlines(), 1):
            if "TODO" in line or "FIXME" in line or "HACK" in line:
                issues.append(f"Line {i}: {line.strip()[:60]}")

        complexity = min(100, lines // 10 + len(functions) * 2 + len(classes) * 5)

        analysis = CodeAnalysis(
            file_path=file_path,
            lines=lines,
            functions=functions,
            classes=classes,
            imports=imports,
            complexity_score=complexity,
            issues=issues,
            suggestions=suggestions,
        )
        self._analysis_cache[file_path] = analysis
        return analysis

    def analyze_all(self) -> List[Dict]:
        """모든 수정 가능 파일 분석"""
        results = []
        for f in self.MODIFIABLE_FILES:
            try:
                analysis = self.analyze_file(f)
                results.append({
                    "file": f,
                    "lines": analysis.lines,
                    "functions": len(analysis.functions),
                    "classes": len(analysis.classes),
                    "complexity": analysis.complexity_score,
                    "issues_count": len(analysis.issues),
                    "issues": analysis.issues[:3],
                    "suggestions": analysis.suggestions[:3],
                })
            except Exception as e:
                results.append({"file": f, "error": str(e)})
        return results

    # ── LLM 기반 코드 개선 ────────────────────────────────────────────────
    def suggest_improvement(self, file_path: str, focus: str = "general") -> Dict:
        """LLM이 특정 파일에 대한 개선 제안 생성"""
        if not self.llm:
            return {"error": "LLM 없음"}
        if file_path in self.PROTECTED_FILES:
            return {"error": f"보호된 파일: {file_path}"}

        p = self.root / file_path
        if not p.exists():
            return {"error": f"파일 없음: {file_path}"}

        source = p.read_text(encoding="utf-8")
        analysis = self.analyze_file(file_path)

        from jarvis.llm.manager import Message
        prompt = f"""다음 Python 파일을 분석하고 개선 제안을 해주세요.

파일: {file_path}
분석 결과:
- 줄 수: {analysis.lines}
- 함수: {len(analysis.functions)}개
- 복잡도: {analysis.complexity_score}/100
- 발견된 문제: {', '.join(analysis.issues[:3]) if analysis.issues else '없음'}
- 집중 영역: {focus}

코드 (처음 100줄):
```python
{chr(10).join(source.splitlines()[:100])}
```

다음 형식으로 JSON 응답해주세요:
{{
  "summary": "전반적인 코드 평가",
  "critical_issues": ["심각한 문제들"],
  "improvements": [
    {{
      "type": "performance|readability|security|functionality",
      "description": "개선 사항 설명",
      "location": "함수명 또는 줄 범위"
    }}
  ],
  "new_feature_suggestion": "추가하면 좋을 기능",
  "refactoring_priority": "high|medium|low"
}}"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="당신은 Python 코드 전문 리뷰어입니다. JSON만 출력하세요.", max_tokens=2048)

        try:
            import re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                return {"success": True, "file": file_path, "suggestion": json.loads(match.group())}
        except Exception:
            pass
        return {"success": True, "file": file_path, "suggestion": {"summary": response.content}}

    def generate_improvement(self, file_path: str, improvement_desc: str) -> Optional[Modification]:
        """특정 개선 사항의 코드를 LLM이 생성"""
        if not self.llm:
            return None
        if file_path in self.PROTECTED_FILES:
            logger.warning(f"Cannot modify protected file: {file_path}")
            return None

        p = self.root / file_path
        source = p.read_text(encoding="utf-8")

        from jarvis.llm.manager import Message
        prompt = f"""다음 Python 파일에서 아래 개선 사항을 구현해주세요.

파일: {file_path}
개선 사항: {improvement_desc}

현재 코드:
```python
{source[:4000]}
```

규칙:
1. 기존 코드의 구조를 최대한 유지
2. 안전한 변경만 수행
3. 반드시 문법적으로 올바른 코드
4. 변경된 부분에만 집중

전체 수정된 파일을 출력해주세요:
```python
[수정된 전체 코드]
```"""

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, system="당신은 Python 코드를 개선하는 전문가입니다. 코드 블록만 출력하세요.", max_tokens=8192)

        import re
        match = re.search(r'```(?:python)?\n(.*?)```', response.content, re.DOTALL)
        if not match:
            return None

        new_code = match.group(1).strip()

        # 문법 검증
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            logger.error(f"Generated code has syntax error: {e}")
            return None

        mod = Modification(
            file_path=file_path,
            description=improvement_desc,
            old_code=source,
            new_code=new_code,
        )
        self._modifications.append(mod)
        return mod

    # ── 수정 적용 ─────────────────────────────────────────────────────────
    def apply_modification(self, mod: Modification, dry_run: bool = True) -> Dict:
        """코드 수정 적용 (백업 후)"""
        if dry_run:
            return {
                "dry_run": True,
                "file": mod.file_path,
                "description": mod.description,
                "lines_before": len(mod.old_code.splitlines()),
                "lines_after": len(mod.new_code.splitlines()),
                "diff_summary": f"+{len(mod.new_code.splitlines()) - len(mod.old_code.splitlines())} 줄",
            }

        if mod.file_path in self.PROTECTED_FILES:
            return {"error": "보호된 파일"}

        p = self.root / mod.file_path

        # 백업
        backup_name = f"{p.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py.bak"
        backup_path = BACKUP_DIR / backup_name
        shutil.copy2(p, backup_path)
        mod.backup_path = str(backup_path)

        # 적용
        try:
            p.write_text(mod.new_code, encoding="utf-8")
            mod.applied = True
            mod.verified = True
            logger.info(f"Modification applied: {mod.file_path} (backup: {backup_path})")
            return {
                "success": True,
                "file": mod.file_path,
                "backup": str(backup_path),
                "description": mod.description,
            }
        except Exception as e:
            # 롤백
            shutil.copy2(backup_path, p)
            return {"error": f"적용 실패, 롤백 완료: {e}"}

    def rollback(self, file_path: str) -> Dict:
        """최신 백업으로 롤백"""
        backups = sorted(BACKUP_DIR.glob(f"{Path(file_path).stem}_*.py.bak"), reverse=True)
        if not backups:
            return {"error": "백업 없음"}
        latest = backups[0]
        dest = self.root / file_path
        shutil.copy2(latest, dest)
        return {"success": True, "restored_from": str(latest)}

    # ── 상태 조회 ─────────────────────────────────────────────────────────
    def get_modification_history(self) -> List[Dict]:
        return [
            {
                "file": m.file_path,
                "description": m.description,
                "applied": m.applied,
                "timestamp": m.timestamp,
                "backup": m.backup_path,
            }
            for m in reversed(self._modifications[-20:])
        ]

    def get_status(self) -> Dict:
        return {
            "modifiable_files": len(self.MODIFIABLE_FILES),
            "protected_files": len(self.PROTECTED_FILES),
            "total_modifications": len(self._modifications),
            "applied_count": sum(1 for m in self._modifications if m.applied),
            "backups": len(list(BACKUP_DIR.glob("*.bak"))),
            "backup_dir": str(BACKUP_DIR),
        }
