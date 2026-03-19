"""
JARVIS 스킬 라이브러리 — Iteration 2
JARVIS가 스스로 새 스킬(Python 함수)을 작성하고 저장/재사용하는 시스템
- 자동 스킬 발견 및 등록
- 실행 통계 및 신뢰도 추적
- 스킬 구성: 단순 스킬 → 파이프라인 조합
- 실시간 스킬 생성 (LLM → 코드 → 저장)
"""

import os
import re
import ast
import json
import time
import inspect
import logging
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

SKILL_DIR = Path(__file__).parent / "library"
SKILL_DIR.mkdir(exist_ok=True)
SKILL_INDEX = SKILL_DIR / "_index.json"


@dataclass
class SkillMeta:
    name: str
    description: str
    category: str
    author: str = "JARVIS"
    version: str = "1.0"
    params: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""
    use_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    avg_duration_ms: float = 0.0
    tags: List[str] = field(default_factory=list)
    file_path: str = ""


class SkillLibrary:
    """
    JARVIS 스킬 라이브러리
    - 스킬 저장/로드/실행
    - LLM으로 새 스킬 자동 생성
    - 실행 통계 추적
    """

    BUILT_IN_SKILLS = {
        "calculate": {
            "description": "수학 표현식 안전 계산",
            "category": "math",
            "tags": ["math", "calculate", "compute"],
        },
        "format_table": {
            "description": "데이터를 Markdown 테이블로 포맷",
            "category": "formatting",
            "tags": ["format", "table", "output"],
        },
        "summarize_text": {
            "description": "긴 텍스트를 요약",
            "category": "nlp",
            "tags": ["summary", "text", "nlp"],
        },
        "extract_keywords": {
            "description": "텍스트에서 핵심 키워드 추출",
            "category": "nlp",
            "tags": ["keyword", "extract", "nlp"],
        },
        "json_to_table": {
            "description": "JSON 데이터를 읽기 좋은 형식으로 변환",
            "category": "formatting",
            "tags": ["json", "format", "convert"],
        },
        "diff_texts": {
            "description": "두 텍스트의 차이점 분석",
            "category": "analysis",
            "tags": ["diff", "compare", "text"],
        },
        "count_words": {
            "description": "텍스트 단어/문자/줄 수 통계",
            "category": "analysis",
            "tags": ["count", "words", "stats"],
        },
        "detect_language": {
            "description": "텍스트 언어 감지",
            "category": "nlp",
            "tags": ["language", "detect", "nlp"],
        },
        "extract_urls": {
            "description": "텍스트에서 URL 추출",
            "category": "extraction",
            "tags": ["url", "extract", "links"],
        },
        "timestamp_now": {
            "description": "현재 시각 다양한 형식으로 반환",
            "category": "utility",
            "tags": ["time", "date", "timestamp"],
        },
    }

    def __init__(self, llm_manager=None):
        self.llm = llm_manager
        self._registry: Dict[str, SkillMeta] = {}
        self._callables: Dict[str, Callable] = {}
        self._load_index()
        self._register_built_ins()
        self._load_custom_skills()
        logger.info(f"SkillLibrary initialized — {len(self._registry)} skills available")

    # ── 인덱스 관리 ────────────────────────────────────────────
    def _load_index(self):
        if SKILL_INDEX.exists():
            try:
                with open(SKILL_INDEX, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for name, meta_dict in data.items():
                    self._registry[name] = SkillMeta(**meta_dict)
            except Exception as e:
                logger.warning(f"Index load failed: {e}")

    def _save_index(self):
        try:
            data = {name: asdict(meta) for name, meta in self._registry.items()}
            with open(SKILL_INDEX, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Index save failed: {e}")

    # ── 빌트인 스킬 등록 ───────────────────────────────────────
    def _register_built_ins(self):
        built_ins = {
            "calculate": self._skill_calculate,
            "format_table": self._skill_format_table,
            "summarize_text": self._skill_summarize_text,
            "extract_keywords": self._skill_extract_keywords,
            "json_to_table": self._skill_json_to_table,
            "diff_texts": self._skill_diff_texts,
            "count_words": self._skill_count_words,
            "detect_language": self._skill_detect_language,
            "extract_urls": self._skill_extract_urls,
            "timestamp_now": self._skill_timestamp_now,
        }
        for name, func in built_ins.items():
            self._callables[name] = func
            if name not in self._registry:
                info = self.BUILT_IN_SKILLS[name]
                sig = inspect.signature(func)
                self._registry[name] = SkillMeta(
                    name=name,
                    description=info["description"],
                    category=info["category"],
                    tags=info["tags"],
                    params=list(sig.parameters.keys()),
                )
        self._save_index()

    # ── 커스텀 스킬 로드 ───────────────────────────────────────
    def _load_custom_skills(self):
        for py_file in SKILL_DIR.glob("skill_*.py"):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SKILL_NAME") and hasattr(mod, "run"):
                    name = mod.SKILL_NAME
                    self._callables[name] = mod.run
                    if name not in self._registry:
                        self._registry[name] = SkillMeta(
                            name=name,
                            description=getattr(mod, "SKILL_DESCRIPTION", ""),
                            category=getattr(mod, "SKILL_CATEGORY", "custom"),
                            tags=getattr(mod, "SKILL_TAGS", []),
                            file_path=str(py_file),
                        )
                    logger.info(f"Custom skill loaded: {name}")
            except Exception as e:
                logger.error(f"Custom skill load error ({py_file}): {e}")

    # ── 스킬 실행 ──────────────────────────────────────────────
    def run(self, name: str, **kwargs) -> Dict:
        if name not in self._callables:
            return {"success": False, "error": f"Skill '{name}' not found"}

        meta = self._registry.get(name)
        start = time.perf_counter()
        try:
            result = self._callables[name](**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            if meta:
                meta.use_count += 1
                meta.success_count += 1
                meta.last_used = datetime.now().isoformat()
                # 이동 평균 업데이트
                meta.avg_duration_ms = (
                    meta.avg_duration_ms * (meta.success_count - 1) + duration_ms
                ) / meta.success_count
            self._save_index()
            return {"success": True, "result": result, "duration_ms": round(duration_ms, 2)}
        except Exception as e:
            if meta:
                meta.fail_count += 1
            self._save_index()
            logger.error(f"Skill '{name}' execution error: {e}")
            return {"success": False, "error": str(e)}

    # ── LLM으로 새 스킬 생성 ───────────────────────────────────
    def create_skill(self, description: str, category: str = "custom", tags: List[str] = None) -> Dict:
        """LLM이 새 스킬 코드를 작성하고 저장"""
        if not self.llm:
            return {"success": False, "error": "LLM 없음 — 자동 스킬 생성 불가"}

        prompt = f"""다음 기능을 수행하는 Python 스킬을 작성하세요.

기능 설명: {description}
카테고리: {category}

규칙:
1. SKILL_NAME = "snake_case_name" (영어, 고유한 이름)
2. SKILL_DESCRIPTION = "한 줄 설명"
3. SKILL_CATEGORY = "{category}"
4. SKILL_TAGS = ["tag1", "tag2"]
5. def run(**kwargs) -> Any: 함수 구현
6. 안전한 코드만 (파일 삭제, 시스템 명령 금지)
7. 외부 라이브러리는 try/except로 처리

```python
SKILL_NAME = "..."
SKILL_DESCRIPTION = "..."
SKILL_CATEGORY = "{category}"
SKILL_TAGS = [...]

def run(**kwargs):
    \"\"\"독스트링\"\"\"
    ...
    return result
```

코드 블록만 출력하세요."""

        from jarvis.llm.manager import Message
        messages = [Message(role="user", content=prompt)]
        system = "당신은 Python 코드만 출력하는 전문 개발자입니다. 설명 없이 코드 블록만 반환하세요."

        try:
            response = self.llm.chat(messages, system=system, max_tokens=2048)
            code = self._extract_code(response.content)
            if not code:
                return {"success": False, "error": "LLM이 코드를 생성하지 못했습니다"}

            # 안전성 검사
            ast.parse(code)  # 문법 오류 체크

            # 파일 저장
            skill_name = self._extract_skill_name(code) or f"skill_{int(time.time())}"
            file_path = SKILL_DIR / f"skill_{skill_name}.py"
            file_path.write_text(code, encoding="utf-8")

            # 동적 로드 및 등록
            spec = importlib.util.spec_from_file_location(f"skill_{skill_name}", file_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._callables[skill_name] = mod.run
            self._registry[skill_name] = SkillMeta(
                name=skill_name,
                description=getattr(mod, "SKILL_DESCRIPTION", description),
                category=category,
                tags=tags or getattr(mod, "SKILL_TAGS", []),
                file_path=str(file_path),
            )
            self._save_index()
            logger.info(f"New skill created: {skill_name}")
            return {"success": True, "skill_name": skill_name, "file": str(file_path), "code": code}
        except SyntaxError as e:
            return {"success": False, "error": f"코드 문법 오류: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _extract_skill_name(self, code: str) -> Optional[str]:
        match = re.search(r'SKILL_NAME\s*=\s*["\'](\w+)["\']', code)
        return match.group(1) if match else None

    # ── 조회 ───────────────────────────────────────────────────
    def list_skills(self, category: str = None, tag: str = None) -> List[Dict]:
        results = []
        for name, meta in self._registry.items():
            if category and meta.category != category:
                continue
            if tag and tag not in meta.tags:
                continue
            results.append({
                "name": name,
                "description": meta.description,
                "category": meta.category,
                "tags": meta.tags,
                "use_count": meta.use_count,
                "success_rate": (
                    round(meta.success_count / (meta.success_count + meta.fail_count) * 100, 1)
                    if (meta.success_count + meta.fail_count) > 0 else None
                ),
                "avg_duration_ms": meta.avg_duration_ms,
                "last_used": meta.last_used,
            })
        return sorted(results, key=lambda x: x["use_count"], reverse=True)

    def search_skills(self, query: str) -> List[Dict]:
        q = query.lower()
        return [
            s for s in self.list_skills()
            if q in s["name"] or q in s["description"].lower() or q in " ".join(s["tags"])
        ]

    def get_skill_code(self, name: str) -> Optional[str]:
        meta = self._registry.get(name)
        if meta and meta.file_path and Path(meta.file_path).exists():
            return Path(meta.file_path).read_text(encoding="utf-8")
        func = self._callables.get(name)
        return inspect.getsource(func) if func else None

    def get_stats(self) -> Dict:
        total = len(self._registry)
        categories = {}
        for meta in self._registry.values():
            categories[meta.category] = categories.get(meta.category, 0) + 1
        top = sorted(self._registry.values(), key=lambda m: m.use_count, reverse=True)[:5]
        return {
            "total_skills": total,
            "categories": categories,
            "total_executions": sum(m.use_count for m in self._registry.values()),
            "top_skills": [{"name": m.name, "uses": m.use_count} for m in top],
        }

    # ── 빌트인 스킬 구현 ───────────────────────────────────────
    def _skill_calculate(self, expression: str = "", **kw) -> Any:
        safe_env = {"__builtins__": {}}
        import math
        for name in dir(math):
            if not name.startswith("_"):
                safe_env[name] = getattr(math, name)
        return eval(expression, safe_env)

    def _skill_format_table(self, data: list = None, headers: list = None, **kw) -> str:
        if not data:
            return "데이터 없음"
        if not headers and isinstance(data[0], dict):
            headers = list(data[0].keys())
        if headers:
            rows = [[str(row.get(h, "")) for h in headers] if isinstance(row, dict) else [str(v) for v in row] for row in data]
        else:
            rows = [[str(v) for v in row] for row in data]
            headers = [f"Col{i+1}" for i in range(len(rows[0]))] if rows else []
        col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
        row_lines = ["| " + " | ".join(str(rows[r][i]).ljust(col_widths[i]) for i in range(len(headers))) + " |" for r in range(len(rows))]
        return "\n".join([header_line, sep] + row_lines)

    def _skill_summarize_text(self, text: str = "", max_sentences: int = 3, **kw) -> str:
        sentences = re.split(r'[.!?。]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return ". ".join(sentences[:max_sentences]) + "." if sentences else text[:200]

    def _skill_extract_keywords(self, text: str = "", top_n: int = 10, **kw) -> List[str]:
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                      "of", "with", "by", "이", "가", "은", "는", "을", "를", "의", "에", "도"}
        words = re.findall(r'\b[가-힣a-zA-Z]{2,}\b', text.lower())
        freq = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    def _skill_json_to_table(self, data=None, **kw) -> str:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return self._skill_format_table(data=data)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _skill_diff_texts(self, text_a: str = "", text_b: str = "", **kw) -> Dict:
        import difflib
        d = difflib.Differ()
        diff = list(d.compare(text_a.splitlines(), text_b.splitlines()))
        added = [l[2:] for l in diff if l.startswith("+ ")]
        removed = [l[2:] for l in diff if l.startswith("- ")]
        return {"added": added, "removed": removed, "changed_lines": len(added) + len(removed),
                "similarity": round(difflib.SequenceMatcher(None, text_a, text_b).ratio() * 100, 1)}

    def _skill_count_words(self, text: str = "", **kw) -> Dict:
        return {
            "chars": len(text),
            "chars_no_space": len(text.replace(" ", "")),
            "words": len(text.split()),
            "lines": len(text.splitlines()),
            "sentences": len(re.split(r'[.!?。]+', text)),
            "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
        }

    def _skill_detect_language(self, text: str = "", **kw) -> str:
        korean = len(re.findall(r'[가-힣]', text))
        japanese = len(re.findall(r'[ぁ-ゔァ-ヴ]', text))
        chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        total = len(text.replace(" ", ""))
        if total == 0:
            return "unknown"
        if korean / total > 0.1:
            return "Korean"
        if japanese / total > 0.1:
            return "Japanese"
        if chinese / total > 0.1:
            return "Chinese"
        return "English"

    def _skill_extract_urls(self, text: str = "", **kw) -> List[str]:
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return list(set(re.findall(pattern, text)))

    def _skill_timestamp_now(self, fmt: str = "all", **kw) -> Any:
        now = datetime.now()
        formats = {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "unix": int(time.time()),
            "weekday": now.strftime("%A"),
        }
        return formats if fmt == "all" else formats.get(fmt, now.isoformat())
