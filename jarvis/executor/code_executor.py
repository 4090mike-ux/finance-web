"""
JARVIS 코드 실행 엔진
- Python 코드 안전 실행 (타임아웃, 샌드박스)
- Shell 명령 실행
- 코드 분석 및 리뷰
"""

import sys
import io
import ast
import time
import textwrap
import traceback
import subprocess
import threading
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    안전한 코드 실행 엔진
    - Python: 타임아웃 및 기본 안전장치
    - Shell: 위험 명령 차단
    - 결과 캡처 및 포맷팅
    """

    BLOCKED_IMPORTS = [
        "ctypes", "winreg", "winsound",  # 저수준 OS 접근 일부
    ]

    BLOCKED_PATTERNS = [
        "os.system('rm -rf",
        "shutil.rmtree('/'",
        "subprocess.run(['rm', '-rf', '/",
        "__import__('ctypes'",
    ]

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.execution_history = []

    def execute_python(self, code: str, globals_dict: Dict = None) -> Dict:
        """Python 코드 실행"""
        start_time = time.time()

        # 코드 안전성 검사
        safety_check = self._check_python_safety(code)
        if not safety_check["safe"]:
            return {
                "success": False,
                "error": f"Safety check failed: {safety_check['reason']}",
                "code": code,
            }

        # 실행 환경 설정
        exec_globals = {
            "__builtins__": __builtins__,
            "__name__": "__jarvis_exec__",
        }
        if globals_dict:
            exec_globals.update(globals_dict)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = {"output": "", "error": "", "return_value": None, "success": False}

        def run_code():
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # 마지막 표현식의 값 캡처
                    tree = ast.parse(code)
                    if isinstance(tree.body[-1], ast.Expr):
                        # 마지막 줄이 표현식이면 값을 출력
                        last_expr = ast.Expression(tree.body[-1].value)
                        exec(compile(ast.Module(tree.body[:-1], []), "<jarvis>", "exec"), exec_globals)
                        return_val = eval(compile(last_expr, "<jarvis>", "eval"), exec_globals)
                        result["return_value"] = repr(return_val)
                    else:
                        exec(compile(tree, "<jarvis>", "exec"), exec_globals)
                result["success"] = True
            except SyntaxError as e:
                result["error"] = f"SyntaxError: {e}"
            except Exception as e:
                result["error"] = traceback.format_exc()

        # 타임아웃으로 실행
        thread = threading.Thread(target=run_code)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return {
                "success": False,
                "error": f"Execution timed out after {self.timeout}s",
                "code": code,
            }

        duration = time.time() - start_time
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()

        # 이력 기록
        self.execution_history.append({
            "code": code[:200],
            "success": result["success"],
            "duration": round(duration, 3),
            "timestamp": time.time(),
        })

        return {
            "success": result["success"],
            "output": output[:5000] if output else "",
            "stderr": error_output[:2000] if error_output else "",
            "error": result.get("error", ""),
            "return_value": result.get("return_value"),
            "duration": round(duration, 3),
            "code": code,
        }

    def _check_python_safety(self, code: str) -> Dict:
        """Python 코드 안전성 검사"""
        code_lower = code.lower()

        # 차단 패턴 검사
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return {"safe": False, "reason": f"Blocked pattern detected: {pattern}"}

        # AST 기반 검사
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 위험한 import 검사
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]

                    for name in names:
                        if any(name.startswith(b) for b in self.BLOCKED_IMPORTS):
                            return {"safe": False, "reason": f"Blocked import: {name}"}
        except SyntaxError:
            pass  # SyntaxError는 실행 시점에 처리

        return {"safe": True, "reason": ""}

    def execute_shell(self, command: str, cwd: str = None, timeout: int = 30) -> Dict:
        """Shell 명령 실행"""
        BLOCKED = [
            "rm -rf /", "format c:", "del /q /s c:\\",
            ":(){:|:&};:", "mkfs", "dd if=/dev/zero",
        ]
        cmd_lower = command.lower()
        for b in BLOCKED:
            if b in cmd_lower:
                return {"success": False, "error": f"Blocked command: {b}"}

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=timeout, cwd=cwd,
                encoding="utf-8", errors="replace",
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
                "command": command,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def analyze_code(self, code: str) -> Dict:
        """코드 분석 (AST 기반)"""
        try:
            tree = ast.parse(code)
            stats = {
                "functions": [],
                "classes": [],
                "imports": [],
                "line_count": len(code.splitlines()),
                "complexity": 0,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    stats["functions"].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    stats["classes"].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        stats["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        stats["imports"].append(node.module)
                # 복잡도 계산 (분기문 수)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                    stats["complexity"] += 1

            stats["imports"] = list(set(stats["imports"]))
            return {"success": True, "analysis": stats}
        except SyntaxError as e:
            return {"success": False, "error": f"SyntaxError: {e}"}

    def format_code(self, code: str) -> str:
        """코드 포맷팅 (들여쓰기 정규화)"""
        return textwrap.dedent(code).strip()

    def get_history(self) -> List[Dict]:
        """실행 이력"""
        return self.execution_history[-20:]
