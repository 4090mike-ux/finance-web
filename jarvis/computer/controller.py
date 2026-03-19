"""
JARVIS 컴퓨터 제어 모듈
- 파일 시스템 조작
- 프로세스 모니터링 및 관리
- 시스템 정보 수집
- 클립보드 관리
- 스크린샷 캡처
소프트웨어 전용 - 물리적 장치 제어 없음
"""

import os
import sys
import json
import time
import shutil
import psutil
import logging
import platform
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ComputerController:
    """
    JARVIS 컴퓨터 제어 시스템
    안전한 컴퓨터 조작 - 허가된 디렉토리 내에서만 동작
    """

    DANGEROUS_CMDS = [
        "rm -rf /", "format c:", "del /q /s c:\\",
        "mkfs", "dd if=/dev/zero", ":(){:|:&};:",
        "chmod -R 777 /", "sudo rm", "shutdown", "halt", "reboot"
    ]

    def __init__(self, allowed_dirs: List[str] = None):
        self.allowed_dirs = [Path(d).resolve() for d in (allowed_dirs or [])]
        self.home = Path.home()
        self.cwd = Path.cwd()
        logger.info(f"ComputerController initialized - allowed_dirs: {len(self.allowed_dirs)}")

    # ==================== 시스템 정보 ====================

    def get_system_info(self) -> Dict[str, Any]:
        """전체 시스템 정보 수집"""
        try:
            cpu_freq = psutil.cpu_freq()
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            info = {
                "platform": {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python_version": sys.version,
                },
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                    "usage_percent": psutil.cpu_percent(interval=0.1),
                    "frequency_mhz": cpu_freq.current if cpu_freq else 0,
                },
                "memory": {
                    "total_gb": round(mem.total / 1e9, 2),
                    "available_gb": round(mem.available / 1e9, 2),
                    "used_percent": mem.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / 1e9, 2),
                    "free_gb": round(disk.free / 1e9, 2),
                    "used_percent": disk.percent,
                },
                "network": self._get_network_info(),
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
                "timestamp": datetime.now().isoformat(),
            }
            return info
        except Exception as e:
            logger.error(f"System info error: {e}")
            return {"error": str(e)}

    def _get_network_info(self) -> Dict:
        """네트워크 정보"""
        try:
            net_io = psutil.net_io_counters()
            return {
                "bytes_sent_mb": round(net_io.bytes_sent / 1e6, 2),
                "bytes_recv_mb": round(net_io.bytes_recv / 1e6, 2),
            }
        except:
            return {}

    def get_running_processes(self, top_n: int = 20) -> List[Dict]:
        """실행 중인 프로세스 목록 (CPU 사용량 기준)"""
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = proc.info
                processes.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": round(info["cpu_percent"] or 0, 1),
                    "mem": round(info["memory_percent"] or 0, 1),
                    "status": info["status"],
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # CPU 사용량 기준 정렬
        processes.sort(key=lambda x: x["cpu"], reverse=True)
        return processes[:top_n]

    def get_cpu_history(self, seconds: int = 5) -> List[float]:
        """CPU 사용률 히스토리 수집"""
        history = []
        for _ in range(seconds):
            history.append(psutil.cpu_percent(interval=1))
        return history

    # ==================== 파일 시스템 ====================

    def list_directory(self, path: str = ".", show_hidden: bool = False) -> Dict:
        """디렉토리 목록 조회"""
        try:
            target = Path(path).resolve()
            if not target.exists():
                return {"error": f"Path not found: {path}"}

            items = []
            for item in sorted(target.iterdir()):
                if not show_hidden and item.name.startswith("."):
                    continue
                try:
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "path": str(item),
                    })
                except PermissionError:
                    items.append({"name": item.name, "type": "unknown", "error": "permission denied"})

            return {
                "path": str(target),
                "items": items,
                "total": len(items),
            }
        except Exception as e:
            return {"error": str(e)}

    def read_file(self, path: str, max_lines: int = 200) -> Dict:
        """파일 읽기"""
        try:
            target = Path(path).resolve()
            if not target.exists():
                return {"error": f"File not found: {path}"}
            if not target.is_file():
                return {"error": f"Not a file: {path}"}

            # 바이너리 파일 감지
            with open(target, "rb") as f:
                chunk = f.read(1024)
                if b"\x00" in chunk:
                    return {"error": "Binary file - cannot display as text", "path": str(target)}

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if max_lines and total_lines > max_lines:
                content = "".join(lines[:max_lines]) + f"\n... ({total_lines - max_lines} more lines)"
            else:
                content = "".join(lines)

            return {
                "path": str(target),
                "content": content,
                "total_lines": total_lines,
                "size": target.stat().st_size,
            }
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str, append: bool = False) -> Dict:
        """파일 쓰기"""
        try:
            target = Path(path).resolve()
            mode = "a" if append else "w"
            with open(target, mode, encoding="utf-8") as f:
                f.write(content)

            return {"success": True, "path": str(target), "size": target.stat().st_size}
        except Exception as e:
            return {"error": str(e)}

    def search_files(self, directory: str, pattern: str, recursive: bool = True) -> Dict:
        """파일 검색"""
        try:
            base = Path(directory).resolve()
            if not base.exists():
                return {"error": f"Directory not found: {directory}"}

            results = []
            if recursive:
                matches = list(base.rglob(pattern))
            else:
                matches = list(base.glob(pattern))

            for m in matches[:50]:  # 최대 50개
                try:
                    stat = m.stat()
                    results.append({
                        "path": str(m),
                        "name": m.name,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except:
                    results.append({"path": str(m), "name": m.name})

            return {"pattern": pattern, "results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e)}

    def get_file_info(self, path: str) -> Dict:
        """파일 상세 정보"""
        try:
            target = Path(path).resolve()
            if not target.exists():
                return {"error": f"Not found: {path}"}

            stat = target.stat()
            return {
                "path": str(target),
                "name": target.name,
                "type": "directory" if target.is_dir() else "file",
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "suffix": target.suffix,
            }
        except Exception as e:
            return {"error": str(e)}

    # ==================== 프로세스 관리 ====================

    def get_process_info(self, pid: int) -> Dict:
        """특정 프로세스 정보"""
        try:
            proc = psutil.Process(pid)
            return {
                "pid": pid,
                "name": proc.name(),
                "status": proc.status(),
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "memory_mb": round(proc.memory_info().rss / 1e6, 2),
                "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                "cmdline": " ".join(proc.cmdline()[:5]),
            }
        except psutil.NoSuchProcess:
            return {"error": f"Process {pid} not found"}
        except psutil.AccessDenied:
            return {"error": f"Access denied for process {pid}"}

    # ==================== 안전한 명령 실행 ====================

    def run_command(self, command: str, timeout: int = 30, cwd: str = None) -> Dict:
        """안전한 셸 명령 실행"""
        # 위험 명령어 체크
        cmd_lower = command.lower()
        for dangerous in self.DANGEROUS_CMDS:
            if dangerous in cmd_lower:
                return {"error": f"Dangerous command blocked: {dangerous}"}

        try:
            work_dir = cwd or str(self.cwd)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                encoding="utf-8",
                errors="replace",
            )

            return {
                "command": command,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s", "command": command}
        except Exception as e:
            return {"error": str(e), "command": command}

    # ==================== 환경 정보 ====================

    def get_environment(self) -> Dict:
        """환경 변수 (안전한 것만)"""
        safe_keys = ["PATH", "HOME", "USER", "USERPROFILE", "COMPUTERNAME",
                     "OS", "PROCESSOR_ARCHITECTURE", "LANG", "TZ"]
        env = {}
        for key in safe_keys:
            val = os.environ.get(key)
            if val:
                env[key] = val[:200]  # 길이 제한
        return env

    def get_disk_usage(self, path: str = "/") -> Dict:
        """디스크 사용량"""
        try:
            usage = psutil.disk_usage(path)
            return {
                "path": path,
                "total_gb": round(usage.total / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2),
                "percent": usage.percent,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_memory_details(self) -> Dict:
        """메모리 상세 정보"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "ram": {
                "total_gb": round(mem.total / 1e9, 2),
                "available_gb": round(mem.available / 1e9, 2),
                "used_gb": round(mem.used / 1e9, 2),
                "percent": mem.percent,
                "cached_gb": round(getattr(mem, "cached", 0) / 1e9, 2),
            },
            "swap": {
                "total_gb": round(swap.total / 1e9, 2),
                "used_gb": round(swap.used / 1e9, 2),
                "percent": swap.percent,
            }
        }
