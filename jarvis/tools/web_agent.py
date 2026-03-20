"""
JARVIS Autonomous Web Agent — Iteration 7
자율 웹 탐색 에이전트: 목표 기반 브라우징, 데이터 추출, 폼 상호작용

Playwright가 설치된 경우 실제 브라우저 제어.
없는 경우 requests + BeautifulSoup 기반 폴백.
"""

import json
import time
import logging
import re
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WebAction:
    action_type: str   # navigate / click / type / extract / scroll / wait
    target: str        # CSS selector or URL
    value: str = ""
    description: str = ""
    result: str = ""


@dataclass
class WebTask:
    id: str
    goal: str
    start_url: str
    actions_taken: List[WebAction] = field(default_factory=list)
    result: str = ""
    success: bool = False
    error: str = ""
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


class WebAgent:
    """
    자율 웹 탐색 에이전트

    실행 모드:
    - playwright: 실제 Chromium 제어 (headless)
    - requests: HTTP 요청 + BeautifulSoup 파싱 (폴백)
    - simulation: LLM 추론 기반 시뮬레이션 (최후 폴백)
    """

    MAX_ACTIONS = 12
    MAX_CONTENT_LEN = 4000
    REQUEST_TIMEOUT = 20

    def __init__(self, llm_manager=None, data_dir: str = "data/jarvis"):
        self.llm = llm_manager
        self.tasks: List[WebTask] = []
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._mode = self._detect_mode()
        logger.info(f"WebAgent initialized — mode: {self._mode}")

    def _detect_mode(self) -> str:
        try:
            import playwright  # noqa
            return "playwright"
        except ImportError:
            pass
        try:
            import requests  # noqa
            import bs4  # noqa
            return "requests"
        except ImportError:
            pass
        return "simulation"

    def _get_client(self):
        if self.llm:
            return getattr(self.llm, "anthropic_client", None) or getattr(self.llm, "_client", None)
        return None

    def _llm_call(self, prompt: str, max_tokens: int = 1200) -> str:
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
            logger.error(f"LLM call failed: {e}")
            return ""

    # ── Public API ────────────────────────────────────────────────────

    def run(self, goal: str, start_url: str = "https://www.google.com") -> WebTask:
        """태스크를 실행하고 결과를 반환"""
        task_id = hashlib.md5(f"{goal}{time.time()}".encode()).hexdigest()[:8]
        task = WebTask(id=task_id, goal=goal, start_url=start_url)
        t0 = time.time()

        try:
            if self._mode == "playwright":
                self._run_playwright(task)
            elif self._mode == "requests":
                self._run_requests(task)
            else:
                self._run_simulation(task)
        except Exception as e:
            task.error = str(e)
            logger.error(f"WebAgent task failed: {e}")

        task.duration = time.time() - t0
        self.tasks.append(task)
        self._save_task(task)
        return task

    # ── Playwright mode ───────────────────────────────────────────────

    def _run_playwright(self, task: WebTask):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    future = ex.submit(asyncio.run, self._async_playwright(task))
                    future.result(timeout=90)
            else:
                loop.run_until_complete(self._async_playwright(task))
        except Exception as e:
            # Fallback to requests mode
            logger.warning(f"Playwright runtime error, falling back: {e}")
            self._run_requests(task)

    async def _async_playwright(self, task: WebTask):
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(task.start_url, timeout=30000)

            content = await page.content()
            actions = self._plan_actions(task.goal, task.start_url, content)

            for action in actions[:self.MAX_ACTIONS]:
                try:
                    if action.action_type == "navigate":
                        await page.goto(action.target, timeout=20000)
                    elif action.action_type == "click":
                        await page.click(action.target, timeout=8000)
                    elif action.action_type == "type":
                        await page.fill(action.target, action.value)
                    elif action.action_type == "scroll":
                        await page.evaluate("window.scrollBy(0,600)")
                    elif action.action_type == "extract":
                        els = await page.query_selector_all(action.target)
                        texts = [await el.text_content() for el in els[:10]]
                        action.result = "\n".join(filter(None, texts))
                    task.actions_taken.append(action)
                except Exception as ae:
                    logger.debug(f"Action skipped: {ae}")
                    break

            final_content = await page.content()
            task.result = self._extract_result(task.goal, final_content)
            task.success = True
            await browser.close()

    # ── Requests mode ─────────────────────────────────────────────────

    def _run_requests(self, task: WebTask):
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "JARVIS/7.0 (+https://github.com/jarvis-ai)"}
        session = requests.Session()
        session.headers.update(headers)

        url = task.start_url
        for _ in range(5):  # max 5 hops
            try:
                resp = session.get(url, timeout=self.REQUEST_TIMEOUT)
                soup = BeautifulSoup(resp.text, "html.parser")
                text_content = soup.get_text(separator="\n", strip=True)

                # Ask LLM what to do next
                next_action = self._plan_next_action_requests(
                    task.goal, url, text_content[:self.MAX_CONTENT_LEN]
                )

                action = WebAction(
                    action_type=next_action.get("action", "extract"),
                    target=next_action.get("target", ""),
                    value=next_action.get("value", ""),
                    description=next_action.get("description", ""),
                )

                if action.action_type == "done" or action.action_type == "extract":
                    task.result = self._extract_result(task.goal, text_content[:self.MAX_CONTENT_LEN])
                    task.success = True
                    task.actions_taken.append(action)
                    break
                elif action.action_type == "navigate":
                    target_url = action.target
                    if not target_url.startswith("http"):
                        from urllib.parse import urljoin
                        target_url = urljoin(url, target_url)
                    url = target_url
                    task.actions_taken.append(action)
                else:
                    task.result = self._extract_result(task.goal, text_content[:self.MAX_CONTENT_LEN])
                    task.success = True
                    break

            except Exception as e:
                task.error = str(e)
                break

    def _plan_next_action_requests(self, goal: str, url: str, content: str) -> Dict:
        prompt = f"""웹 태스크 계획 (requests 모드).

목표: {goal}
현재 URL: {url}
페이지 텍스트 (앞부분): {content[:2000]}

다음 중 하나를 JSON으로 반환하세요:
- {{"action": "navigate", "target": "URL", "description": "이동 이유"}}
- {{"action": "extract", "description": "정보 추출"}}
- {{"action": "done", "description": "완료"}}

JSON만 반환."""
        text_out = self._llm_call(prompt, 400)
        try:
            m = re.search(r'\{.*\}', text_out, re.DOTALL)
            return json.loads(m.group()) if m else {"action": "extract"}
        except Exception:
            return {"action": "extract"}

    # ── Simulation mode ───────────────────────────────────────────────

    def _run_simulation(self, task: WebTask):
        prompt = f"""웹 탐색 시뮬레이션 모드 (실제 브라우저 없음).

목표: {task.goal}
시작 URL: {task.start_url}

목표를 달성하기 위해 어떤 웹 페이지들을 방문하고 무엇을 찾을지 추론하여,
찾을 수 있는 정보를 상세히 제공하세요. 실제 검색처럼 자세하게."""
        task.result = self._llm_call(prompt, 1000)
        task.success = bool(task.result)
        task.actions_taken.append(WebAction(
            action_type="simulate",
            target=task.start_url,
            description="LLM 시뮬레이션",
        ))

    # ── Helpers ───────────────────────────────────────────────────────

    def _plan_actions(self, goal: str, url: str, content: str) -> List[WebAction]:
        prompt = f"""웹 태스크를 위한 액션 계획을 JSON 배열로 반환하세요.

목표: {goal}
현재 URL: {url}
페이지 내용 (앞부분): {content[:2000]}

JSON 배열 (최대 5개):
[
  {{"action_type": "navigate|click|type|extract|scroll", "target": "CSS선택자 또는 URL", "value": "입력값", "description": "설명"}}
]

JSON만 반환."""
        text_out = self._llm_call(prompt, 800)
        try:
            m = re.search(r'\[.*\]', text_out, re.DOTALL)
            data = json.loads(m.group()) if m else []
            return [WebAction(**a) for a in data if "action_type" in a]
        except Exception:
            return []

    def _extract_result(self, goal: str, content: str) -> str:
        prompt = f"""다음 웹 페이지 텍스트에서 목표와 관련된 핵심 정보를 추출하세요.

목표: {goal}
페이지 내용: {content[:3000]}

간결하고 구체적으로 추출하세요."""
        return self._llm_call(prompt, 800) or content[:500]

    def _save_task(self, task: WebTask):
        try:
            f = self.data_dir / "web_agent_tasks.json"
            tasks = []
            if f.exists():
                tasks = json.loads(f.read_text(encoding="utf-8"))
            tasks.append({
                "id": task.id, "goal": task.goal,
                "start_url": task.start_url,
                "success": task.success,
                "result_preview": task.result[:200],
                "duration": task.duration,
                "timestamp": task.timestamp,
            })
            f.write_text(json.dumps(tasks[-50:], ensure_ascii=False, indent=2),
                         encoding="utf-8")
        except Exception:
            pass

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "available": True,
            "mode": self._mode,
            "playwright_available": self._mode == "playwright",
            "tasks_total": len(self.tasks),
            "tasks_success": len([t for t in self.tasks if t.success]),
            "tasks_failed": len([t for t in self.tasks if not t.success and t.error]),
        }
