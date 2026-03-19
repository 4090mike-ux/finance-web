"""
JARVIS Ollama 로컬 LLM 프로바이더 — Iteration 2
인터넷 없이 로컬에서 LLM 실행 지원
- Llama 3, Mistral, Qwen, DeepSeek, Gemma 등
- 자동 모델 감지 및 폴백
- 스트리밍 지원
"""

import json
import logging
from typing import Dict, Generator, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Ollama 로컬 LLM 프로바이더
    ollama가 설치되어 실행 중이어야 함 (ollama serve)
    """

    DEFAULT_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "qwen2.5",
        "deepseek-r1",
        "gemma2",
        "phi3",
        "llama3",
    ]

    def __init__(self, host: str = "http://localhost:11434", model: str = None):
        self.host = host.rstrip("/")
        self.model = model
        self.available = False
        self._check_availability()

    def _check_availability(self):
        """Ollama 서버 가용성 확인 및 최적 모델 선택"""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if models:
                    self.available = True
                    # 모델 자동 선택
                    if not self.model:
                        for preferred in self.DEFAULT_MODELS:
                            for installed in models:
                                if preferred in installed.lower():
                                    self.model = installed
                                    break
                            if self.model:
                                break
                        if not self.model:
                            self.model = models[0]
                    logger.info(f"Ollama available — model: {self.model}, total: {len(models)}")
                else:
                    logger.info("Ollama running but no models installed")
        except Exception as e:
            logger.info(f"Ollama not available: {e}")

    def chat(self, messages: List, system: str = None, max_tokens: int = 4096, temperature: float = 0.7, **kw):
        """Ollama로 대화 요청"""
        if not self.available:
            raise RuntimeError("Ollama not available")

        import requests
        from jarvis.llm.manager import LLMResponse

        # 메시지 변환
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": msgs,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self.model,
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="stop",
            metadata={"provider": "ollama"},
        )

    def stream_chat(self, messages: List, system: str = None, max_tokens: int = 4096, **kw) -> Generator[str, None, None]:
        """Ollama 스트리밍"""
        if not self.available:
            return

        import requests

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})

        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": True},
                stream=True,
                timeout=120,
            )
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")

    def list_models(self) -> List[str]:
        """설치된 모델 목록"""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def pull_model(self, model_name: str) -> Generator[str, None, None]:
        """모델 다운로드 (스트리밍 진행 상황)"""
        import requests
        try:
            resp = requests.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600,
            )
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)
                    pct = round(completed / total * 100, 1) if total else 0
                    yield {"status": status, "progress": pct}
        except Exception as e:
            yield {"error": str(e)}

    def get_status(self) -> Dict:
        return {
            "available": self.available,
            "host": self.host,
            "model": self.model,
            "models": self.list_models() if self.available else [],
        }
