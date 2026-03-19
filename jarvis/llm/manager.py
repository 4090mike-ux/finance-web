"""
LLM Manager - 다중 LLM 모델 통합 관리자
Claude (Anthropic), OpenAI GPT 모델들을 통합하여 최적의 응답 제공
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any] = None


class ClaudeProvider:
    """Anthropic Claude API 제공자"""

    def __init__(self, api_key: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.available = True
        logger.info("Claude provider initialized")

    def chat(
        self,
        messages: List[Message],
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system: str = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Claude API로 대화 요청"""
        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    def stream_chat(
        self,
        messages: List[Message],
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        system: str = None,
    ) -> Generator[str, None, None]:
        """스트리밍 응답"""
        import anthropic

        anthropic_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider:
    """OpenAI API 제공자"""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
            logger.info("OpenAI provider initialized")
        except Exception as e:
            self.available = False
            logger.warning(f"OpenAI provider failed: {e}")

    def chat(
        self,
        messages: List[Message],
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: str = None,
        stream: bool = False,
    ) -> LLMResponse:
        """OpenAI API로 대화 요청"""
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})

        for m in messages:
            oai_messages.append({"role": m.role, "content": m.content})

        response = self.client.chat.completions.create(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
        )


class LLMManager:
    """
    멀티 LLM 관리자
    - Claude를 기본으로 사용
    - OpenAI를 폴백으로 사용
    - 모델 자동 선택 및 폴백 처리
    """

    def __init__(self, anthropic_key: str = "", openai_key: str = ""):
        self.providers = {}
        self.primary = None
        self.fallback = None

        # Claude 초기화
        if anthropic_key:
            try:
                self.providers["claude"] = ClaudeProvider(anthropic_key)
                self.primary = "claude"
                logger.info("Primary LLM: Claude")
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")

        # OpenAI 초기화
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
                if not self.primary:
                    self.primary = "openai"
                else:
                    self.fallback = "openai"
                logger.info("OpenAI provider ready")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")

        if not self.primary:
            logger.warning("No LLM provider available! Using mock responses.")

    def chat(
        self,
        messages: List[Message],
        model: str = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        system: str = None,
        provider: str = None,
    ) -> LLMResponse:
        """메인 채팅 함수 - 자동 폴백 포함"""

        # Mock response if no provider
        if not self.primary:
            return self._mock_response(messages)

        # Provider 선택
        target_provider = provider or self.primary

        try:
            p = self.providers.get(target_provider)
            if not p:
                raise ValueError(f"Provider {target_provider} not found")

            if target_provider == "claude":
                m = model or "claude-sonnet-4-6"
                return p.chat(messages, model=m, max_tokens=max_tokens,
                             temperature=temperature, system=system)
            elif target_provider == "openai":
                m = model or "gpt-4o"
                return p.chat(messages, model=m, max_tokens=max_tokens,
                             temperature=temperature, system=system)

        except Exception as e:
            logger.error(f"Primary provider {target_provider} failed: {e}")

            # Fallback
            if self.fallback and self.fallback != target_provider:
                try:
                    logger.info(f"Falling back to {self.fallback}")
                    p = self.providers[self.fallback]
                    return p.chat(messages, max_tokens=max_tokens,
                                 temperature=temperature, system=system)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")

            return self._mock_response(messages, error=str(e))

    def stream_chat(
        self,
        messages: List[Message],
        system: str = None,
    ) -> Generator[str, None, None]:
        """스트리밍 채팅"""
        if "claude" in self.providers:
            yield from self.providers["claude"].stream_chat(
                messages, system=system
            )
        else:
            response = self.chat(messages, system=system)
            yield response.content

    def _mock_response(self, messages: List[Message], error: str = None) -> LLMResponse:
        """API key 없을 때 Mock 응답"""
        if error:
            content = f"[JARVIS Mock] API Error: {error}. 환경변수 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY를 설정해주세요."
        else:
            last_msg = messages[-1].content if messages else ""
            content = f"[JARVIS Mock] '{last_msg[:50]}...' - API key가 없어 실제 응답을 생성할 수 없습니다. .env 파일에 ANTHROPIC_API_KEY를 설정해주세요."

        return LLMResponse(
            content=content,
            model="mock",
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="mock",
        )

    @property
    def available_models(self) -> Dict[str, List[str]]:
        """사용 가능한 모델 목록"""
        models = {}
        if "claude" in self.providers:
            models["claude"] = [
                "claude-opus-4-6",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ]
        if "openai" in self.providers:
            models["openai"] = ["gpt-4o", "gpt-4o-mini", "o1-mini"]
        return models
