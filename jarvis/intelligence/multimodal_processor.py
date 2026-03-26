"""
JARVIS 멀티모달 처리기 — Iteration 11
이미지, 문서, 오디오를 Claude Vision API로 분석한다

영감:
  - GPT-4V / Claude 3 Vision 멀티모달
  - OCR + 이미지 이해 (Optical Character Recognition)
  - 음성 → 텍스트 (Whisper-style STT)
  - 문서 이해 (Document Understanding)

핵심 개념:
  JARVIS는 텍스트 너머를 본다 — 이미지 속 정보를 읽고
  차트를 해석하고 손글씨를 인식하고 스크린샷을 분석한다
  모든 감각 채널을 통합해 완전한 상황 인식을 구현한다
"""

import os
import re
import base64
import json
import logging
import time
import threading
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════

@dataclass
class ModalityResult:
    """단일 모달리티 처리 결과"""
    modality: str                       # "image", "audio", "document", "chart"
    content_hash: str                   # 입력 콘텐츠 해시
    raw_description: str                # 원시 설명
    structured_data: Dict = field(default_factory=dict)   # 구조화된 추출 데이터
    confidence: float = 0.8             # 신뢰도
    processing_time: float = 0.0        # 처리 시간(초)
    tags: List[str] = field(default_factory=list)         # 자동 태그
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "modality": self.modality,
            "content_hash": self.content_hash,
            "raw_description": self.raw_description,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


@dataclass
class FusionResult:
    """멀티모달 융합 결과"""
    session_id: str
    modalities_processed: List[str]
    fused_understanding: str            # 통합된 이해
    key_entities: List[str]             # 핵심 개체
    action_items: List[str]             # 권장 행동
    confidence: float = 0.85
    timestamp: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════
# 멀티모달 처리기
# ════════════════════════════════════════════════════════════════

class MultimodalProcessor:
    """
    JARVIS 멀티모달 처리기
    이미지 / 차트 / 문서 / 오디오를 이해하고 텍스트로 변환한다
    """

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self._cache: Dict[str, ModalityResult] = {}       # 해시 기반 캐시
        self._stats = {
            "images_processed": 0,
            "charts_analyzed": 0,
            "documents_ocr": 0,
            "fusions_completed": 0,
            "cache_hits": 0,
        }
        self._lock = threading.Lock()

        # 지원 이미지 확장자
        self.SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        self.SUPPORTED_DOC_EXTS = {".pdf", ".txt", ".md", ".csv", ".json"}

        logger.info("MultimodalProcessor initialized — Claude Vision ready")

    # ── 이미지 처리 ──────────────────────────────────────────────

    def process_image(self, image_path: str, task: str = "describe") -> ModalityResult:
        """
        이미지 파일을 Claude Vision으로 분석
        task: "describe" | "ocr" | "chart" | "classify" | "custom:<prompt>"
        """
        start = time.time()
        path = Path(image_path)

        if not path.exists():
            return self._error_result("image", f"파일 없음: {image_path}")

        if path.suffix.lower() not in self.SUPPORTED_IMAGE_EXTS:
            return self._error_result("image", f"지원하지 않는 형식: {path.suffix}")

        # 이미지 → base64
        with open(path, "rb") as f:
            image_bytes = f.read()

        content_hash = hashlib.md5(image_bytes + task.encode()).hexdigest()[:12]

        # 캐시 확인
        cached = self._get_cache(content_hash)
        if cached:
            return cached

        # base64 인코딩
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        media_type = self._get_media_type(path.suffix)

        # 프롬프트 선택
        prompt = self._build_image_prompt(task)

        try:
            result_text = self._call_vision_api(b64_image, media_type, prompt)
        except Exception as e:
            logger.warning(f"Vision API 오류: {e}")
            result_text = f"이미지 분석 실패: {e}"

        # 구조화된 데이터 추출
        structured = self._extract_structured_info(result_text, task)
        tags = self._auto_tag(result_text)

        result = ModalityResult(
            modality="image",
            content_hash=content_hash,
            raw_description=result_text,
            structured_data=structured,
            confidence=0.9,
            processing_time=time.time() - start,
            tags=tags,
        )

        self._set_cache(content_hash, result)
        with self._lock:
            self._stats["images_processed"] += 1

        logger.info(f"Image processed: {path.name} [{result.processing_time:.1f}s]")
        return result

    def process_image_bytes(self, image_bytes: bytes, task: str = "describe",
                             media_type: str = "image/png") -> ModalityResult:
        """바이트 스트림으로 이미지 처리 (스크린샷, URL 다운로드 등)"""
        start = time.time()
        content_hash = hashlib.md5(image_bytes + task.encode()).hexdigest()[:12]

        cached = self._get_cache(content_hash)
        if cached:
            return cached

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = self._build_image_prompt(task)

        try:
            result_text = self._call_vision_api(b64_image, media_type, prompt)
        except Exception as e:
            result_text = f"이미지 분석 실패: {e}"

        structured = self._extract_structured_info(result_text, task)
        tags = self._auto_tag(result_text)

        result = ModalityResult(
            modality="image",
            content_hash=content_hash,
            raw_description=result_text,
            structured_data=structured,
            confidence=0.85,
            processing_time=time.time() - start,
            tags=tags,
        )
        self._set_cache(content_hash, result)
        with self._lock:
            self._stats["images_processed"] += 1
        return result

    def analyze_chart(self, image_path: str) -> ModalityResult:
        """차트/그래프 특화 분석"""
        return self.process_image(image_path, task="chart")

    def ocr_image(self, image_path: str) -> ModalityResult:
        """OCR — 이미지 속 텍스트 추출"""
        return self.process_image(image_path, task="ocr")

    # ── URL 이미지 처리 ──────────────────────────────────────────

    def process_image_url(self, url: str, task: str = "describe") -> ModalityResult:
        """URL에서 이미지 다운로드 후 분석"""
        if not REQUESTS_AVAILABLE:
            return self._error_result("image", "requests 패키지 없음")

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            media_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            return self.process_image_bytes(resp.content, task=task, media_type=media_type)
        except Exception as e:
            return self._error_result("image", f"URL 다운로드 실패: {e}")

    # ── 문서 처리 ────────────────────────────────────────────────

    def process_document(self, file_path: str, extract_mode: str = "summarize") -> ModalityResult:
        """
        문서 처리 (텍스트 파일, CSV, JSON, 마크다운)
        extract_mode: "summarize" | "key_points" | "entities" | "qa"
        """
        start = time.time()
        path = Path(file_path)

        if not path.exists():
            return self._error_result("document", f"파일 없음: {file_path}")

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return self._error_result("document", f"파일 읽기 실패: {e}")

        content_hash = hashlib.md5(text.encode() + extract_mode.encode()).hexdigest()[:12]
        cached = self._get_cache(content_hash)
        if cached:
            return cached

        # 텍스트가 너무 길면 앞부분만
        if len(text) > 8000:
            text = text[:8000] + "\n\n[... 이하 생략 ...]"

        prompt = self._build_document_prompt(extract_mode, text, path.suffix)

        try:
            result_text = self._call_text_api(prompt)
        except Exception as e:
            result_text = f"문서 처리 실패: {e}"

        structured = self._extract_document_structure(result_text, path.suffix)
        tags = self._auto_tag(result_text)

        result = ModalityResult(
            modality="document",
            content_hash=content_hash,
            raw_description=result_text,
            structured_data=structured,
            confidence=0.88,
            processing_time=time.time() - start,
            tags=tags,
        )
        self._set_cache(content_hash, result)
        with self._lock:
            self._stats["documents_ocr"] += 1
        return result

    # ── 멀티모달 융합 ────────────────────────────────────────────

    def fuse(self, results: List[ModalityResult], context: str = "") -> FusionResult:
        """
        여러 모달리티 결과를 통합하여 고차원 이해 생성
        예: 이미지 + 문서 + 차트를 동시에 이해
        """
        if not results:
            return FusionResult(
                session_id="empty",
                modalities_processed=[],
                fused_understanding="처리할 결과가 없습니다",
                key_entities=[],
                action_items=[],
            )

        modalities = [r.modality for r in results]
        descriptions = "\n\n".join([
            f"[{r.modality.upper()}]\n{r.raw_description}"
            for r in results
        ])

        prompt = f"""다음은 여러 정보 채널에서 수집된 데이터입니다:

{descriptions}

{'추가 컨텍스트: ' + context if context else ''}

이 모든 정보를 통합하여:
1. **핵심 이해**: 전체 상황을 2-3문장으로 요약
2. **핵심 개체**: 중요한 사람/사물/개념 목록 (콤마로 구분)
3. **권장 행동**: 다음에 취해야 할 행동 목록

JSON 형식으로 응답:
{{
  "fused_understanding": "...",
  "key_entities": ["...", "..."],
  "action_items": ["...", "..."]
}}"""

        try:
            result_text = self._call_text_api(prompt)
            parsed = self._parse_json_response(result_text)
        except Exception as e:
            logger.warning(f"Fusion 실패: {e}")
            parsed = {
                "fused_understanding": descriptions[:500],
                "key_entities": [],
                "action_items": [],
            }

        import uuid
        fusion = FusionResult(
            session_id=str(uuid.uuid4())[:8],
            modalities_processed=modalities,
            fused_understanding=parsed.get("fused_understanding", ""),
            key_entities=parsed.get("key_entities", []),
            action_items=parsed.get("action_items", []),
        )

        with self._lock:
            self._stats["fusions_completed"] += 1

        logger.info(f"Fusion complete: {modalities} → {len(fusion.key_entities)} entities")
        return fusion

    # ── Vision API 호출 ──────────────────────────────────────────

    def _call_vision_api(self, b64_image: str, media_type: str, prompt: str) -> str:
        """Claude Vision API 호출"""
        if not hasattr(self.llm, "_anthropic_client") and not hasattr(self.llm, "client"):
            # LLM Manager에서 Anthropic 클라이언트 접근
            raise RuntimeError("Anthropic 클라이언트 없음")

        # LLM Manager를 통해 Claude에 Vision 요청
        # LLMManager.generate_with_image() 또는 유사 메서드 활용
        if hasattr(self.llm, "generate_with_image"):
            return self.llm.generate_with_image(b64_image, media_type, prompt)

        # 직접 Anthropic SDK 사용 폴백
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
            message = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return message.content[0].text
        except Exception as e:
            raise RuntimeError(f"Vision API 호출 실패: {e}")

    def _call_text_api(self, prompt: str) -> str:
        """텍스트 전용 LLM 호출"""
        try:
            result = self.llm.generate(prompt, max_tokens=1024)
            if isinstance(result, dict):
                return result.get("content", str(result))
            return str(result)
        except Exception as e:
            raise RuntimeError(f"LLM 호출 실패: {e}")

    # ── 프롬프트 빌더 ────────────────────────────────────────────

    def _build_image_prompt(self, task: str) -> str:
        prompts = {
            "describe": "이 이미지를 한국어로 상세히 설명해주세요. 주요 내용, 색상, 구성 요소를 포함하세요.",
            "ocr": "이 이미지에서 모든 텍스트를 정확히 추출하세요. 텍스트만 출력하고 다른 설명은 하지 마세요.",
            "chart": (
                "이 차트/그래프를 분석해주세요:\n"
                "1. 차트 유형 (막대/선/파이 등)\n"
                "2. X축과 Y축의 의미\n"
                "3. 주요 데이터 포인트와 트렌드\n"
                "4. 핵심 인사이트 2-3가지\n"
                "5. 데이터에서 보이는 이상치나 패턴"
            ),
            "classify": "이 이미지를 분류하세요: 카테고리, 주제, 주요 객체, 감정/톤을 알려주세요.",
        }
        if task.startswith("custom:"):
            return task[7:]
        return prompts.get(task, prompts["describe"])

    def _build_document_prompt(self, mode: str, text: str, ext: str) -> str:
        mode_instructions = {
            "summarize": "다음 문서를 3-5개 핵심 문장으로 요약하세요.",
            "key_points": "다음 문서의 핵심 포인트를 불릿 리스트로 추출하세요.",
            "entities": "다음 문서에서 사람, 조직, 날짜, 장소, 숫자 등 핵심 개체를 추출하세요.",
            "qa": "다음 문서를 분석하여 가장 중요한 질문 5개와 그 답변을 제공하세요.",
        }
        instruction = mode_instructions.get(mode, mode_instructions["summarize"])
        return f"{instruction}\n\n문서 내용:\n{text}"

    # ── 유틸리티 ─────────────────────────────────────────────────

    def _extract_structured_info(self, text: str, task: str) -> Dict:
        """분석 결과에서 구조화된 정보 추출"""
        structured = {"raw_length": len(text)}
        if task == "chart":
            # 차트 분석에서 데이터 포인트 추출 시도
            numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
            structured["mentioned_values"] = numbers[:10]
        elif task == "ocr":
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            structured["line_count"] = len(lines)
            structured["word_count"] = len(text.split())
        return structured

    def _extract_document_structure(self, text: str, ext: str) -> Dict:
        return {
            "format": ext,
            "summary_length": len(text),
            "sections": len([l for l in text.split("\n") if l.startswith("#")]),
        }

    def _auto_tag(self, text: str) -> List[str]:
        """텍스트 내용 기반 자동 태그 생성"""
        text_lower = text.lower()
        tag_keywords = {
            "chart": ["chart", "graph", "axis", "data", "trend", "차트", "그래프"],
            "text": ["text", "word", "letter", "ocr", "텍스트"],
            "person": ["person", "people", "face", "사람", "얼굴"],
            "code": ["code", "function", "variable", "코드", "함수"],
            "financial": ["price", "stock", "revenue", "profit", "주식", "가격", "수익"],
            "scientific": ["research", "study", "paper", "연구", "논문"],
        }
        tags = []
        for tag, keywords in tag_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)
        return tags

    def _get_media_type(self, ext: str) -> str:
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return media_types.get(ext.lower(), "image/jpeg")

    def _error_result(self, modality: str, error_msg: str) -> ModalityResult:
        return ModalityResult(
            modality=modality,
            content_hash="error",
            raw_description=f"오류: {error_msg}",
            structured_data={"error": error_msg},
            confidence=0.0,
            processing_time=0.0,
            tags=["error"],
        )

    def _get_cache(self, key: str) -> Optional[ModalityResult]:
        result = self._cache.get(key)
        if result:
            with self._lock:
                self._stats["cache_hits"] += 1
        return result

    def _set_cache(self, key: str, value: ModalityResult):
        # 최대 100개 캐시
        if len(self._cache) >= 100:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value

    def _parse_json_response(self, text: str) -> Dict:
        """JSON 파싱 — 코드 블록 추출 포함"""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"raw": text}

    # ── 통계 ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                "cache_size": len(self._cache),
                "pil_available": PIL_AVAILABLE,
                "requests_available": REQUESTS_AVAILABLE,
            }

    def describe_capabilities(self) -> str:
        return (
            "MultimodalProcessor 능력:\n"
            "• 이미지 분석: 설명, OCR, 차트 해석, 분류\n"
            "• URL 이미지: 원격 이미지 다운로드 후 분석\n"
            "• 문서 처리: TXT, MD, CSV, JSON 요약/추출\n"
            "• 멀티모달 융합: 여러 소스 통합 이해\n"
            f"• 처리 현황: 이미지 {self._stats['images_processed']}개, "
            f"문서 {self._stats['documents_ocr']}개"
        )
