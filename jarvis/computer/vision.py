"""
JARVIS 컴퓨터 비전 모듈 — Iteration 2
- 스크린샷 캡처
- Claude Vision API로 화면 분석
- 특정 영역 분석
- OCR (텍스트 추출)
- 화면 내 UI 요소 감지
"""

import io
import base64
import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class VisionSystem:
    """
    JARVIS 비전 시스템
    화면을 캡처하고 Claude Vision으로 분석
    """

    def __init__(self, llm_manager=None):
        self.llm = llm_manager
        self._pil_available = self._check_pil()
        self._screenshot_available = self._check_screenshot()
        logger.info(f"VisionSystem initialized — PIL={self._pil_available}, Screenshot={self._screenshot_available}")

    def _check_pil(self) -> bool:
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def _check_screenshot(self) -> bool:
        try:
            import PIL.ImageGrab
            return True
        except Exception:
            try:
                import pyautogui
                return True
            except Exception:
                return False

    # ── 스크린샷 ───────────────────────────────────────────────
    def capture_screen(self, region: Tuple[int, int, int, int] = None, save_path: str = None) -> Optional[bytes]:
        """
        화면 캡처
        region: (left, top, right, bottom) — None이면 전체 화면
        """
        try:
            # 방법 1: PIL ImageGrab
            try:
                from PIL import ImageGrab
                img = ImageGrab.grab(bbox=region)
            except Exception:
                # 방법 2: pyautogui
                import pyautogui
                if region:
                    left, top, right, bottom = region
                    img_arr = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
                else:
                    img_arr = pyautogui.screenshot()
                from PIL import Image
                img = img_arr

            # PNG 바이트로 변환
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()

            if save_path:
                with open(save_path, "wb") as f:
                    f.write(png_bytes)
                logger.info(f"Screenshot saved: {save_path}")

            return png_bytes

        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None

    def capture_and_encode(self, region: Tuple = None) -> Optional[str]:
        """스크린샷을 base64로 인코딩 (Claude Vision API용)"""
        png_bytes = self.capture_screen(region=region)
        if png_bytes:
            return base64.standard_b64encode(png_bytes).decode("utf-8")
        return None

    # ── Claude Vision 분석 ─────────────────────────────────────
    def analyze_screen(self, question: str = "화면에 무엇이 있나요?", region: Tuple = None) -> Dict:
        """현재 화면을 Claude Vision으로 분석"""
        if not self.llm:
            return {"success": False, "error": "LLM 없음"}

        img_b64 = self.capture_and_encode(region=region)
        if not img_b64:
            return {"success": False, "error": "스크린샷 캡처 실패"}

        try:
            import anthropic
            client = None
            for provider in self.llm.providers.values():
                if hasattr(provider, "client") and isinstance(provider.client, anthropic.Anthropic):
                    client = provider.client
                    break

            if not client:
                return {"success": False, "error": "Claude 클라이언트 없음"}

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }],
            )
            return {
                "success": True,
                "analysis": response.content[0].text,
                "timestamp": datetime.now().isoformat(),
                "region": region,
            }
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def analyze_image_file(self, file_path: str, question: str = "이 이미지를 분석해주세요.") -> Dict:
        """이미지 파일 분석"""
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

            # 파일 형식 감지
            media_type = "image/png"
            if file_path.lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            elif file_path.lower().endswith(".gif"):
                media_type = "image/gif"
            elif file_path.lower().endswith(".webp"):
                media_type = "image/webp"

            if not self.llm:
                return {"success": False, "error": "LLM 없음"}

            import anthropic
            client = None
            for provider in self.llm.providers.values():
                if hasattr(provider, "client") and isinstance(provider.client, anthropic.Anthropic):
                    client = provider.client
                    break

            if not client:
                return {"success": False, "error": "Claude 클라이언트 없음"}

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                        },
                        {"type": "text", "text": question},
                    ],
                }],
            )
            return {"success": True, "analysis": response.content[0].text, "file": file_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def extract_text_from_screen(self, region: Tuple = None) -> Dict:
        """화면에서 텍스트 추출 (OCR via Vision)"""
        return self.analyze_screen(
            question="이 화면의 모든 텍스트를 정확하게 추출해주세요. 형식을 최대한 유지하세요.",
            region=region,
        )

    def detect_ui_elements(self, region: Tuple = None) -> Dict:
        """UI 요소 감지 (버튼, 입력창, 메뉴 등)"""
        return self.analyze_screen(
            question="""화면의 UI 요소를 분석하세요:
1. 버튼 위치 및 텍스트
2. 입력 필드
3. 메뉴 및 드롭다운
4. 현재 활성 창/앱
5. 중요 텍스트 콘텐츠
JSON 형식으로 반환하세요.""",
            region=region,
        )

    def compare_screens(self, before_bytes: bytes, after_bytes: bytes) -> Dict:
        """두 화면 비교 (변경사항 감지)"""
        if not self.llm:
            return {"success": False, "error": "LLM 없음"}
        try:
            b64_before = base64.standard_b64encode(before_bytes).decode()
            b64_after = base64.standard_b64encode(after_bytes).decode()

            import anthropic
            client = None
            for provider in self.llm.providers.values():
                if hasattr(provider, "client") and isinstance(provider.client, anthropic.Anthropic):
                    client = provider.client
                    break

            if not client:
                return {"success": False, "error": "Claude 클라이언트 없음"}

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_before}},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_after}},
                        {"type": "text", "text": "첫 번째 이미지와 두 번째 이미지의 차이점을 상세히 분석하세요."},
                    ],
                }],
            )
            return {"success": True, "changes": response.content[0].text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict:
        return {
            "available": self._screenshot_available,
            "pil": self._pil_available,
            "vision_api": self.llm is not None,
        }
