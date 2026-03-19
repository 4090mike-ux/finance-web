"""
JARVIS 음성 인터페이스
- TTS (Text-to-Speech): pyttsx3 사용
- STT (Speech-to-Text): SpeechRecognition 사용
- 웨이크워드 감지: "JARVIS" 또는 "자비스"
"""

import os
import io
import time
import queue
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class VoiceInterface:
    """JARVIS 음성 시스템"""

    def __init__(self, voice_rate: int = 200, voice_volume: float = 1.0):
        self.voice_rate = voice_rate
        self.voice_volume = voice_volume
        self.tts_engine = None
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        self.tts_queue = queue.Queue()
        self._tts_thread = None
        self.on_speech_callback: Optional[Callable] = None
        self.wake_words = ["jarvis", "자비스", "hey jarvis"]

        self._init_tts()
        self._init_stt()

    def _init_tts(self):
        """TTS 엔진 초기화"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.voice_rate)
            self.tts_engine.setProperty("volume", self.voice_volume)

            # 한국어 목소리 찾기
            voices = self.tts_engine.getProperty("voices")
            korean_voice = None
            for voice in voices:
                if "korean" in voice.name.lower() or "ko" in voice.id.lower():
                    korean_voice = voice.id
                    break

            if korean_voice:
                self.tts_engine.setProperty("voice", korean_voice)
                logger.info(f"Korean TTS voice set: {korean_voice}")
            else:
                logger.info("Korean voice not found, using default")

            # TTS 스레드 시작
            self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self._tts_thread.start()
            logger.info("TTS engine initialized")
        except Exception as e:
            logger.warning(f"TTS init failed: {e}")
            self.tts_engine = None

    def _init_stt(self):
        """STT 엔진 초기화"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.microphone = sr.Microphone()
            logger.info("STT engine initialized")
        except Exception as e:
            logger.warning(f"STT init failed: {e}")
            self.recognizer = None

    def _tts_worker(self):
        """TTS 큐 처리 워커"""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:
                    break
                if self.tts_engine:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def speak(self, text: str, blocking: bool = False):
        """텍스트를 음성으로 출력"""
        if not self.tts_engine:
            logger.debug(f"[TTS Disabled] {text}")
            return

        # 긴 텍스트는 문장으로 분할
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                self.tts_queue.put(sentence)

        if blocking:
            self.tts_queue.join()

    def _split_sentences(self, text: str, max_len: int = 200) -> list:
        """텍스트를 문장 단위로 분할"""
        import re
        # 문장 분리
        sentences = re.split(r'[.!?。]\s*', text)
        result = []
        for s in sentences:
            if len(s) > max_len:
                # 긴 문장은 쉼표로 분리
                parts = s.split(",")
                result.extend(parts)
            else:
                result.append(s)
        return [s for s in result if s.strip()]

    def listen_once(self, timeout: int = 5, phrase_timeout: int = 3) -> Optional[str]:
        """한 번 음성 입력 받기"""
        if not self.recognizer or not self.microphone:
            return None

        try:
            import speech_recognition as sr
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_timeout)

            # Google STT 사용 (무료)
            try:
                text = self.recognizer.recognize_google(audio, language="ko-KR")
                logger.info(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                logger.error(f"STT request error: {e}")
                return None
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return None

    def start_continuous_listening(self, callback: Callable):
        """지속적 음성 감지 시작"""
        if not self.recognizer or not self.microphone:
            logger.warning("STT not available")
            return

        self.on_speech_callback = callback
        self.is_listening = True
        thread = threading.Thread(target=self._listening_loop, daemon=True)
        thread.start()
        logger.info("Continuous listening started")

    def stop_listening(self):
        """음성 감지 중지"""
        self.is_listening = False
        logger.info("Listening stopped")

    def _listening_loop(self):
        """음성 감지 루프"""
        import speech_recognition as sr
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)

                try:
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    if text and self.on_speech_callback:
                        self.on_speech_callback(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
            except Exception:
                time.sleep(0.5)

    @property
    def is_available(self) -> bool:
        """음성 기능 사용 가능 여부"""
        return self.tts_engine is not None or self.recognizer is not None

    @property
    def tts_available(self) -> bool:
        return self.tts_engine is not None

    @property
    def stt_available(self) -> bool:
        return self.recognizer is not None

    def get_status(self) -> dict:
        """음성 시스템 상태"""
        return {
            "tts_available": self.tts_available,
            "stt_available": self.stt_available,
            "is_listening": self.is_listening,
            "wake_words": self.wake_words,
            "voice_rate": self.voice_rate,
        }
