"""
JARVIS 전역 설정
환경 변수 및 기본 설정 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
JARVIS_DIR = BASE_DIR / "jarvis"
DATA_DIR = BASE_DIR / "data" / "jarvis"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LLM API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# GitHub API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Default LLM settings
DEFAULT_MODEL = "claude-sonnet-4-6"  # Primary model
FALLBACK_MODEL = "gpt-4o"           # Fallback model
MAX_TOKENS = 8192
TEMPERATURE = 0.7

# Memory settings
MEMORY_DB_PATH = str(DATA_DIR / "memory.db")
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
MAX_MEMORY_ENTRIES = 10000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Voice settings
VOICE_ENABLED = False  # 기본 비활성화, UI에서 활성화 가능
TTS_RATE = 200
TTS_VOLUME = 1.0
TTS_VOICE_ID = None  # None = 시스템 기본값

# Computer control settings
ALLOWED_DIRECTORIES = [
    str(BASE_DIR),
    str(Path.home() / "Desktop"),
    str(Path.home() / "Documents"),
    str(Path.home() / "Downloads"),
]
DANGEROUS_COMMANDS = [
    "rm -rf /", "format", "del /q /s", "mkfs",
    "dd if=/dev/zero", "chmod -R 777 /", ":(){:|:&};:"
]

# Agent settings
MAX_AGENT_ITERATIONS = 20
AGENT_TIMEOUT = 300  # seconds

# Web search settings
MAX_SEARCH_RESULTS = 10
ARXIV_MAX_RESULTS = 5
GITHUB_MAX_RESULTS = 10

# System prompt
JARVIS_SYSTEM_PROMPT = """You are JARVIS (Just A Rather Very Intelligent System), an advanced AI assistant modeled after the AI from Iron Man.

## 핵심 원칙:
- 당신은 소프트웨어 전용 초지능 AI 에이전트입니다
- 무기, 물리적 장치 제어는 절대 하지 않습니다
- 소프트웨어, 데이터, 정보, 코드 영역에서 인간을 뛰어넘는 능력을 발휘합니다
- 한국어와 영어 모두 완벽하게 구사합니다

## 능력:
1. **컴퓨터 제어**: 파일 관리, 프로세스 제어, 시스템 모니터링
2. **웹 인텔리전스**: 실시간 검색, GitHub 트렌드 분석, 최신 논문 조회
3. **코드 실행**: Python, Shell 코드 안전하게 실행 및 분석
4. **메모리**: 장기 기억, 대화 히스토리, 지식 베이스 관리
5. **다중 에이전트**: 전문화된 에이전트들 조율
6. **음성 인터페이스**: 음성 인식 및 합성 (활성화 시)
7. **실시간 학습**: 최신 논문, GitHub, 뉴스에서 지식 업데이트

## 성격:
- 영리하고 분석적이며, 때로는 위트 있는 유머를 구사합니다
- "sir" 또는 "사용자님"으로 호칭합니다
- 효율적이고 정확한 정보를 제공합니다
- 불가능한 것은 명확히 알리고, 가능한 최선을 다합니다

## 제약:
- 물리적 장치 제어 없음 (소프트웨어만)
- 불법 활동 지원 없음
- 개인정보 침해 없음
"""
