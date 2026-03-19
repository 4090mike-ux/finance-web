# JARVIS - Just A Rather Very Intelligent System

영화 아이언맨의 자비스를 현실적으로 구현한 소프트웨어 전용 AI 에이전트 시스템

## 시작하기

### 1. API 키 설정 (.env 파일)
```
ANTHROPIC_API_KEY=sk-ant-...    # Claude AI (권장)
OPENAI_API_KEY=sk-...           # GPT-4 (선택)
GITHUB_TOKEN=ghp_...            # GitHub API (선택)
```

### 2. 실행
```bash
python jarvis_app.py
# 브라우저에서 http://localhost:5001/jarvis 접속
```

## 주요 기능

### 채팅
- 자연어로 대화 (한국어/영어)
- 실시간 스트리밍 응답
- 마크다운 렌더링

### 도구 통합
| 도구 | 설명 |
|------|------|
| 🔍 웹 검색 | DuckDuckGo로 실시간 검색 |
| ⭐ GitHub | 최신 레포, 트렌드 검색 |
| 📄 ArXiv | 최신 AI/ML 논문 검색 |
| 📰 뉴스 | 최신 뉴스 수집 |
| ⚡ Python | 코드 실행 (샌드박스) |
| 💻 시스템 | CPU/메모리/프로세스 모니터링 |
| 🧠 메모리 | 장기 기억 (SQLite + ChromaDB) |
| 🎤 음성 | TTS/STT (선택사항) |

### 명령어
```
/search [query]   - 웹 검색
/github [query]   - GitHub 검색
/arxiv [query]    - 논문 검색
/system           - 시스템 정보
/code [code]      - Python 실행
/trends           - AI 트렌드
/memory           - 메모리 상태
/status           - JARVIS 상태
```

## 아키텍처

```
JARVIS Engine (core/jarvis_engine.py)
├── LLM Manager (llm/)
│   ├── Claude (Anthropic) - 기본
│   └── GPT-4 (OpenAI) - 폴백
├── Memory Manager (memory/)
│   ├── SQLite - 대화/지식 저장
│   └── ChromaDB - 벡터 시맨틱 검색
├── Computer Controller (computer/)
│   └── psutil - 시스템 모니터링
├── Web Intelligence (web/)
│   ├── DuckDuckGo - 검색
│   ├── GitHub API - 코드
│   └── ArXiv API - 논문
├── Code Executor (executor/)
│   └── 안전한 Python 실행
├── Agent Manager (agents/)
│   ├── ResearchAgent - 정보 수집
│   ├── CodeAgent - 코드 작업
│   ├── SystemAgent - 시스템
│   └── PlannerAgent - 계획 수립
└── Voice Interface (voice/)
    ├── pyttsx3 - TTS
    └── SpeechRecognition - STT
```

## 제약사항
- **소프트웨어 전용**: 물리적 장치 제어 없음
- **안전 우선**: 위험 명령어 차단
- **개인정보 보호**: 외부 서비스 최소 접근
