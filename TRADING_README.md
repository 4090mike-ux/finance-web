# AI 트레이딩 시스템 사용법

## 빠른 시작

```bash
# 1. 패키지 설치 (처음 한 번만)
uv pip install -r requirements.txt

# 2. 트레이딩 앱 실행
uv run python trading_app.py

# 3. 브라우저에서 접속
# http://localhost:5001/trading/
```

## 기능 소개

### 실시간 데이터
- **Binance API**: 무료, API 키 불필요, 전 세계 암호화폐 실시간 데이터
- **미국 주식**: 시뮬레이션 모드 (API 키 없이 테스트 가능)

### AI 트레이딩 전략
- **모멘텀 탐지**: 24시간 ±3% 이상 급등/급락 종목 스캔
- **거래량 분석**: 평균 거래량 대비 급증 감지
- **RSI 필터**: 과열/과매도 구간 진입 제한
- **자동 매도**:
  - 목표 수익률 달성 (기본 +5%)
  - 손절 실행 (기본 -2%)
  - 트레일링 스탑 (최고가 대비 -1.5%)
  - 4시간 시간 제한

### 적응형 AI
- 10번 거래마다 자동 파라미터 조정
- 승률 낮으면: 진입 조건 강화, 리스크 축소
- 승률 높으면: 투자 비중 증가, 목표 수익 상향

## API 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| GET /trading/ | 대시보드 |
| GET /trading/api/status | 전체 상태 조회 |
| POST /trading/api/start | 자동매매 시작 |
| POST /trading/api/stop | 자동매매 정지 |
| POST /trading/api/scan | 즉시 스캔 실행 |
| POST /trading/api/portfolio/reset | 포트폴리오 초기화 |
| POST /trading/api/trade/buy | 수동 매수 |
| POST /trading/api/trade/sell | 수동 매도 |
| GET /trading/api/price/<symbol> | 종목 현재가 |

## 실제 API 키 설정 (선택)

`.env` 파일에 추가:
```
# Alpha Vantage (미국 주식 실제 데이터)
# https://www.alphavantage.co/support/#api-key 에서 무료 발급
ALPHA_VANTAGE_KEY=your_key_here

# Binance (이미 무료로 사용 가능, 고급 기능용)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
```

## 수익률 목표

- **단기 목표**: 일 5~10% (모의 트레이딩)
- **중기 목표**: 주 50% 이상
- **장기 목표**: 100% 이상 (AI 적응 후)

## 주의사항

⚠️ 이 시스템은 **모의 트레이딩** 전용입니다.
실제 자금 투자 전 충분한 테스트를 권장합니다.
암호화폐 및 주식 투자에는 원금 손실 위험이 있습니다.
