"""
data_fetcher.py - 시장 데이터 수집 모듈
- Binance API: 실시간 암호화폐 데이터 (API 키 불필요)
- Alpha Vantage: 미국 주식 데이터 (무료 API 키 필요)
- 시뮬레이션 모드: API 없이 랜덤 시장 데이터 생성
"""

import requests
import json
import time
import random
import math
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Binance API 기본 URL
BINANCE_BASE = "https://api.binance.com/api/v3"

# Alpha Vantage 기본 URL
AV_BASE = "https://www.alphavantage.co/query"

# HTTP 세션 (SSL 우회)
_session = None


def get_session():
    """requests 세션 반환 (SSL 우회 설정)"""
    global _session
    if _session is None:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _session = requests.Session()
        _session.verify = False
        _session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    return _session


# ============================================================
# Binance 암호화폐 데이터
# ============================================================

def get_binance_ticker_24h(symbol: str) -> Optional[dict]:
    """Binance 24시간 티커 데이터"""
    try:
        url = f"{BINANCE_BASE}/ticker/24hr"
        res = get_session().get(url, params={"symbol": symbol.upper()}, timeout=10)
        res.raise_for_status()
        data = res.json()
        return {
            "symbol": data["symbol"],
            "price": float(data["lastPrice"]),
            "open": float(data["openPrice"]),
            "high": float(data["highPrice"]),
            "low": float(data["lowPrice"]),
            "day_change_pct": float(data["priceChangePercent"]),
            "volume": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),  # USDT 거래량
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.debug(f"Binance 티커 오류 ({symbol}): {e}")
        return None


def get_binance_top_movers(min_volume_usdt: float = 1_000_000,
                            min_change_pct: float = 3.0,
                            top_n: int = 20) -> list:
    """
    Binance에서 급등 코인 목록 조회
    Args:
        min_volume_usdt: 최소 24시간 거래량 (USDT)
        min_change_pct: 최소 변화율 (%)
        top_n: 상위 N개 반환
    """
    try:
        url = f"{BINANCE_BASE}/ticker/24hr"
        res = get_session().get(url, timeout=15)
        res.raise_for_status()
        all_tickers = res.json()

        # USDT 페어만 필터링
        usdt_pairs = [t for t in all_tickers if t["symbol"].endswith("USDT")]

        movers = []
        for t in usdt_pairs:
            try:
                change_pct = float(t["priceChangePercent"])
                quote_vol = float(t["quoteVolume"])
                price = float(t["lastPrice"])

                # 최소 필터
                if abs(change_pct) < min_change_pct:
                    continue
                if quote_vol < min_volume_usdt:
                    continue
                if price < 0.0001:  # 극소가 코인 제외
                    continue

                movers.append({
                    "symbol": t["symbol"],
                    "price": price,
                    "open": float(t["openPrice"]),
                    "high": float(t["highPrice"]),
                    "low": float(t["lowPrice"]),
                    "day_change_pct": change_pct,
                    "volume": float(t["volume"]),
                    "quote_volume": quote_vol,
                    "volume_ratio": 1.0,  # 실시간이므로 1로 설정
                    "rsi": 50.0,          # 별도 계산 필요
                    "atr_pct": abs(float(t["highPrice"]) - float(t["lowPrice"])) / price * 100,
                    "momentum_score": 0.0,
                    "source": "binance",
                    "timestamp": datetime.now().isoformat(),
                })
            except (ValueError, KeyError):
                continue

        # 변화율 절대값 기준 정렬
        movers.sort(key=lambda x: abs(x["day_change_pct"]), reverse=True)
        return movers[:top_n]

    except Exception as e:
        logger.error(f"Binance 급등 코인 조회 오류: {e}")
        return []


def get_binance_klines(symbol: str, interval: str = "1h",
                        limit: int = 50) -> list:
    """
    Binance 캔들스틱 데이터 조회
    interval: 1m, 5m, 15m, 1h, 4h, 1d
    """
    try:
        url = f"{BINANCE_BASE}/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        res = get_session().get(url, params=params, timeout=10)
        res.raise_for_status()
        raw = res.json()

        candles = []
        for c in raw:
            candles.append({
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        return candles
    except Exception as e:
        logger.debug(f"Binance 캔들 오류 ({symbol}): {e}")
        return []


def calculate_rsi_from_candles(candles: list, period: int = 14) -> float:
    """캔들 데이터에서 RSI 계산"""
    if len(candles) < period + 1:
        return 50.0
    closes = [c["close"] for c in candles]
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ============================================================
# 미국 주식 시뮬레이션 모드
# ============================================================

# 시뮬레이션용 기준 가격 (실제 시장 가격 근사값)
STOCK_BASE_PRICES = {
    "NVDA": 110.0, "AMD": 105.0, "TSLA": 250.0, "META": 590.0,
    "AMZN": 195.0, "GOOGL": 170.0, "AAPL": 225.0, "MSFT": 415.0,
    "PLTR": 85.0, "IONQ": 28.0, "SMCI": 42.0, "ARM": 145.0,
    "AVGO": 195.0, "MRVL": 72.0, "MU": 95.0, "SOFI": 13.0,
    "RIVN": 12.0, "MARA": 15.0, "RIOT": 9.0, "COIN": 225.0,
    "HOOD": 42.0, "RBLX": 55.0, "SNAP": 11.0, "PINS": 35.0,
    "TQQQ": 68.0, "SOXL": 22.0, "UPRO": 89.0,
}

# 시뮬레이션 상태 유지
_sim_prices = {}
_sim_last_update = {}


def get_simulated_stock_data(symbol: str) -> dict:
    """
    미국 주식 시뮬레이션 데이터 생성
    실제 시장 특성(랜덤 워크, 모멘텀, 변동성)을 시뮬레이션
    """
    global _sim_prices, _sim_last_update

    base_price = STOCK_BASE_PRICES.get(symbol, 50.0)
    now = time.time()

    # 초기화 또는 오래된 경우
    if symbol not in _sim_prices:
        # 장 시간 기준 랜덤 개장가
        open_variation = random.gauss(0, 0.015)  # ±1.5% 변동
        _sim_prices[symbol] = {
            "open": base_price * (1 + open_variation),
            "current": base_price * (1 + open_variation),
            "high": base_price * (1 + open_variation),
            "low": base_price * (1 + open_variation),
            "volume": random.randint(1_000_000, 50_000_000),
            "momentum": random.gauss(0, 0.005),  # 모멘텀 팩터
        }

    state = _sim_prices[symbol]
    last_update = _sim_last_update.get(symbol, 0)
    elapsed = now - last_update

    if elapsed > 30:  # 30초마다 가격 업데이트
        # 모멘텀 기반 랜덤 워크
        momentum = state["momentum"]
        noise = random.gauss(0, 0.003)
        momentum_decay = 0.95  # 모멘텀 감쇠

        new_momentum = momentum * momentum_decay + noise
        # 급등 이벤트 (1% 확률)
        if random.random() < 0.01:
            new_momentum += random.choice([-1, 1]) * random.uniform(0.02, 0.08)

        price_change = new_momentum * state["current"]
        new_price = max(state["current"] + price_change, 0.01)

        state["current"] = new_price
        state["momentum"] = new_momentum
        state["high"] = max(state["high"], new_price)
        state["low"] = min(state["low"], new_price)
        state["volume"] += random.randint(1000, 50000)
        _sim_last_update[symbol] = now

    current = state["current"]
    open_price = state["open"]
    day_change_pct = ((current - open_price) / open_price) * 100

    # 거래량 비율 (시뮬레이션)
    avg_volume = STOCK_BASE_PRICES.get(symbol, 50.0) * 200000
    volume_ratio = state["volume"] / avg_volume if avg_volume > 0 else 1.0

    # RSI 시뮬레이션
    rsi = 50 + day_change_pct * 2 + random.gauss(0, 5)
    rsi = max(10, min(90, rsi))

    return {
        "symbol": symbol,
        "price": round(current, 2),
        "open": round(open_price, 2),
        "day_change_pct": round(day_change_pct, 2),
        "hour_change_pct": round(state["momentum"] * 60, 2),  # 시간 변화 근사
        "volume": int(state["volume"]),
        "avg_volume": int(avg_volume),
        "volume_ratio": round(volume_ratio, 2),
        "rsi": round(rsi, 1),
        "atr_pct": round(abs(state["high"] - state["low"]) / current * 100, 2),
        "source": "simulation",
        "timestamp": datetime.now().isoformat(),
    }


def get_simulated_top_movers(min_change_pct: float = 2.0, top_n: int = 10) -> list:
    """미국 주식 시뮬레이션 급등주 목록"""
    results = []
    for symbol in STOCK_BASE_PRICES.keys():
        data = get_simulated_stock_data(symbol)
        if abs(data["day_change_pct"]) >= min_change_pct:
            results.append(data)

    results.sort(key=lambda x: abs(x["day_change_pct"]), reverse=True)
    return results[:top_n]
