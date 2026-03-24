"""
scanner.py - 실시간 급등 자산 스캐너
- Binance API: 실시간 암호화폐 급등 코인 탐지
- 시뮬레이션: 미국 주식 시뮬레이션 급등주 탐지
- 통합 모멘텀 점수 계산
"""

from datetime import datetime
import logging

from .data_fetcher import (
    get_binance_top_movers,
    get_binance_klines,
    calculate_rsi_from_candles,
    get_simulated_top_movers,
)

logger = logging.getLogger(__name__)


class MomentumScanner:
    """
    통합 급등주/급등코인 스캐너
    - 코인: Binance API 실시간 데이터
    - 주식: 시뮬레이션 데이터
    """

    def __init__(self, min_change_pct: float = 3.0,
                 min_volume_usdt: float = 1_000_000):
        self.min_change_pct = min_change_pct
        self.min_volume_usdt = min_volume_usdt
        self.scan_results = []
        self.last_scan_time = None

    def scan(self, mode: str = "both") -> list:
        """
        스캔 실행
        mode: "crypto" (코인만), "stock" (주식만), "both" (모두)
        """
        results = []

        if mode in ("crypto", "both"):
            crypto_results = self._scan_crypto()
            results.extend(crypto_results)
            logger.info(f"코인 스캔: {len(crypto_results)}개 발견")

        if mode in ("stock", "both"):
            stock_results = self._scan_stocks()
            results.extend(stock_results)
            logger.info(f"주식 스캔: {len(stock_results)}개 발견")

        # 모멘텀 점수 계산 및 정렬
        for item in results:
            if item.get("momentum_score", 0) == 0:
                item["momentum_score"] = self._calculate_score(item)

        results.sort(key=lambda x: x["momentum_score"], reverse=True)
        self.scan_results = results
        self.last_scan_time = datetime.now().isoformat()
        return results

    def _scan_crypto(self) -> list:
        """Binance 암호화폐 급등 코인 스캔"""
        try:
            movers = get_binance_top_movers(
                min_volume_usdt=self.min_volume_usdt,
                min_change_pct=self.min_change_pct,
                top_n=30,
            )

            # RSI 계산 (상위 20개만)
            enhanced = []
            for m in movers[:20]:
                try:
                    candles = get_binance_klines(m["symbol"], interval="1h", limit=20)
                    if candles:
                        rsi = calculate_rsi_from_candles(candles)
                        m["rsi"] = rsi
                        # 1시간 변화율 계산
                        if len(candles) >= 2:
                            hour_change = ((candles[-1]["close"] - candles[-2]["open"]) /
                                         candles[-2]["open"]) * 100
                            m["hour_change_pct"] = round(hour_change, 2)
                        # 거래량 비율 (최근 vs 이전)
                        if len(candles) >= 10:
                            recent_vol = sum(c["volume"] for c in candles[-3:]) / 3
                            avg_vol = sum(c["volume"] for c in candles[:-3]) / max(len(candles) - 3, 1)
                            m["volume_ratio"] = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0
                    m["momentum_score"] = self._calculate_score(m)
                    enhanced.append(m)
                except Exception as e:
                    logger.debug(f"RSI 계산 오류 ({m['symbol']}): {e}")
                    m["momentum_score"] = self._calculate_score(m)
                    enhanced.append(m)

            # RSI 계산 못한 나머지
            for m in movers[20:]:
                m["momentum_score"] = self._calculate_score(m)
                enhanced.append(m)

            return enhanced

        except Exception as e:
            logger.error(f"코인 스캔 오류: {e}")
            return []

    def _scan_stocks(self) -> list:
        """미국 주식 시뮬레이션 스캔"""
        try:
            movers = get_simulated_top_movers(
                min_change_pct=self.min_change_pct,
                top_n=15,
            )
            for m in movers:
                m["momentum_score"] = self._calculate_score(m)
            return movers
        except Exception as e:
            logger.error(f"주식 스캔 오류: {e}")
            return []

    def _calculate_score(self, data: dict) -> float:
        """
        모멘텀 점수 계산
        - 당일 변화율 (가중치 높음)
        - 1시간 변화율
        - 거래량 비율
        - RSI 적정 범위 보너스
        """
        change = data.get("day_change_pct", 0)
        hour_change = data.get("hour_change_pct", 0)
        vol_ratio = data.get("volume_ratio", 1.0)
        rsi = data.get("rsi", 50.0)

        # 기본 점수
        score = abs(change) * 2.0 + abs(hour_change) * 3.0

        # 거래량 보너스
        if vol_ratio > 1.5:
            score += vol_ratio * 1.5

        # RSI 적정 범위 보너스 (30~70: 추세 중간)
        if 30 < rsi < 70:
            score *= 1.1

        # RSI 극단값 감점
        if rsi > 85 or rsi < 15:
            score *= 0.7

        # 방향성 보너스 (상승 방향 우대)
        if change > 0:
            score *= 1.1

        return round(score, 2)

    def get_top_movers(self, n: int = 10) -> list:
        """상위 N개 급등 종목 반환"""
        return self.scan_results[:n]
