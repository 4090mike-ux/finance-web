"""
engine.py - 메인 트레이딩 엔진
스캐너 + 전략 + 포트폴리오를 통합하는 핵심 모듈
백그라운드에서 주기적으로 실행되며 자동 매매 신호를 생성합니다.
"""

import threading
import time
import logging
from datetime import datetime
from typing import Optional

from .scanner import MomentumScanner
from .strategies import AdaptiveStrategy
from .portfolio import PaperTradingPortfolio

logger = logging.getLogger(__name__)

# 싱글톤 엔진 인스턴스
_engine_instance: Optional["TradingEngine"] = None


def get_engine() -> "TradingEngine":
    """전역 트레이딩 엔진 인스턴스 반환"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TradingEngine()
    return _engine_instance


class TradingEngine:
    """
    AI 트레이딩 엔진
    - 백그라운드 스캔 (5분 주기)
    - 자동 매수/매도 신호 실행
    - 실시간 포트폴리오 추적
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.scanner = MomentumScanner()
        self.strategy = AdaptiveStrategy()
        self.portfolio = PaperTradingPortfolio(initial_capital)

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 포지션별 최고가 추적 (트레일링 스탑용)
        self._position_highs = {}

        # 최근 스캔 결과 캐시
        self.last_scan_results = []
        self.last_scan_time = None
        self.scan_count = 0
        self.error_log = []

        logger.info(f"트레이딩 엔진 초기화: 초기 자본 ${initial_capital:,.2f}")

    def start(self, scan_interval: int = 300):
        """백그라운드 트레이딩 시작 (기본 5분 주기)"""
        if self.running:
            logger.warning("엔진이 이미 실행 중입니다.")
            return

        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(scan_interval,),
            daemon=True,
            name="TradingEngine",
        )
        self._thread.start()
        logger.info(f"트레이딩 엔진 시작 (스캔 주기: {scan_interval}초)")

    def stop(self):
        """엔진 정지"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("트레이딩 엔진 정지")

    def _run_loop(self, interval: int):
        """메인 트레이딩 루프"""
        while self.running:
            try:
                self._scan_and_trade()
            except Exception as e:
                error_msg = f"트레이딩 루프 오류: {e}"
                logger.error(error_msg)
                self.error_log.append({
                    "time": datetime.now().isoformat(),
                    "error": str(e),
                })
                self.error_log = self.error_log[-20:]  # 최근 20개 유지

            # 다음 스캔까지 대기
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)

    def _scan_and_trade(self):
        """스캔 실행 및 매매 결정"""
        with self._lock:
            self.scan_count += 1
            logger.info(f"=== 스캔 #{self.scan_count} 시작 ({datetime.now().strftime('%H:%M:%S')}) ===")

            # 1. 급등주 스캔
            scan_results = self.scanner.scan()
            self.last_scan_results = scan_results
            self.last_scan_time = datetime.now().isoformat()

            if not scan_results:
                logger.info("급등 종목 없음")
                return

            # 2. 현재 보유 포지션 점검 (매도 신호 확인)
            self._check_exit_signals(scan_results)

            # 3. 신규 진입 신호 확인
            self._check_entry_signals(scan_results)

    def _check_exit_signals(self, scan_results: list):
        """보유 포지션 매도 조건 확인"""
        if not self.portfolio.positions:
            return

        # 스캔 결과를 딕셔너리로 변환 (빠른 조회)
        price_map = {r["symbol"]: r["price"] for r in scan_results}

        positions_to_check = list(self.portfolio.positions.items())
        for symbol, pos in positions_to_check:
            current_price = price_map.get(symbol)
            if current_price is None:
                # 스캔 결과에 없는 종목은 yfinance로 직접 조회
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    info = ticker.fast_info
                    current_price = float(info.last_price)
                except Exception:
                    continue

            # 최고가 업데이트
            if symbol not in self._position_highs:
                self._position_highs[symbol] = pos["avg_price"]
            if current_price > self._position_highs[symbol]:
                self._position_highs[symbol] = current_price

            # 매도 신호 확인
            sell_signal = self.strategy.should_sell(
                symbol, current_price, pos,
                highest_price=self._position_highs.get(symbol),
            )

            if sell_signal["signal"]:
                result = self.portfolio.sell(
                    symbol, current_price,
                    reason=sell_signal["reason"],
                )
                if result["success"]:
                    self.strategy.update_performance(result["pnl"])
                    if symbol in self._position_highs:
                        del self._position_highs[symbol]
                    logger.info(f"매도 실행: {symbol} @ ${current_price:.2f} | {sell_signal['reason']}")

    def _check_entry_signals(self, scan_results: list):
        """신규 매수 신호 확인"""
        current_prices = {r["symbol"]: r["price"] for r in scan_results}
        portfolio_value = self.portfolio.get_current_value(current_prices)
        open_positions = len(self.portfolio.positions)

        for stock_data in scan_results[:10]:  # 상위 10개만 검토
            symbol = stock_data["symbol"]

            # 이미 보유 중인 종목 스킵
            if symbol in self.portfolio.positions:
                continue

            buy_signal = self.strategy.should_buy(
                stock_data, portfolio_value, open_positions,
            )

            if buy_signal["signal"]:
                price = stock_data["price"]
                result = self.portfolio.buy(
                    symbol, price,
                    invest_pct=buy_signal["invest_pct"],
                    reason=buy_signal["reason"],
                )
                if result["success"]:
                    open_positions += 1
                    logger.info(f"매수 실행: {symbol} @ ${price:.2f} | {buy_signal['reason']}")

                # 최대 포지션 도달 시 중단
                if open_positions >= self.strategy.params["max_positions"]:
                    break

    def get_status(self) -> dict:
        """엔진 전체 상태 조회"""
        with self._lock:
            # 현재가 맵
            current_prices = {r["symbol"]: r["price"] for r in self.last_scan_results}

            return {
                "engine": {
                    "running": self.running,
                    "scan_count": self.scan_count,
                    "last_scan_time": self.last_scan_time,
                },
                "portfolio": self.portfolio.get_pnl_summary(current_prices),
                "positions": self.portfolio.get_positions_detail(current_prices),
                "strategy": self.strategy.get_status(),
                "top_movers": self.last_scan_results[:10],
                "recent_trades": self.portfolio.trade_history[-20:],
                "errors": self.error_log[-5:],
            }

    def manual_scan(self, execute_trades: bool = True) -> list:
        """
        수동 스캔 실행 (API에서 호출)
        execute_trades: True면 매매 신호도 실행
        """
        with self._lock:
            self.scan_count += 1
            results = self.scanner.scan()
            self.last_scan_results = results
            self.last_scan_time = datetime.now().isoformat()

            if execute_trades:
                # 매도 신호 확인
                self._check_exit_signals(results)
                # 매수 신호 확인
                self._check_entry_signals(results)

            return results
