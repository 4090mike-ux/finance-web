"""
portfolio.py - 모의 트레이딩 포트폴리오 관리
실제 거래 없이 가상 자금으로 매매를 시뮬레이션합니다.
"""

import json
import os
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "portfolio.json")


class PaperTradingPortfolio:
    """
    모의 트레이딩 포트폴리오
    - 가상 자금으로 매수/매도 시뮬레이션
    - 수익률 추적
    - 거래 내역 기록
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}      # {symbol: {shares, avg_price, entry_time}}
        self.trade_history = []  # 모든 거래 내역
        self.daily_pnl = []      # 일별 손익 기록
        self._load()

    def buy(self, symbol: str, price: float, shares: float = None,
            invest_pct: float = 0.1, reason: str = "") -> dict:
        """
        매수 실행
        Args:
            symbol: 종목 코드
            price: 매수 가격
            shares: 매수 수량 (None이면 invest_pct 비율로 자동 계산)
            invest_pct: 투자 비율 (0.1 = 총 자본의 10%)
            reason: 매수 이유
        """
        # 매수 금액 계산
        if shares is None:
            invest_amount = self.total_value * invest_pct
            invest_amount = min(invest_amount, self.cash)  # 보유 현금 초과 방지
            shares = invest_amount / price
            shares = round(shares, 4)

        cost = shares * price

        if cost > self.cash:
            logger.warning(f"{symbol} 매수 실패: 현금 부족 (필요: ${cost:.2f}, 보유: ${self.cash:.2f})")
            return {"success": False, "error": "insufficient_cash"}

        if shares <= 0:
            return {"success": False, "error": "invalid_shares"}

        # 기존 포지션에 추가
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos["shares"] + shares
            avg_price = (pos["shares"] * pos["avg_price"] + shares * price) / total_shares
            self.positions[symbol] = {
                "shares": total_shares,
                "avg_price": avg_price,
                "entry_time": pos["entry_time"],  # 처음 진입 시간 유지
            }
        else:
            self.positions[symbol] = {
                "shares": shares,
                "avg_price": price,
                "entry_time": datetime.now().isoformat(),
            }

        self.cash -= cost

        trade = {
            "type": "BUY",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "total": cost,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "cash_after": self.cash,
        }
        self.trade_history.append(trade)
        self._save()

        logger.info(f"매수: {symbol} {shares:.4f}주 @ ${price:.2f} = ${cost:.2f}")
        return {"success": True, "trade": trade}

    def sell(self, symbol: str, price: float, shares: float = None,
             reason: str = "") -> dict:
        """
        매도 실행
        Args:
            symbol: 종목 코드
            price: 매도 가격
            shares: 매도 수량 (None이면 전량 매도)
            reason: 매도 이유
        """
        if symbol not in self.positions:
            return {"success": False, "error": "no_position"}

        pos = self.positions[symbol]

        if shares is None or shares >= pos["shares"]:
            shares = pos["shares"]
            del self.positions[symbol]
        else:
            self.positions[symbol]["shares"] -= shares

        revenue = shares * price
        cost_basis = shares * pos["avg_price"]
        pnl = revenue - cost_basis
        pnl_pct = (pnl / cost_basis) * 100

        self.cash += revenue

        trade = {
            "type": "SELL",
            "symbol": symbol,
            "price": price,
            "shares": shares,
            "total": revenue,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "cash_after": self.cash,
        }
        self.trade_history.append(trade)
        self._save()

        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"매도: {symbol} {shares:.4f}주 @ ${price:.2f} | P&L: {emoji} ${pnl:.2f} ({pnl_pct:.2f}%)")
        return {"success": True, "trade": trade, "pnl": pnl, "pnl_pct": pnl_pct}

    @property
    def total_value(self) -> float:
        """포트폴리오 총 가치 (현금 + 보유 주식 현재가 기준)"""
        return self.cash + self.position_value

    @property
    def position_value(self) -> float:
        """보유 주식의 평균 매입가 기준 가치"""
        total = 0.0
        for pos in self.positions.values():
            total += pos["shares"] * pos["avg_price"]
        return total

    def update_position_prices(self, prices: dict):
        """현재 시장 가격으로 포지션 가치 업데이트"""
        self._current_prices = prices

    def get_current_value(self, current_prices: dict) -> float:
        """현재 시장가 기준 포트폴리오 총 가치"""
        total = self.cash
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos["avg_price"])
            total += pos["shares"] * price
        return total

    def get_pnl_summary(self, current_prices: dict = None) -> dict:
        """손익 요약"""
        if current_prices:
            current_value = self.get_current_value(current_prices)
        else:
            current_value = self.total_value

        total_pnl = current_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        # 실현 손익 (매도 완료 거래)
        realized_pnl = sum(t["pnl"] for t in self.trade_history if t["type"] == "SELL")

        # 미실현 손익
        unrealized_pnl = 0
        if current_prices:
            for symbol, pos in self.positions.items():
                price = current_prices.get(symbol, pos["avg_price"])
                unrealized_pnl += pos["shares"] * (price - pos["avg_price"])

        # 오늘 거래
        today = datetime.now().date().isoformat()
        today_trades = [t for t in self.trade_history if t["timestamp"][:10] == today]
        today_pnl = sum(t.get("pnl", 0) for t in today_trades if t["type"] == "SELL")

        return {
            "initial_capital": self.initial_capital,
            "current_value": round(current_value, 2),
            "cash": round(self.cash, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "total_trades": len(self.trade_history),
            "open_positions": len(self.positions),
        }

    def get_positions_detail(self, current_prices: dict = None) -> list:
        """포지션 상세 정보"""
        result = []
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos["avg_price"]) if current_prices else pos["avg_price"]
            unrealized_pnl = pos["shares"] * (current_price - pos["avg_price"])
            unrealized_pnl_pct = ((current_price - pos["avg_price"]) / pos["avg_price"]) * 100

            result.append({
                "symbol": symbol,
                "shares": round(pos["shares"], 4),
                "avg_price": round(pos["avg_price"], 2),
                "current_price": round(current_price, 2),
                "value": round(pos["shares"] * current_price, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "entry_time": pos["entry_time"],
            })
        return result

    def reset(self, initial_capital: float = None):
        """포트폴리오 초기화"""
        if initial_capital:
            self.initial_capital = initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = []
        self._save()
        logger.info(f"포트폴리오 초기화: ${self.initial_capital:,.2f}")

    def _load(self):
        """저장된 포트폴리오 로드"""
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cash = data.get("cash", self.initial_capital)
                self.positions = data.get("positions", {})
                self.trade_history = data.get("trade_history", [])
                self.daily_pnl = data.get("daily_pnl", [])
                self.initial_capital = data.get("initial_capital", self.initial_capital)
                logger.info(f"포트폴리오 로드: 현금 ${self.cash:,.2f}, 포지션 {len(self.positions)}개")
            except Exception as e:
                logger.error(f"포트폴리오 로드 실패: {e}")

    def _save(self):
        """포트폴리오 저장"""
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        data = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "trade_history": self.trade_history[-500:],  # 최근 500건만 저장
            "daily_pnl": self.daily_pnl,
            "last_updated": datetime.now().isoformat(),
        }
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
