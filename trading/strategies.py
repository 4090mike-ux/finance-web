"""
strategies.py - 적응형 트레이딩 전략
시장 상황에 따라 자동으로 파라미터를 조정하는 AI 트레이딩 전략
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveStrategy:
    """
    적응형 모멘텀 전략
    - 시장 변동성에 따라 파라미터 자동 조정
    - 최근 성과에 따라 진입 조건 강화/완화
    - 목표 수익률: 일 100% (모의 트레이딩 최적화)
    """

    def __init__(self):
        # 기본 파라미터
        self.params = {
            "momentum_threshold": 3.0,    # 최소 모멘텀 변화율 (%)
            "volume_threshold": 1.5,       # 최소 거래량 배수
            "max_position_pct": 0.20,      # 포지션당 최대 비중 (20%)
            "profit_target_pct": 5.0,      # 목표 수익률 (%)
            "stop_loss_pct": 2.0,          # 손절 비율 (%)
            "trailing_stop_pct": 1.5,      # 트레일링 스탑 (%)
            "max_positions": 5,            # 최대 동시 포지션 수
            "rsi_buy_max": 70,             # RSI 매수 최대값
            "rsi_buy_min": 30,             # RSI 매수 최소값
        }

        # 성과 추적
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.recent_trades = []  # 최근 20건 성과

        # 적응 파라미터
        self.adaptation_round = 0

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return (self.win_count / total) if total > 0 else 0.5

    def should_buy(self, stock_data: dict, portfolio_value: float,
                   open_positions: int) -> dict:
        """
        매수 신호 판단
        Returns: {"signal": bool, "invest_pct": float, "reason": str}
        """
        # 최대 포지션 초과
        if open_positions >= self.params["max_positions"]:
            return {"signal": False, "reason": "max_positions_reached"}

        symbol = stock_data["symbol"]
        day_change = stock_data.get("day_change_pct", 0)
        hour_change = stock_data.get("hour_change_pct", 0)
        vol_ratio = stock_data.get("volume_ratio", 0)
        rsi = stock_data.get("rsi", 50)
        score = stock_data.get("momentum_score", 0)

        # 기본 필터: 상승 모멘텀만 매수 (롱 전략)
        if day_change < self.params["momentum_threshold"] and hour_change < self.params["momentum_threshold"]:
            return {"signal": False, "reason": "weak_momentum"}

        # 거래량 필터 (코인은 vol_ratio 1.0이 기본이므로 좀 더 유연하게)
        if vol_ratio < self.params["volume_threshold"] and vol_ratio > 0.5:
            # 모멘텀이 매우 강하면 거래량 조건 완화
            if abs(day_change) < self.params["momentum_threshold"] * 3:
                return {"signal": False, "reason": "low_volume"}

        if rsi > self.params["rsi_buy_max"]:
            return {"signal": False, "reason": f"rsi_overbought({rsi:.1f})"}

        if rsi < self.params["rsi_buy_min"]:
            return {"signal": False, "reason": f"rsi_oversold({rsi:.1f})"}

        # 투자 비중 결정 (모멘텀 강도에 따라)
        base_pct = 0.10
        if score > 20:
            invest_pct = min(self.params["max_position_pct"], base_pct * 2)
        elif score > 10:
            invest_pct = min(self.params["max_position_pct"], base_pct * 1.5)
        else:
            invest_pct = base_pct

        reason = f"모멘텀 매수 | 일변화:{day_change:.1f}% 시간변화:{hour_change:.1f}% 거래량:{vol_ratio:.1f}x RSI:{rsi:.1f} 점수:{score}"

        return {
            "signal": True,
            "invest_pct": invest_pct,
            "reason": reason,
        }

    def should_sell(self, symbol: str, current_price: float,
                    position: dict, highest_price: float = None) -> dict:
        """
        매도 신호 판단
        Args:
            symbol: 종목 코드
            current_price: 현재가
            position: 포지션 정보 {shares, avg_price, entry_time}
            highest_price: 진입 이후 최고가 (트레일링 스탑용)
        """
        avg_price = position["avg_price"]
        pnl_pct = ((current_price - avg_price) / avg_price) * 100

        # 목표 수익률 달성
        if pnl_pct >= self.params["profit_target_pct"]:
            return {
                "signal": True,
                "reason": f"목표수익달성 +{pnl_pct:.2f}%",
                "type": "profit_target",
            }

        # 손절 (Stop Loss)
        if pnl_pct <= -self.params["stop_loss_pct"]:
            return {
                "signal": True,
                "reason": f"손절 {pnl_pct:.2f}%",
                "type": "stop_loss",
            }

        # 트레일링 스탑 (최고가에서 하락 시)
        if highest_price and highest_price > avg_price:
            trail_pnl_pct = ((current_price - highest_price) / highest_price) * 100
            if trail_pnl_pct <= -self.params["trailing_stop_pct"]:
                return {
                    "signal": True,
                    "reason": f"트레일링스탑 최고가:{highest_price:.2f}→현재:{current_price:.2f} ({trail_pnl_pct:.2f}%)",
                    "type": "trailing_stop",
                }

        # 보유 시간 초과 (4시간 이상 보유 후 손익 없으면 정리)
        entry_time = datetime.fromisoformat(position["entry_time"])
        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600
        if hold_hours > 4 and abs(pnl_pct) < 1.0:
            return {
                "signal": True,
                "reason": f"시간초과({hold_hours:.1f}시간) 수익없음 {pnl_pct:.2f}%",
                "type": "time_exit",
            }

        return {"signal": False, "reason": f"보유중 {pnl_pct:.2f}%"}

    def update_performance(self, pnl: float):
        """거래 성과 업데이트 및 파라미터 적응"""
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.total_pnl += pnl
        self.recent_trades.append(pnl)
        if len(self.recent_trades) > 20:
            self.recent_trades.pop(0)

        # 10번 거래마다 파라미터 적응
        total_trades = self.win_count + self.loss_count
        if total_trades > 0 and total_trades % 10 == 0:
            self._adapt_params()

    def _adapt_params(self):
        """
        최근 성과에 따라 전략 파라미터 자동 조정
        - 승률이 낮으면 진입 조건 강화
        - 승률이 높으면 공격적 투자
        """
        self.adaptation_round += 1
        recent_win_rate = sum(1 for p in self.recent_trades if p > 0) / len(self.recent_trades)

        logger.info(f"전략 적응 #{self.adaptation_round}: 최근 승률 {recent_win_rate:.1%}")

        if recent_win_rate < 0.4:
            # 승률 낮음: 조건 강화, 리스크 축소
            self.params["momentum_threshold"] = min(5.0, self.params["momentum_threshold"] + 0.5)
            self.params["volume_threshold"] = min(3.0, self.params["volume_threshold"] + 0.2)
            self.params["max_position_pct"] = max(0.10, self.params["max_position_pct"] - 0.02)
            self.params["stop_loss_pct"] = max(1.0, self.params["stop_loss_pct"] - 0.2)
            logger.info("  → 방어적 모드: 진입 조건 강화")

        elif recent_win_rate > 0.65:
            # 승률 높음: 조건 완화, 공격적 투자
            self.params["momentum_threshold"] = max(2.0, self.params["momentum_threshold"] - 0.3)
            self.params["max_position_pct"] = min(0.30, self.params["max_position_pct"] + 0.02)
            self.params["profit_target_pct"] = min(10.0, self.params["profit_target_pct"] + 0.5)
            logger.info("  → 공격적 모드: 투자 비중 증가")

        else:
            # 중간: 기본 파라미터 유지
            logger.info("  → 균형 모드: 파라미터 유지")

    def get_status(self) -> dict:
        """전략 현재 상태"""
        return {
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_rate, 3),
            "total_pnl": round(self.total_pnl, 2),
            "adaptation_round": self.adaptation_round,
            "params": self.params.copy(),
        }
