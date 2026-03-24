"""
trading_api.py - 트레이딩 AI Flask API 라우트
기존 app.py에 통합하거나 별도로 실행 가능
"""

from flask import Blueprint, jsonify, request, render_template
import logging

from trading.engine import get_engine

logger = logging.getLogger(__name__)

trading_bp = Blueprint("trading", __name__, url_prefix="/trading")


@trading_bp.route("/")
def dashboard():
    """트레이딩 대시보드 메인 페이지"""
    return render_template("trading/dashboard.html")


@trading_bp.route("/api/status")
def api_status():
    """엔진 전체 상태 API"""
    try:
        engine = get_engine()
        status = engine.get_status()
        return jsonify({"success": True, "data": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/scan", methods=["POST"])
def api_scan():
    """수동 스캔 실행"""
    try:
        engine = get_engine()
        results = engine.manual_scan()
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results[:20],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/start", methods=["POST"])
def api_start():
    """트레이딩 엔진 시작"""
    try:
        data = request.get_json() or {}
        interval = data.get("interval", 300)
        engine = get_engine()
        engine.start(scan_interval=interval)
        return jsonify({"success": True, "message": f"엔진 시작 (주기: {interval}초)"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/stop", methods=["POST"])
def api_stop():
    """트레이딩 엔진 정지"""
    try:
        engine = get_engine()
        engine.stop()
        return jsonify({"success": True, "message": "엔진 정지"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/portfolio/reset", methods=["POST"])
def api_reset():
    """포트폴리오 초기화"""
    try:
        data = request.get_json() or {}
        capital = data.get("capital", 10000.0)
        engine = get_engine()
        engine.portfolio.reset(capital)
        return jsonify({"success": True, "message": f"포트폴리오 초기화: ${capital:,.2f}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/trade/buy", methods=["POST"])
def api_manual_buy():
    """수동 매수"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        price = float(data.get("price", 0))
        invest_pct = float(data.get("invest_pct", 0.1))

        if not symbol or price <= 0:
            return jsonify({"success": False, "error": "symbol과 price 필요"}), 400

        engine = get_engine()
        result = engine.portfolio.buy(
            symbol, price, invest_pct=invest_pct, reason="수동 매수"
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/trade/sell", methods=["POST"])
def api_manual_sell():
    """수동 매도"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        price = float(data.get("price", 0))

        if not symbol or price <= 0:
            return jsonify({"success": False, "error": "symbol과 price 필요"}), 400

        engine = get_engine()
        result = engine.portfolio.sell(symbol, price, reason="수동 매도")
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@trading_bp.route("/api/price/<symbol>")
def api_price(symbol):
    """단일 종목 실시간 가격 조회"""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol.upper())
        info = ticker.fast_info
        price = float(info.last_price)
        return jsonify({
            "success": True,
            "symbol": symbol.upper(),
            "price": price,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
