"""
trading_app.py - AI 트레이딩 시스템 독립 실행 앱
기존 app.py 없이 트레이딩 대시보드만 단독으로 실행 가능

실행: python trading_app.py
접속: http://localhost:5001/trading/
"""

import logging
import os
from flask import Flask, redirect
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "trading-secret-2026")

from trading_api import trading_bp
app.register_blueprint(trading_bp)


@app.route("/")
def index():
    return redirect("/trading/")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Trading System Started")
    print("=" * 60)
    print("Dashboard: http://localhost:5001/trading/")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
