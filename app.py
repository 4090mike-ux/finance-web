# =============================================
# 파일명: app.py
# 역할: Flask 웹서버의 시작점(진입점)
#       사용자가 어떤 URL로 접속하면
#       어떤 HTML 파일을 보여줄지 결정하는 파일
# =============================================

# Flask 패키지에서 필요한 도구 2개를 가져옴
# Flask        = 웹서버를 만드는 핵심 도구
# render_template = HTML 파일을 브라우저에 보내주는 도구
from flask import Flask, render_template
from trading_api import trading_bp

# os = 운영체제 기능을 사용하는 도구
# 여기서는 .env 파일의 값을 읽어오는 데 사용
import os

# .env 파일을 읽어서 환경변수로 등록해주는 도구
from dotenv import load_dotenv

# models 폴더의 __init__.py 파일에서
# init_db 함수를 가져옴 (DB 테이블 자동 생성 함수)
from models import init_db

# .env 파일 읽기 실행
# 이 줄 이후부터 os.getenv()로 .env 값을 꺼낼 수 있음
load_dotenv()

# Flask 웹서버 객체 생성
# __name__ = 현재 파일명(app.py)을 의미
# Flask가 templates/, static/ 폴더 위치를 찾을 때 사용
app = Flask(__name__)

# 세션(로그인 유지 정보)을 암호화할 비밀 열쇠 설정
# os.getenv("SECRET_KEY") = .env 파일에서 SECRET_KEY 값을 읽어옴
# "dev-secret-key" = .env에 값이 없을 경우 사용할 기본값
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

# 앱이 시작될 때 DB 테이블을 자동으로 생성
# app.app_context() = Flask 앱이 켜진 상태를 만들어주는 블록
# 이 블록 안에서만 DB 작업이 가능
with app.app_context():
    init_db()  # models/__init__.py 의 init_db() 함수 호출

# 트레이딩 블루프린트 등록
app.register_blueprint(trading_bp)


# =============================================
# URL 라우팅 (어떤 URL → 어떤 화면)
# @app.route() = 데코레이터
# 특정 URL로 접속하면 바로 아래 함수를 실행해라
# =============================================

# "/" URL 접속 시 → 메인 대시보드 화면 표시
@app.route("/")
def index():
    # templates/index.html 파일을 브라우저에 전송
    return render_template("index.html")

# "/stock" URL 접속 시 → 주식 화면 표시
@app.route("/stock")
def stock():
    return render_template("stock.html")

# "/coin" URL 접속 시 → 코인 화면 표시
@app.route("/coin")
def coin():
    return render_template("coin.html")

# "/bank" URL 접속 시 → 은행 화면 표시
@app.route("/bank")
def bank():
    return render_template("bank.html")

# "/realty" URL 접속 시 → 부동산 화면 표시
@app.route("/realty")
def realty():
    return render_template("realty.html")

# "/analysis" URL 접속 시 → 통합 분석 화면 표시
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

# "/login" URL 접속 시 → 로그인 화면 표시
# (POST 처리는 2단계에서 추가 예정)
@app.route("/login")
def login():
    return render_template("login.html")

# "/register" URL 접속 시 → 회원가입 화면 표시
# (POST 처리는 2단계에서 추가 예정)
@app.route("/register")
def register():
    return render_template("register.html")


# =============================================
# 서버 실행
# =============================================

# 이 파일을 직접 실행할 때만 서버를 시작
# (다른 파일에서 import 했을 때는 실행 안 됨)
if __name__ == "__main__":
    # debug=True = 코드 수정 시 서버 자동 재시작
    #              에러 발생 시 브라우저에 상세 메시지 표시
    app.run(debug=True)
