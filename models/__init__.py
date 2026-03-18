# =============================================
# 파일명: models/__init__.py
# 역할: 데이터베이스(DB) 연결 및 테이블 생성
#       앱 시작 시 자동으로 실행되어
#       필요한 테이블 4개를 만들어줌
# =============================================

# sqlite3 = Python 기본 내장 DB 도구
# 별도 설치 없이 사용 가능
# 데이터를 파일(.db) 형태로 저장함
import sqlite3

# DB 파일의 이름과 저장 위치 설정
# app.py 실행 시 같은 폴더에 finance.db 파일이 자동 생성됨
DB_PATH = "finance.db"


def get_db():
    """
    DB에 연결하고 연결 객체를 반환하는 함수
    이 함수를 호출하면 finance.db 파일이 열림
    (파일이 없으면 자동으로 새로 만들어짐)
    """
    # finance.db 파일에 연결 (없으면 자동 생성)
    conn = sqlite3.connect(DB_PATH)

    # 조회 결과를 딕셔너리처럼 컬럼 이름으로 접근 가능하게 설정
    # 예: row["email"] 처럼 이름으로 꺼낼 수 있음
    conn.row_factory = sqlite3.Row

    # 연결 객체를 반환 (이걸로 SQL 쿼리 실행)
    return conn


def init_db():
    """
    테이블 4개를 자동으로 생성하는 함수
    app.py에서 서버 시작 시 딱 한 번 호출됨
    이미 테이블이 있으면 건드리지 않음 (데이터 보존)
    """
    # DB 연결
    conn = get_db()

    # cursor = DB에 SQL 명령을 내리는 도구 (펜 같은 것)
    cursor = conn.cursor()


    # =============================================
    # 테이블 1: users (회원 정보)
    # 역할: 가입한 회원의 정보를 저장
    # =============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            nickname   TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now', 'localtime')),
            is_active  INTEGER DEFAULT 1
        )
    """)
    # CREATE TABLE IF NOT EXISTS = 테이블이 없을 때만 생성
    # id         = 자동으로 1,2,3... 증가하는 고유번호 (PRIMARY KEY)
    # email      = 로그인 아이디. UNIQUE = 같은 이메일로 중복 가입 불가
    # password   = 비밀번호. NOT NULL = 반드시 입력해야 함
    # nickname   = 화면에 표시될 닉네임
    # created_at = 가입한 날짜·시각 자동 기록
    # is_active  = 1이면 활성 계정, 0이면 탈퇴 처리된 계정


    # =============================================
    # 테이블 2: watchlist (즐겨찾기)
    # 역할: 사용자가 즐겨찾기한 주식·코인·은행·부동산 저장
    # =============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            category  TEXT    NOT NULL,
            item_code TEXT    NOT NULL,
            item_name TEXT    NOT NULL,
            added_at  TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # user_id   = 어떤 회원의 즐겨찾기인지 (users 테이블의 id 참조)
    # category  = 종류 구분: "stock"(주식) / "coin"(코인) / "bank"(은행) / "realty"(부동산)
    # item_code = 종목 코드 (예: 005930 = 삼성전자, BTC = 비트코인)
    # item_name = 종목 이름 (예: 삼성전자, 비트코인)
    # FOREIGN KEY = user_id가 반드시 users.id에 존재하는 값이어야 함
    #               (없는 회원의 즐겨찾기는 저장 불가 → 데이터 무결성 보장)


    # =============================================
    # 테이블 3: alerts (가격 알림)
    # 역할: 특정 가격 도달 시 알림을 보내기 위한 설정 저장
    # =============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            category     TEXT    NOT NULL,
            item_code    TEXT    NOT NULL,
            target_price REAL    NOT NULL,
            condition    TEXT    NOT NULL,
            is_triggered INTEGER DEFAULT 0,
            created_at   TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # target_price = 알림 발동 목표 가격 (REAL = 소수점 포함 숫자)
    # condition    = "above"(목표가 이상) / "below"(목표가 이하)
    # is_triggered = 0이면 아직 대기중, 1이면 알림이 이미 발동됨


    # =============================================
    # 테이블 4: trouble_logs (트러블슈팅 기록)
    # 역할: 에러 발생 및 해결 과정을 기록
    #       포트폴리오 탭2(트러블슈팅 & AI 로그)에 자동 기록용
    # =============================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trouble_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            stage      TEXT    NOT NULL,
            error_desc TEXT    NOT NULL,
            ai_prompt  TEXT,
            solution   TEXT,
            screenshot TEXT,
            logged_at  TEXT    DEFAULT (datetime('now', 'localtime'))
        )
    """)
    # stage      = 몇 단계 작업 중 발생했는지 (예: "2단계")
    # error_desc = 에러 내용 설명
    # ai_prompt  = AI에게 질문한 프롬프트 전문 (NOT NULL 없음 = 선택 입력)
    # solution   = 해결 방법 및 적용한 코드
    # screenshot = 에러 화면 스크린샷 링크


    # 모든 변경사항을 DB 파일에 저장
    # (commit 안 하면 프로그램 종료 시 사라짐)
    conn.commit()

    # DB 연결 종료 (안 닫으면 메모리 낭비)
    conn.close()

    # 터미널에 완료 메시지 출력
    print("✅ DB 초기화 완료")
