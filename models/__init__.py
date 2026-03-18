import sqlite3

DB_PATH = "finance.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 회원 테이블
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

    # 즐겨찾기 테이블 (주식/코인/은행/부동산 공통)
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

    # 알림 테이블
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

    # 트러블슈팅 로그 테이블 (포트폴리오 탭2용)
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

    conn.commit()
    conn.close()
    print("✅ DB 초기화 완료")
