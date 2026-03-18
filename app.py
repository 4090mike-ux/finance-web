from flask import Flask, render_template
import os
from dotenv import load_dotenv
from models import init_db

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

with app.app_context():
    init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stock")
def stock():
    return render_template("stock.html")

@app.route("/coin")
def coin():
    return render_template("coin.html")

@app.route("/bank")
def bank():
    return render_template("bank.html")

@app.route("/realty")
def realty():
    return render_template("realty.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

if __name__ == "__main__":
    app.run(debug=True)
