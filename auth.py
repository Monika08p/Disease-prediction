"""
auth.py

Authentication module for the Disease Prediction System.
Handles user registration, login, logout, and session management.
Uses SQLite for user storage and bcrypt for password hashing.
"""

import os
import sqlite3
from datetime import datetime

from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

auth_bp = Blueprint("auth", __name__)


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            registered_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------

def get_user_by_email(email: str):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
    conn.close()
    return user


def create_user(full_name: str, email: str, password: str):
    password_hash = generate_password_hash(password)
    registered_at = datetime.now().isoformat()
    conn = get_db()
    conn.execute(
        "INSERT INTO users (full_name, email, password_hash, registered_at) VALUES (?, ?, ?, ?)",
        (full_name.strip(), email.lower().strip(), password_hash, registered_at),
    )
    conn.commit()
    conn.close()


def login_required(f):
    """Decorator to protect routes — redirects to login if not authenticated."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # Already logged in — go home
    if "user_id" in session:
        return redirect(url_for("landing"))

    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        user = get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["user_email"] = user["email"]
            return redirect(url_for("landing"))
        else:
            error = "Invalid Email or Password."

    return render_template("login.html", error=error)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    # Already logged in — go home
    if "user_id" in session:
        return redirect(url_for("landing"))

    error = None
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not full_name:
            error = "Full name is required."
        elif not email or "@" not in email:
            error = "A valid email address is required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif get_user_by_email(email):
            error = "An account with this email already exists."
        else:
            create_user(full_name, email, password)
            # Auto-login after registration
            user = get_user_by_email(email)
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["user_email"] = user["email"]
            return redirect(url_for("landing"))

    return render_template("register.html", error=error)


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
