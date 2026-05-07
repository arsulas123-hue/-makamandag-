import json
import csv
import io
import os
import requests
import base64
import re
import get_remote_address
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, request, jsonify, session, Response, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from sqlalchemy import inspect, text
import google.generativeai as genai
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
app = Flask(__name__)
# ---------- Manual rate‑limiter (no extra package needed) ----------
from threading import Lock
from collections import defaultdict
import time as _time

class ManualRateLimiter:
    def __init__(self):
        self.lock = Lock()
        self.requests = defaultdict(list)

    def allow(self, key, max_calls, period):
        now = _time.time()
        with self.lock:
            self.requests[key] = [t for t in self.requests[key] if now - t < period]
            if len(self.requests[key]) >= max_calls:
                return False
            self.requests[key].append(now)
            return True

manual_rl = ManualRateLimiter()

def your_ml_planning_function(income, mindset, selected_categories):
    """Simple ML plan that distributes income based on mindset."""
    # Base percentages for "Neutral" mindset
    if mindset == 'Saver':
        savings_pct = 30
        needs_pct = 50
        wants_pct = 20
    elif mindset == 'Spender':
        savings_pct = 10
        needs_pct = 50
        wants_pct = 40
    else:  # Neutral
        savings_pct = 20
        needs_pct = 60
        wants_pct = 20

    allocation = {}
    allocation_amounts = {}

    # Distribute percentages among categories
    need_cats = [c for c in selected_categories if c in ['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage']]
    want_cats = [c for c in selected_categories if c not in need_cats and c != 'Savings']
    savings_cats = ['Savings'] if 'Savings' in selected_categories else []

    # Assign percentages
    for cat in need_cats:
        allocation[cat] = needs_pct / max(len(need_cats), 1)
    for cat in want_cats:
        allocation[cat] = wants_pct / max(len(want_cats), 1)
    for cat in savings_cats:
        allocation[cat] = savings_pct

    # Normalise to 100% (avoid floating point issues)
    total_pct = sum(allocation.values())
    if total_pct > 0:
        for cat in allocation:
            allocation[cat] = allocation[cat] / total_pct * 100

    # Compute amounts
    for cat, pct in allocation.items():
        allocation_amounts[cat] = income * pct / 100

    # Savings plan
    monthly_savings = allocation_amounts.get('Savings', 0) if 'Savings' in allocation else 0
    savings_plan = {
        'daily': monthly_savings / 30,
        'weekly': monthly_savings / 4,
        'monthly': monthly_savings,
        'tip': 'Try to automate this amount to a separate account.'
    }

    # Simple advice
    advice = []
    if monthly_savings < income * 0.1:
        advice.append({'type': 'warning', 'title': 'Low savings rate', 'body': 'Try to save at least 10% of your income.'})
    else:
        advice.append({'type': 'success', 'title': 'Good savings habit', 'body': 'You are saving a healthy percentage.'})
    if wants_pct > 30:
        advice.append({'type': 'info', 'title': 'Watch discretionary spending', 'body': 'Consider reducing wants to 20-30% of your budget.'})

    financial_summary = f"Based on your {mindset} mindset, we allocated {savings_pct:.0f}% to savings, {needs_pct:.0f}% to needs, and {wants_pct:.0f}% to wants. Your monthly savings target is {fmt(monthly_savings)}."

    return {
        'allocation': allocation,
        'allocation_amounts': allocation_amounts,
        'savings_plan': savings_plan,
        'financial_summary': financial_summary,
        'advice': advice,
        'monthly_income': income
    }

def fmt(amount):
    return f"₱{amount:,.2f}"


app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-secret-key-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///smartspend.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp'

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CORS(app, supports_credentials=True)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ----------------------------------------------------------------------
# Multi-AI Router
# ----------------------------------------------------------------------
GEMINI_MODELS = ["gemini-2.0-flash"]
FREE_MODELS = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

def route_ai_request(prompt, max_tokens=400):
    """Try OpenRouter free models first, then Gemini. Return content string or None."""
    if OPENROUTER_API_KEY:
        # Updated model IDs that are actually available on OpenRouter free tier
        FREE_MODELS = [
            "google/gemini-2.0-flash-001:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ]
        for model in FREE_MODELS:
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    },
                    timeout=20,  # Slightly longer timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        print(f"✅ Used OpenRouter: {model}")
                        return data["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"⚠️ OpenRouter ({model}) returned unexpected format: {data}")
                else:
                    error_msg = resp.json().get("error", {}).get("message", resp.text)
                    print(f"❌ OpenRouter ({model}) HTTP {resp.status_code}: {error_msg}")
            except Exception as e:
                print(f"❌ OpenRouter ({model}) exception: {e}")
                continue
    else:
        print("ℹ️ OPENROUTER_API_KEY not set – skipping OpenRouter")

    if GEMINI_API_KEY:
        GEMINI_MODELS = ["gemini-2.0-flash"]  # Use the latest stable
        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    print(f"✅ Used Gemini: {model_name}")
                    return response.text.strip()
                else:
                    print(f"⚠️ Gemini ({model_name}) returned no text")
            except Exception as e:
                print(f"❌ Gemini ({model_name}) failed: {e}")
    else:
        print("ℹ️ GEMINI_API_KEY not set – skipping Gemini")

    return None  # Explicitly signal failure

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    social_status = db.Column(db.String(20), default='Middle')
    spending_mindset = db.Column(db.String(20), default='Neutral')
    monthly_budget_limit = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    role = db.Column(db.String(20), default='user')
    avatar_url = db.Column(db.Text, nullable=True)

    transactions = db.relationship('Transaction', backref='user', lazy=True)
    budgets = db.relationship('Budget', backref='user', lazy=True)
    future_expenses = db.relationship('FutureExpense', backref='user', lazy=True)
    allocations = db.relationship('UserAllocation', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        try:
            return bcrypt.check_password_hash(self.password_hash, password)
        except ValueError:
            return False

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'social_status': self.social_status,
            'spending_mindset': self.spending_mindset,
            'monthly_budget_limit': self.monthly_budget_limit,
            'role': self.role,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    tx_type = db.Column(db.String(10), nullable=False)
    is_need = db.Column(db.Boolean, default=True)
    priority = db.Column(db.Integer, default=1)
    note = db.Column(db.String(200))
    tx_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'amount': self.amount,
            'category': self.category,
            'tx_type': self.tx_type,
            'is_need': self.is_need,
            'priority': self.priority,
            'note': self.note,
            'tx_date': self.tx_date.isoformat()
        }

class Budget(db.Model):
    __tablename__ = 'budgets'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    limit_amount = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category', name='unique_user_category'),)

class FutureExpense(db.Model):
    __tablename__ = 'future_expenses'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    description = db.Column(db.String(200), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    cycle = db.Column(db.String(20), default='One-time')
    expense_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id': self.id,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'cycle': self.cycle,
            'date': self.expense_date.isoformat()
        }

class UserAllocation(db.Model):
    __tablename__ = 'user_allocations'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category_name = db.Column(db.String(50), nullable=False)
    type = db.Column(db.String(10), nullable=False)
    percentage = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category_name', name='unique_user_category_allocation'),)

# ----------------------------------------------------------------------
# Schema migration (keeps existing DB compatible)
# ----------------------------------------------------------------------
def ensure_schema():
    inspector = inspect(db.engine)
    if inspector.has_table('users'):
        existing_columns = [col['name'] for col in inspector.get_columns('users')]
        if 'password' in existing_columns and 'password_hash' in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users DROP COLUMN password'))
                conn.commit()
        for col, defn in [
            ('password_hash', "VARCHAR(128) NOT NULL DEFAULT ''"),
            ('social_status', "VARCHAR(20) DEFAULT 'Middle'"),
            ('spending_mindset', "VARCHAR(20) DEFAULT 'Neutral'"),
            ('monthly_budget_limit', "FLOAT DEFAULT 0.0"),
            ('role', "VARCHAR(20) DEFAULT 'user'"),
            ('avatar_url', "TEXT"),
        ]:
            if col not in existing_columns:
                with db.engine.connect() as conn:
                    conn.execute(text(f'ALTER TABLE users ADD COLUMN {col} {defn}'))
                    conn.commit()
    if inspector.has_table('transactions'):
        tx_columns = [col['name'] for col in inspector.get_columns('transactions')]
        if 'is_need' not in tx_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE transactions ADD COLUMN is_need BOOLEAN DEFAULT TRUE'))
                conn.commit()
        if 'priority' not in tx_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE transactions ADD COLUMN priority INTEGER DEFAULT 1'))
                conn.commit()
    db.create_all()
    admin = User.query.filter_by(email='admin@smartspend.com').first()
    if not admin:
        admin = User(name='Admin', email='admin@smartspend.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created: admin@smartspend.com / admin123")

# ----------------------------------------------------------------------
# Auth helpers
# ----------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    return User.query.get(session.get('user_id')) if 'user_id' in session else None

# ----------------------------------------------------------------------
# Validation helper
# ----------------------------------------------------------------------
needs_set = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}

def validate_transaction(user_id, amount, tx_type):
    if amount < 1:
        raise ValueError("Transaction amount must be at least 1.")
    if tx_type == 'expense':
        user = User.query.get(user_id)
        if user.monthly_budget_limit <= 0:
            raise ValueError("Your monthly budget is zero. Please set an income first.")
        now = datetime.now(timezone.utc)
        first_of_month = datetime(now.year, now.month, 1)
        total_income = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.tx_type == 'income',
            Transaction.tx_date >= first_of_month
        ).scalar() or 0
        total_expense = db.session.query(db.func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.tx_type == 'expense',
            Transaction.tx_date >= first_of_month
        ).scalar() or 0
        if total_expense + amount > total_income:
            raise ValueError("This expense would exceed your monthly income. Add more income first.")

# ----------------------------------------------------------------------
# Analytics helpers
# ----------------------------------------------------------------------
def compute_health_score(user_id):
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    income = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'income',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    total_expense = sum(e.amount for e in expenses)
    total_income = sum(i.amount for i in income)
    savings_rate = max(0, (total_income - total_expense) / total_income) if total_income > 0 else 0
    want_expense = sum(e.amount for e in expenses if not e.is_need)
    total_exp = total_expense or 1
    want_ratio = want_expense / total_exp
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    overspend_penalty = 0.0
    for cat, limit in budgets.items():
        if limit == 0:
            continue
        spent = sum(e.amount for e in expenses if e.category == cat)
        if spent > limit:
            overspend_penalty += (spent - limit) / limit
    score = 70 + int(savings_rate * 20) - int(want_ratio * 15) - min(20, int(overspend_penalty * 10))
    return max(0, min(100, score))

def generate_weekly_forecast(user_id):
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    if not expenses:
        return {'Week 1': 0, 'Week 2': 0, 'Week 3': 0, 'Week 4': 0}
    daily_totals = defaultdict(float)
    for e in expenses:
        daily_totals[e.tx_date.date()] += e.amount
    days_sorted = sorted(daily_totals.keys())
    values = [daily_totals[d] for d in days_sorted]
    n = len(values)
    if n > 1:
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den else 0
        intercept = y_mean - slope * x_mean
        week_sums = [0, 0, 0, 0]
        for day in range(1, 29):
            week_sums[(day - 1) // 7] += max(0, intercept + slope * (n + day))
        return {f'Week {i+1}': week_sums[i] for i in range(4)}
    avg = sum(values) / n * 7
    return {f'Week {i+1}': avg for i in range(4)}

def get_category_totals(user_id):
    now = datetime.now(timezone.utc)
    first_of_month = datetime(now.year, now.month, 1)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= first_of_month
    ).all()
    totals = defaultdict(float)
    for e in expenses:
        totals[e.category] += e.amount
    return dict(totals)

def get_monthly_summary(user_id):
    all_trans = Transaction.query.filter_by(user_id=user_id).all()
    balance = sum(t.amount if t.tx_type == 'income' else -t.amount for t in all_trans)
    now = datetime.now(timezone.utc)
    first_of_month = datetime(now.year, now.month, 1)
    this_month_expense = sum(t.amount for t in all_trans if t.tx_type == 'expense' and t.tx_date >= first_of_month)
    this_month_income = sum(t.amount for t in all_trans if t.tx_type == 'income' and t.tx_date >= first_of_month)
    monthly = defaultdict(lambda: {'income': 0, 'expense': 0})
    for t in all_trans:
        key = t.tx_date.strftime('%Y-%m')
        monthly[key]['income' if t.tx_type == 'income' else 'expense'] += t.amount
    sorted_months = sorted(monthly.keys())[-6:]
    return {
        'balance': balance,
        'expense': this_month_expense,
        'income': this_month_income,
        'monthly': {m: monthly[m] for m in sorted_months}
    }

def compute_longevity(user_id):
    summary = get_monthly_summary(user_id)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    avg_daily = sum(e.amount for e in expenses) / 30 if expenses else 0
    days = int(summary['balance'] / avg_daily) if avg_daily > 0 else 0
    return {'balance': summary['balance'], 'avg_daily_spend': avg_daily, 'days': days}

# ----------------------------------------------------------------------
# Auth routes
# ----------------------------------------------------------------------
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not all(k in data for k in ['name', 'email', 'password']):
        return jsonify({'error': 'Missing fields'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    user = User(name=data['name'], email=data['email'], role='user')
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify(user.to_dict()), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data.get('email')).first()
    if not user or not user.check_password(data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    session['user_id'] = user.id
    return jsonify(user.to_dict()), 200

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'message': 'Logged out'}), 200

@app.route('/api/me', methods=['GET'])
def me():
    user = get_current_user()
    return jsonify(user.to_dict()) if user else (jsonify({'error': 'Unauthorized'}), 401)

@app.route('/api/me/avatar', methods=['PUT'])
@login_required
def update_avatar():
    user = get_current_user()
    data = request.json
    url = data.get('avatar_url', '').strip()
    if url and not (url.startswith('http://') or url.startswith('https://') or url.startswith('data:image/')):
        return jsonify({'error': 'Invalid URL'}), 400
    user.avatar_url = url if url else None
    db.session.commit()
    return jsonify({'avatar_url': user.avatar_url}), 200

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    user = get_current_user()
    data = request.json
    if 'name' in data:
        user.name = data['name']
    if 'email' in data:
        if User.query.filter(User.email == data['email'], User.id != user.id).first():
            return jsonify({'error': 'Email already in use'}), 400
        user.email = data['email']
    if 'password' in data:
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        user.set_password(data['password'])
    if 'avatar_url' in data:
        url = data['avatar_url'].strip()
        if url and not (url.startswith('http://') or url.startswith('https://') or url.startswith('data:image/')):
            return jsonify({'error': 'Invalid avatar URL'}), 400
        user.avatar_url = url if url else None
    db.session.commit()
    return jsonify(user.to_dict()), 200

# ----------------------------------------------------------------------
# Transaction routes
# ----------------------------------------------------------------------
@app.route('/api/transactions', methods=['GET'])
@login_required
def list_transactions():
    user = get_current_user()
    return jsonify([t.to_dict() for t in Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).all()])

@app.route('/api/transactions', methods=['POST'])
@login_required
def create_transaction():
    user = get_current_user()
    data = request.json
    if not all(k in data for k in ['amount', 'category', 'tx_type']):
        return jsonify({'error': 'Missing required fields'}), 400

    amount = data['amount']
    tx_type = data['tx_type']
    category = data['category']
    is_need = data.get('is_need', True)
    priority = data.get('priority', 1)
    note = data.get('note', '')
    tx_date_str = data.get('tx_date')
    tx_date = datetime.fromisoformat(tx_date_str) if tx_date_str else datetime.now(timezone.utc)

    try:
        validate_transaction(user.id, amount, tx_type)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    tx = Transaction(
        user_id=user.id,
        amount=amount,
        category=category,
        tx_type=tx_type,
        is_need=is_need,
        priority=priority,
        note=note,
        tx_date=tx_date
    )
    db.session.add(tx)
    db.session.commit()
    return jsonify(tx.to_dict()), 201

@app.route('/api/transactions/<int:tx_id>', methods=['DELETE'])
@login_required
def delete_transaction(tx_id):
    user = get_current_user()
    tx = Transaction.query.get(tx_id)
    if not tx or tx.user_id != user.id:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(tx)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200

# ----------------------------------------------------------------------
# Summary / Predict / Longevity
# ----------------------------------------------------------------------
@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    # --- manual rate limit: 5 requests per 60 seconds ---
    if not manual_rl.allow(f"summary_{user_id}", 5, 60):
        return jsonify({'error': 'Too many requests – slow down'}), 429
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(get_monthly_summary(user_id))


@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if not manual_rl.allow(f"predict_{user_id}", 5, 60):
        return jsonify({'error': 'Too many requests – slow down'}), 429
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    has_data = Transaction.query.filter_by(user_id=user_id, tx_type='expense').count() > 0
    if not has_data:
        return jsonify({
            'has_data': False,
            'score': None,
            'predictions': {'weekly': {}, 'categories': {}},
            'advice': []
        })
    return jsonify({
        'has_data': True,
        'score': compute_health_score(user_id),
        'predictions': {
            'weekly': generate_weekly_forecast(user_id),
            'categories': get_category_totals(user_id)
        },
        'advice': []
    })


@app.route('/api/longevity/<int:user_id>')
@login_required
def longevity(user_id):
    if not manual_rl.allow(f"longevity_{user_id}", 5, 60):
        return jsonify({'error': 'Too many requests – slow down'}), 429
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(compute_longevity(user_id))
# ----------------------------------------------------------------------
# Budget routes
# ----------------------------------------------------------------------
@app.route('/api/budgets/<int:user_id>', methods=['GET'])
@login_required
def get_budgets(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    budgets = Budget.query.filter_by(user_id=user_id).all()
    return jsonify([{'category': b.category, 'limit': b.limit_amount} for b in budgets])

@app.route('/api/budgets/<int:user_id>', methods=['POST'])
@login_required
def upsert_budget(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.json
    category = data.get('category')
    limit = data.get('limit')
    if not category or limit is None:
        return jsonify({'error': 'Category and limit required'}), 400
    existing = Budget.query.filter_by(user_id=user_id, category=category).first()
    if existing:
        existing.limit_amount = limit
    else:
        existing = Budget(user_id=user_id, category=category, limit_amount=limit)
        db.session.add(existing)
    db.session.commit()
    return jsonify({'message': 'Budget saved'})

@app.route('/api/budgets/<int:user_id>/<category>', methods=['PUT'])
@login_required
def update_single_budget(user_id, category):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.json
    limit = data.get('limit')
    if limit is None:
        return jsonify({'error': 'Limit required'}), 400
    budget = Budget.query.filter_by(user_id=user_id, category=category).first()
    if budget:
        budget.limit_amount = limit
    else:
        budget = Budget(user_id=user_id, category=category, limit_amount=limit)
        db.session.add(budget)
    db.session.commit()
    return jsonify({'message': 'Budget updated', 'limit': limit})

@app.route('/api/budgets/reset_to_ai/<int:user_id>', methods=['POST'])
@login_required
def reset_budgets_to_ai(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    allocations = UserAllocation.query.filter_by(user_id=user_id).all()
    if not allocations:
        return jsonify({'error': 'No AI allocation found. Run AI plan first.'}), 400
    user = get_current_user()
    monthly_income = user.monthly_budget_limit
    if monthly_income <= 0:
        return jsonify({'error': 'Monthly income not set. Run AI plan.'}), 400
    for alloc in allocations:
        budget = Budget.query.filter_by(user_id=user_id, category=alloc.category_name).first()
        new_limit = round(monthly_income * alloc.percentage / 100, 2)
        if budget:
            budget.limit_amount = new_limit
        else:
            db.session.add(Budget(user_id=user_id, category=alloc.category_name, limit_amount=new_limit))
    db.session.commit()
    return jsonify({'message': 'Budgets reset to AI recommendations'}), 200

# ----------------------------------------------------------------------
# Future expenses
# ----------------------------------------------------------------------
@app.route('/api/future_expenses', methods=['GET'])
@login_required
def get_future_expenses():
    user = get_current_user()
    return jsonify([e.to_dict() for e in FutureExpense.query.filter_by(user_id=user.id).order_by(FutureExpense.expense_date).all()])

@app.route('/api/future_expenses', methods=['POST'])
@login_required
def create_future_expense():
    user = get_current_user()
    data = request.json
    if not all(k in data for k in ['description', 'amount', 'category', 'date']):
        return jsonify({'error': 'Missing fields'}), 400
    exp = FutureExpense(
        user_id=user.id,
        description=data['description'],
        amount=data['amount'],
        category=data['category'],
        cycle=data.get('cycle', 'One-time'),
        expense_date=datetime.fromisoformat(data['date']).date()
    )
    db.session.add(exp)
    db.session.commit()
    return jsonify(exp.to_dict()), 201

@app.route('/api/future_expenses/<int:exp_id>', methods=['DELETE'])
@login_required
def delete_future_expense(exp_id):
    user = get_current_user()
    exp = FutureExpense.query.get(exp_id)
    if not exp or exp.user_id != user.id:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(exp)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200

# ----------------------------------------------------------------------
# Allocations
# ----------------------------------------------------------------------
@app.route('/api/allocations', methods=['GET'])
@login_required
def get_allocations():
    user = get_current_user()
    return jsonify([{'category_name': a.category_name, 'type': a.type, 'percentage': a.percentage}
                    for a in UserAllocation.query.filter_by(user_id=user.id).all()])

@app.route('/api/allocations', methods=['POST'])
@login_required
def update_allocations():
    user = get_current_user()
    data = request.json
    if not isinstance(data, list):
        return jsonify({'error': 'Expected list'}), 400
    UserAllocation.query.filter_by(user_id=user.id).delete()
    for item in data:
        db.session.add(UserAllocation(
            user_id=user.id,
            category_name=item['category_name'],
            type=item['type'],
            percentage=item['percentage']
        ))
    db.session.commit()
    return jsonify({'message': 'Saved'}), 200

# ----------------------------------------------------------------------
# Export CSV
# ----------------------------------------------------------------------
@app.route('/api/export/csv')
@login_required
def export_csv():
    user = get_current_user()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Category', 'Type', 'Need?', 'Priority', 'Note', 'Amount'])
    for t in Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).all():
        writer.writerow([t.tx_date.strftime('%Y-%m-%d %H:%M'), t.category, t.tx_type,
                         'Need' if t.is_need else 'Want', t.priority, t.note, t.amount])
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers.set('Content-Disposition', 'attachment', filename='transactions.csv')
    return resp

# ----------------------------------------------------------------------
# Apply future expenses
# ----------------------------------------------------------------------
@app.route('/api/apply_future_expenses', methods=['POST'])
@login_required
def apply_future_expenses():
    user = get_current_user()
    today = datetime.now(timezone.utc).date()
    onetime = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle == 'One-time'
    ).all()
    applied = []
    for exp in onetime:
        tx = Transaction(
            user_id=user.id,
            amount=exp.amount,
            category=exp.category,
            tx_type='expense',
            is_need=(exp.category in needs_set),
            priority=1,
            note=f"Auto-deducted future expense: {exp.description}"
        )
        db.session.add(tx)
        applied.append(exp.description)
        db.session.delete(exp)
    db.session.commit()
    return jsonify({'applied': applied, 'count': len(applied)}), 200

# ----------------------------------------------------------------------
# ADMIN API ROUTES
# ----------------------------------------------------------------------
@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    total_users = User.query.count()
    total_transactions = Transaction.query.count()
    total_income = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.tx_type == 'income').scalar() or 0
    total_expense = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.tx_type == 'expense').scalar() or 0
    all_health_scores = [compute_health_score(user.id) for user in User.query.all()]
    avg_health = sum(all_health_scores) / len(all_health_scores) if all_health_scores else 0
    return jsonify({
        'total_users': total_users,
        'total_transactions': total_transactions,
        'total_income': total_income,
        'total_expense': total_expense,
        'avg_health_score': round(avg_health, 1)
    })

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])

@app.route('/api/admin/transactions', methods=['GET'])
@admin_required
def admin_transactions():
    user_id = request.args.get('user_id', type=int)
    query = Transaction.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    transactions = query.order_by(Transaction.tx_date.desc()).all()
    return jsonify([t.to_dict() for t in transactions])

# ----------------------------------------------------------------------
# OCR endpoint
# ----------------------------------------------------------------------
@app.route('/api/ocr_income', methods=['POST'])
@login_required
def ocr_income():
    user = get_current_user()
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_bytes = file.read()
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """You are a budget OCR AI. Extract all income and expense items from the image.
Return ONLY a JSON array of objects. Each object must have:
- type: "income" or "expense"
- amount: number (no currency symbols)
- category: one of: Food & Dining, Transport, Groceries, Health, Entertainment, Debt repayment, Mortgage, Subscription, Hobbies, Salary, Savings, Other
- note: short description (e.g., "Paycheck #1")
Do NOT wrap in markdown."""
    raw = None
    if OPENROUTER_API_KEY:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}]}],
                    "max_tokens": 1000
                },
                timeout=30
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                print("✅ Vision extraction via OpenRouter")
        except Exception as e:
            print("OpenRouter vision failed:", e)
    if raw is None and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content([{'mime_type': 'image/png', 'data': b64_image}, prompt])
            if response and response.text:
                raw = response.text.strip()
                print("✅ Vision extraction via Gemini")
        except Exception as e:
            print("Gemini vision failed:", e)
    if not raw:
        return jsonify({'error': 'Could not extract transactions from image'}), 500

    items = None
    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('```')[1]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        items = json.loads(cleaned)
        if not isinstance(items, list):
            items = None
    except Exception:
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
        except Exception:
            items = None
    if not items:
        return jsonify({'error': 'Failed to parse AI output'}), 500

    created_txs = []
    errors = []
    for item in items:
        if not isinstance(item, dict):
            continue
        amount = abs(float(item.get('amount', 0)))
        tx_type = item.get('type', 'expense')
        if tx_type not in ('income', 'expense'):
            tx_type = 'expense'
        category = item.get('category', 'Other')
        note = item.get('note', '')
        is_need = category in needs_set
        try:
            validate_transaction(user.id, amount, tx_type)
        except ValueError as e:
            errors.append(f"{category}: {str(e)}")
            continue
        tx = Transaction(
            user_id=user.id,
            amount=amount,
            category=category,
            tx_type=tx_type,
            is_need=is_need,
            priority=2 if is_need else 1,
            note=note
        )
        db.session.add(tx)
        created_txs.append(tx)
    if errors:
        db.session.rollback()
        return jsonify({'error': 'Some transactions could not be added', 'details': errors}), 400
    db.session.flush()
    created = [tx.to_dict() for tx in created_txs]
    db.session.commit()
    total_income = sum(t['amount'] for t in created if t['tx_type']=='income')
    total_expense = sum(t['amount'] for t in created if t['tx_type']=='expense')
    return jsonify({
        'transactions': created,
        'total_income': total_income,
        'total_expense': total_expense,
        'count': len(created)
    })

# ----------------------------------------------------------------------
# AI full setup (with smart fallback)
# ----------------------------------------------------------------------
@app.route('/api/ai/full_setup', methods=['POST'])
def ai_full_setup():
    try:
        data = request.get_json()
        monthly_income = data.get('monthly_income', 0)
        mindset = data.get('mindset', 'Neutral')
        
        # ✅ FIX: properly extract selected_categories from request
        selected_categories = data.get('selected_categories', [])
        
        # Fallback default categories if none provided
        if not selected_categories:
            selected_categories = [
                'Food & Dining', 'Transport', 'Groceries', 'Health',
                'Entertainment', 'Debt repayment', 'Savings'
            ]
        
        # Now call your ML planning function – make sure it accepts selected_categories
        result = your_ml_planning_function(monthly_income, mindset, selected_categories)
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"ML plan error: {e}")   # or use logging
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------
# AI classification & chat
# ----------------------------------------------------------------------
@app.route('/api/ai/classify_transaction', methods=['POST'])
@login_required
def ai_classify_transaction():
    data = request.json
    if data.get('tx_type') == 'income':
        return jsonify({'is_need': True, 'priority': 3, 'suggested_note': 'Income received'})
    prompt = f"Classify expense: category {data['category']}, amount ₱{data['amount']}, note {data.get('note','none')}. Return JSON: {{'is_need':bool,'priority':int,'suggested_note':str}}"
    raw = route_ai_request(prompt, max_tokens=150)
    try:
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        return jsonify(json.loads(raw.strip()))
    except:
        return jsonify({'is_need': data['category'] in needs_set, 'priority': 2 if data['category'] in needs_set else 1, 'suggested_note': data.get('note','')})

@app.route('/api/ai/chat', methods=['POST'])
@login_required
def ai_chat():
    user = get_current_user()
    data = request.json
    summary = get_monthly_summary(user.id)
    score = compute_health_score(user.id)
    prompt = f"""
You are SmartSpend AI for {user.name}.
Balance: ₱{summary['balance']:,.2f}, Income: ₱{summary['income']:,.2f}, Expenses: ₱{summary['expense']:,.2f}, Health: {score}/100.
User asks: "{data.get('message','')}"
Reply in 3-5 sentences, warm, actionable, use ₱.
"""
    reply = route_ai_request(prompt, max_tokens=400)
    return jsonify({'reply': reply})
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartSpend — Finance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<style>
/* ───────── RESET & BASE ───────── */
* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --void:#060A10; --bg:#0B1120; --bg2:#111928; --bg3:#17223A; --bg4:#1E2E4A;
  --green:#00E5A0; --green-dim:rgba(0,229,160,0.1); --green-glow:rgba(0,229,160,0.35);
  --red:#FF3B5C; --red-dim:rgba(255,59,92,0.12);
  --amber:#F5A623; --blue:#3B8BFF; --purple:#9B59F5;
  --muted:#4A6080; --muted2:#6B88A8; --text:#D8EAF8; --text2:#B0C8E0;
  --border:rgba(0,229,160,0.15); --border2:rgba(255,255,255,0.06);
  --r:14px; --r2:20px;
  --font-display:'Syne',sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
}
body {
  font-family:var(--font-display); background:var(--void); color:var(--text);
  min-height:100vh; overflow-x:hidden;
}
::selection { background:var(--green-dim); color:var(--green); }
body::before {
  content:''; position:fixed; inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events:none; z-index:0; opacity:0.4;
}

/* ───────── SIDEBAR ───────── */
.sidebar {
  position:fixed; left:0; top:0; bottom:0; width:72px;
  background:rgba(11,17,32,0.95); backdrop-filter:blur(20px);
  border-right:1px solid var(--border2);
  display:flex; flex-direction:column; align-items:center;
  padding:20px 0; gap:6px; z-index:200;
  transition:width 0.3s cubic-bezier(0.4,0,0.2,1);
}
.sidebar:hover { width:220px; }
.logo {
  width:44px; height:44px; border-radius:12px; margin-bottom:20px;
  background:linear-gradient(135deg,var(--green),#009e6a);
  display:flex; align-items:center; justify-content:center;
  font-size:1.4rem; cursor:pointer; flex-shrink:0;
  box-shadow:0 0 24px var(--green-glow);
}
.nav-item {
  width:calc(100% - 16px); display:flex; align-items:center; gap:14px;
  padding:12px 14px; border-radius:10px; cursor:pointer;
  border:none; background:transparent; color:var(--muted2);
  font-family:var(--font-display); font-size:0.875rem; font-weight:500;
  white-space:nowrap; overflow:hidden; transition:all 0.2s; text-align:left;
}
.nav-item:hover { background:var(--green-dim); color:var(--text); }
.nav-item.active { background:var(--green-dim); color:var(--green); box-shadow:inset 2px 0 0 var(--green); }
.nav-icon { font-size:1.1rem; flex-shrink:0; width:20px; text-align:center; }
.nav-label { opacity:0; transition:opacity 0.2s; font-size:0.85rem; }
.sidebar:hover .nav-label { opacity:1; }
.sidebar-sep { width:40px; height:1px; background:var(--border2); margin:8px 0; }
.sidebar:hover .sidebar-sep { width:calc(100% - 28px); }

/* ───────── MAIN ───────── */
.main { margin-left:72px; padding:28px 36px; min-height:100vh; position:relative; z-index:1; }
@media(max-width:768px) {
  .main { margin-left:0; padding:16px; }
  .sidebar { display:none; }
}
.topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:32px; gap:16px; flex-wrap:wrap; }
.page-title { font-size:1.6rem; font-weight:800; letter-spacing:-0.5px; }
.topbar-right { display:flex; align-items:center; gap:12px; }
.health-pill {
  display:flex; align-items:center; gap:8px;
  padding:7px 16px; border-radius:99px;
  background:var(--green-dim); border:1px solid var(--border);
  font-family:var(--font-mono); font-size:0.8rem; color:var(--green);
}
.health-dot { width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 2s infinite; }
@keyframes pulse {
  0%,100%{ opacity:1; box-shadow:0 0 0 0 var(--green-glow); }
  50%{ opacity:0.8; box-shadow:0 0 0 6px transparent; }
}
.avatar {
  width:40px; height:40px; border-radius:10px;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:0.9rem; cursor:pointer; position:relative;
}
.avatar img { width:100%; height:100%; border-radius:10px; object-fit:cover; }
.btn-signout {
  background:var(--red-dim); border:1px solid rgba(255,59,92,0.2);
  color:var(--red); padding:8px 16px; border-radius:10px; cursor:pointer;
  font-family:var(--font-display); font-weight:600; font-size:0.85rem;
}

/* ───────── SCREENS ───────── */
.screen { display:none; animation:fadeIn 0.3s ease; }
.screen.active { display:block; }
@keyframes fadeIn { from{opacity:0;transform:translateY(12px);} to{opacity:1;transform:translateY(0);} }

/* ───────── CARDS ───────── */
.card {
  background:var(--bg2); border:1px solid var(--border2); border-radius:var(--r2);
  padding:24px; margin-bottom:20px; transition:border-color 0.2s;
}
.card:hover { border-color:var(--border); }
.card-header {
  display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;
  flex-wrap:wrap; gap:12px;
}
.card-title {
  font-size:0.95rem; font-weight:600; color:var(--text2);
  text-transform:uppercase; letter-spacing:0.5px;
}
.ai-badge {
  background:linear-gradient(90deg,var(--purple),var(--blue)); color:#fff;
  padding:3px 10px; border-radius:99px; font-size:0.68rem; font-weight:600;
  letter-spacing:0.5px;
}

/* ───────── STATS GRID ───────── */
.stats-grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:24px;
}
.stat-card {
  background:var(--bg2); border:1px solid var(--border2); border-radius:var(--r2); padding:22px;
  position:relative; overflow:hidden; transition:all 0.25s; cursor:default;
}
.stat-card::before {
  content:''; position:absolute; top:0; right:0; width:80px; height:80px;
  background:radial-gradient(circle,var(--green-dim),transparent 70%);
  border-radius:50%; transform:translate(30px,-30px);
}
.stat-card.neg::before { background:radial-gradient(circle,var(--red-dim),transparent 70%); }
.stat-card.blue-glow::before { background:radial-gradient(circle,rgba(59,139,255,0.1),transparent 70%); }
.stat-value { font-family:var(--font-mono); font-size:1.7rem; font-weight:600; margin-bottom:6px; letter-spacing:-1px; }
.stat-label { font-size:0.72rem; color:var(--muted2); text-transform:uppercase; letter-spacing:0.5px; }
.stat-sub { font-size:0.75rem; color:var(--muted); font-family:var(--font-mono); margin-top:4px; }

/* ───────── INCOME HERO ───────── */
.income-hero {
  background:linear-gradient(135deg,var(--bg2) 0%,rgba(0,229,160,0.05) 100%);
  border:1px solid var(--border); border-radius:var(--r2); padding:32px;
  margin-bottom:24px; position:relative; overflow:hidden;
}
.income-hero::after {
  content:''; position:absolute; top:-60px; right:-60px;
  width:200px; height:200px; border-radius:50%;
  background:radial-gradient(circle,var(--green-glow),transparent 70%);
}
.income-label { font-size:0.8rem; color:var(--muted2); text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }
.income-tool-row {
  display:flex; gap:16px; align-items:center; flex-wrap:wrap; margin-bottom:16px;
}
.tool-picker {
  background:var(--bg3); border:1px solid var(--border2); border-radius:10px;
  padding:8px 12px; font-family:var(--font-display); font-size:0.9rem;
  cursor:pointer; color:var(--text);
}

/* ───────── FORMS ───────── */
.income-input-row { display:flex; gap:12px; align-items:stretch; flex-wrap:wrap; }
.income-peso { font-family:var(--font-mono); font-size:2rem; font-weight:500; color:var(--green); display:flex; align-items:center; padding:0 8px; }
.income-input {
  flex:2; min-width:200px; font-family:var(--font-mono); font-size:1.8rem; font-weight:500;
  background:transparent; border:none; border-bottom:2px solid var(--border);
  color:var(--text); outline:none; padding:8px 4px; transition:border-color 0.2s;
}
.income-input:focus { border-color:var(--green); }

.mindset-row { display:flex; gap:10px; margin-top:20px; flex-wrap:wrap; }
.mindset-btn {
  padding:8px 20px; border-radius:99px; border:1px solid var(--border2);
  background:transparent; color:var(--muted2); cursor:pointer;
  font-family:var(--font-display); font-size:0.85rem; font-weight:500;
  transition:all 0.2s;
}
.mindset-btn.active { background:var(--green-dim); border-color:var(--green); color:var(--green); }

.btn-analyze {
  background:linear-gradient(135deg,var(--green),#00b87a); color:#000;
  border:none; border-radius:12px; padding:14px 28px; cursor:pointer;
  font-family:var(--font-display); font-weight:700; font-size:0.95rem;
  display:flex; align-items:center; gap:10px; transition:all 0.2s; flex-shrink:0;
  box-shadow:0 4px 20px var(--green-glow);
}
.btn-analyze:hover { transform:translateY(-2px); box-shadow:0 8px 28px var(--green-glow); }
.btn-analyze:disabled { opacity:0.5; cursor:not-allowed; transform:none; }

.btn-add {
  background:linear-gradient(135deg,var(--green),#00b87a); color:#000;
  border:none; border-radius:10px; padding:12px 20px; cursor:pointer;
  font-family:var(--font-display); font-weight:700; font-size:0.9rem;
  transition:all 0.2s; white-space:nowrap;
}
.btn-add:hover { transform:translateY(-1px); }
.btn-add:disabled { opacity:0.5; cursor:not-allowed; }

.btn {
  background:var(--bg3); color:var(--text); border:1px solid var(--border2);
  border-radius:10px; padding:10px 16px; cursor:pointer;
  font-family:var(--font-display); font-weight:600; font-size:0.85rem;
  transition:all 0.2s;
}
.btn:hover { border-color:var(--border); background:var(--bg4); }
.btn-primary { background:var(--green-dim); border-color:var(--border); color:var(--green); }
.btn-danger { background:var(--red-dim); border-color:rgba(255,59,92,0.2); color:var(--red); }

/* ───────── AI CHECKLIST ───────── */
.ai-checklist {
  background:var(--bg3); border-radius:12px; padding:20px; margin-top:20px;
}
.checklist-section { margin-bottom:16px; }
.checklist-section-title { font-size:0.85rem; font-weight:600; color:var(--green); margin-bottom:8px; }
.checklist-item { display:flex; align-items:center; gap:12px; margin-bottom:8px; flex-wrap:wrap; }
.checklist-item label { display:flex; align-items:center; gap:6px; cursor:pointer; }
.checklist-item .cat-percent { font-family:var(--font-mono); font-size:0.8rem; color:var(--muted2); min-width:45px; }
.total-warning { color:var(--red); font-size:0.75rem; margin-top:8px; }
.add-cat-btn {
  background:var(--green-dim); border:1px solid var(--border); color:var(--green);
  border-radius:6px; width:26px; height:26px; display:flex; align-items:center;
  justify-content:center; cursor:pointer; font-size:1.2rem;
}

/* ───────── FORM ELEMENTS ───────── */
.form-group { display:flex; flex-direction:column; gap:6px; }
.form-label { font-size:0.72rem; color:var(--muted2); text-transform:uppercase; letter-spacing:0.5px; }
.form-input, .form-select {
  background:var(--bg3); color:var(--text);
  border:1px solid var(--border2); border-radius:10px;
  padding:11px 14px; outline:none;
  font-family:var(--font-display); font-size:0.9rem;
  transition:border-color 0.2s; width:100%;
}
.form-input:focus, .form-select:focus { border-color:var(--green); }
.form-select option { background:var(--bg2); }
.needwant-group { display:flex; gap:16px; align-items:center; margin-top:8px; }
.needwant-group label { font-size:0.82rem; display:flex; align-items:center; gap:4px; cursor:pointer; }
.needwant-group input[type="radio"] { accent-color:var(--green); }

/* ───────── SAVINGS CARDS ───────── */
.savings-cards { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px; }
.savings-card { background:var(--bg3); border-radius:12px; padding:16px; text-align:center; }
.savings-period { font-size:0.72rem; color:var(--muted2); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px; }
.savings-amount { font-family:var(--font-mono); font-size:1.3rem; font-weight:600; color:var(--green); }

/* ───────── ALLOCATION GRID ───────── */
.alloc-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }
.alloc-item { background:var(--bg3); border-radius:12px; padding:16px; position:relative; overflow:hidden; }
.alloc-item-bar { position:absolute; bottom:0; left:0; height:3px; background:var(--green); transition:width 1s ease; }
.alloc-item-bar.want { background:var(--amber); }
.alloc-item-bar.savings { background:var(--blue); }
.alloc-cat { font-size:0.78rem; color:var(--muted2); margin-bottom:6px; text-transform:uppercase; letter-spacing:0.3px; }
.alloc-pct { font-family:var(--font-mono); font-size:1.4rem; font-weight:600; color:var(--text); }
.alloc-amount { font-family:var(--font-mono); font-size:0.8rem; color:var(--muted2); margin-top:4px; }
.alloc-type { font-size:0.65rem; padding:2px 8px; border-radius:99px; display:inline-block; margin-top:6px; font-weight:600; }
.alloc-type.need { background:var(--green-dim); color:var(--green); }
.alloc-type.want { background:rgba(245,166,35,0.12); color:var(--amber); }
.alloc-type.savings { background:rgba(59,139,255,0.12); color:var(--blue); }

/* ───────── ADVICE ───────── */
.advice-list { display:flex; flex-direction:column; gap:10px; }
.advice-card {
  background:var(--bg3); border-radius:12px; padding:16px; display:flex; gap:12px;
  align-items:flex-start; border-left:3px solid var(--muted);
}
.advice-card.info { border-color:var(--blue); }
.advice-card.warning { border-color:var(--amber); }
.advice-card.success { border-color:var(--green); }
.advice-icon { font-size:1.1rem; flex-shrink:0; }
.advice-title { font-size:0.85rem; font-weight:600; margin-bottom:4px; }
.advice-body { font-size:0.8rem; color:var(--text2); line-height:1.5; }

/* ───────── BUDGET LIST ───────── */
.budget-item {
  display:flex; align-items:center; gap:12px; background:var(--bg3);
  border-radius:10px; padding:10px 14px; margin-bottom:8px;
}
.budget-cat { font-size:0.82rem; width:130px; }
.budget-input {
  width:80px; background:var(--bg4); border:1px solid var(--border2);
  color:var(--text); border-radius:6px; padding:6px 8px;
  font-family:var(--font-mono); font-size:0.8rem;
}
.budget-save-btn {
  background:var(--green-dim); border:1px solid var(--border); color:var(--green);
  border-radius:6px; padding:6px 10px; cursor:pointer; font-size:0.75rem; font-weight:600;
}
.budget-amount { font-family:var(--font-mono); font-size:0.8rem; color:var(--muted2); margin-left:auto; }

/* ───────── TRANSACTION LIST ───────── */
.tx-list { display:flex; flex-direction:column; gap:8px; }
.tx-item {
  display:flex; align-items:center; gap:14px; background:var(--bg3);
  border-radius:12px; padding:14px 16px; transition:background 0.15s;
}
.tx-item:hover { background:var(--bg4); }
.tx-cat-icon { width:36px; height:36px; border-radius:10px; background:var(--bg4); display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
.tx-info { flex:1; }
.tx-cat { font-size:0.88rem; font-weight:500; }
.tx-meta { font-size:0.72rem; color:var(--muted); margin-top:2px; font-family:var(--font-mono); }
.tx-amount { font-family:var(--font-mono); font-weight:600; font-size:0.95rem; flex-shrink:0; }
.tx-amount.income { color:var(--green); }
.tx-amount.expense { color:var(--red); }
.tx-badge { font-size:0.65rem; padding:2px 7px; border-radius:99px; margin-left:6px; font-weight:600; }
.tx-badge.need { background:var(--green-dim); color:var(--green); }
.tx-badge.want { background:rgba(245,166,35,0.12); color:var(--amber); }
.btn-del { background:none; border:none; color:var(--muted); cursor:pointer; font-size:1rem; padding:4px; border-radius:6px; transition:all 0.15s; }
.btn-del:hover { color:var(--red); background:var(--red-dim); }

/* ───────── FUTURE EXPENSES ───────── */
.future-item {
  display:flex; align-items:center; gap:14px; background:var(--bg3);
  border-radius:12px; padding:14px 16px; margin-bottom:8px;
}
.future-info { flex:1; }
.future-desc { font-size:0.88rem; font-weight:500; }
.future-meta { font-size:0.72rem; color:var(--muted); margin-top:2px; font-family:var(--font-mono); }
.future-amount { font-family:var(--font-mono); font-weight:600; font-size:0.9rem; color:var(--amber); }

/* ───────── FORECAST BARS ───────── */
.forecast-bar-row { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.forecast-week-label { font-family:var(--font-mono); font-size:0.75rem; color:var(--green); width:60px; flex-shrink:0; }
.forecast-track { flex:1; height:6px; background:var(--bg3); border-radius:99px; overflow:hidden; }
.forecast-fill { height:100%; background:linear-gradient(90deg,var(--green),#00b87a); border-radius:99px; width:0; transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }
.forecast-val { font-family:var(--font-mono); font-size:0.75rem; color:var(--text2); width:90px; text-align:right; flex-shrink:0; }

/* ───────── LONGEVITY ───────── */
.longevity-display { display:flex; align-items:center; gap:24px; padding:20px 0; flex-wrap:wrap; }
.longevity-days { font-family:var(--font-mono); font-size:3rem; font-weight:600; color:var(--green); line-height:1; }
.longevity-label { color:var(--muted2); font-size:0.85rem; margin-top:6px; }
.longevity-sep { width:1px; height:60px; background:var(--border2); }
.longevity-stat { text-align:center; }
.longevity-stat-val { font-family:var(--font-mono); font-size:1.1rem; font-weight:600; }
.longevity-stat-label { font-size:0.72rem; color:var(--muted); margin-top:4px; }

/* ───────── SCENARIO BARS ───────── */
.scenario-bar-row { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.scenario-label { font-family:var(--font-mono); font-size:0.75rem; color:var(--green); width:70px; flex-shrink:0; }
.scenario-track { flex:1; height:6px; background:var(--bg3); border-radius:99px; overflow:hidden; }
.scenario-fill { height:100%; border-radius:99px; width:0; transition:width 1.2s ease; }
.saver-fill { background:linear-gradient(90deg,var(--blue),var(--green)); }
.neutral-fill { background:linear-gradient(90deg,var(--green),#00b87a); }
.spender-fill { background:linear-gradient(90deg,var(--amber),var(--red)); }
.scenario-val { font-family:var(--font-mono); font-size:0.75rem; color:var(--text2); width:90px; text-align:right; flex-shrink:0; }

/* ───────── CHARTS ───────── */
.chart-wrapper { position:relative; height:220px; }
.vs-chart-container { position:relative; height:280px; width:100%; }

/* ───────── TOAST ───────── */
#toast {
  position:fixed; bottom:28px; left:50%; transform:translateX(-50%) translateY(80px);
  background:var(--bg2); border:1px solid var(--border); border-radius:12px;
  padding:12px 24px; font-size:0.875rem;
  opacity:0; transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
  z-index:9999; white-space:nowrap;
  box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
#toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

/* ───────── AUTH ───────── */
.auth-overlay {
  position:fixed; inset:0;
  background:rgba(6,10,16,0.97); backdrop-filter:blur(20px);
  z-index:9000; display:flex; align-items:center; justify-content:center;
}
.auth-card {
  background:var(--bg2); border:1px solid var(--border); border-radius:24px;
  padding:40px; width:420px; max-width:90%;
  box-shadow:0 24px 80px rgba(0,0,0,0.5);
}
.auth-logo { font-size:2rem; margin-bottom:4px; }
.auth-title { font-size:1.6rem; font-weight:800; margin-bottom:4px; }
.auth-sub { font-size:0.85rem; color:var(--muted2); margin-bottom:28px; }
.auth-input {
  display:block; width:100%;
  background:var(--bg3); color:var(--text);
  border:1px solid var(--border2); border-radius:12px;
  padding:13px 16px; outline:none; font-family:var(--font-display);
  font-size:0.9rem; margin-bottom:12px; transition:border-color 0.2s;
}
.auth-input:focus { border-color:var(--green); }
.btn-auth {
  width:100%; background:linear-gradient(135deg,var(--green),#00b87a);
  color:#000; border:none; border-radius:12px;
  padding:14px; font-family:var(--font-display); font-weight:700;
  font-size:1rem; cursor:pointer; margin-top:8px;
  box-shadow:0 4px 20px var(--green-glow); transition:all 0.2s;
}
.btn-auth:hover { transform:translateY(-2px); }
.auth-toggle { text-align:center; margin-top:16px; font-size:0.85rem; color:var(--muted2); cursor:pointer; }
.auth-toggle span { color:var(--green); font-weight:600; }
.auth-error { color:var(--red); font-size:0.8rem; margin-top:8px; min-height:18px; }
.auth-checkbox { display:flex; align-items:center; gap:8px; margin-bottom:12px; font-size:0.85rem; color:var(--muted2); cursor:pointer; }

/* ───────── CHATBOT ───────── */
.chatbot {
  position:fixed; bottom:20px; right:20px;
  width:360px; height:460px;
  min-width:280px; min-height:300px; max-width:85vw; max-height:70vh;
  resize:both; overflow:auto; z-index:8000;
}
.chat-window {
  width:100%; height:100%;
  background:var(--bg2); border:1px solid var(--border); border-radius:20px;
  display:flex; flex-direction:column; overflow:hidden;
  box-shadow:0 12px 48px rgba(0,0,0,0.4);
}
.chat-header {
  padding:14px 16px; background:var(--bg3); border-bottom:1px solid var(--border2);
  display:flex; align-items:center; gap:10px; cursor:move; user-select:none; flex-shrink:0;
}
.chat-header-title { font-size:0.875rem; font-weight:600; }
.chat-msgs { flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:8px; }
.msg { max-width:84%; padding:10px 14px; border-radius:16px; font-size:0.82rem; line-height:1.5; }
.msg.user { align-self:flex-end; background:var(--green-dim); color:var(--green); border-bottom-right-radius:4px; }
.msg.bot { align-self:flex-start; background:var(--bg3); color:var(--text2); border-bottom-left-radius:4px; }
.chat-input-row { display:flex; padding:12px; gap:8px; background:var(--bg3); border-top:1px solid var(--border2); flex-shrink:0; }
.chat-inp { flex:1; background:var(--bg4); border:1px solid var(--border2); color:var(--text); border-radius:10px; padding:9px 12px; font-family:var(--font-display); font-size:0.82rem; outline:none; transition:border-color 0.2s; }
.chat-inp:focus { border-color:var(--green); }
.chat-send { background:var(--green-dim); border:1px solid var(--border); color:var(--green); border-radius:10px; padding:8px 14px; cursor:pointer; font-weight:600; font-size:0.82rem; transition:all 0.15s; }
.chat-send:hover { background:var(--green); color:#000; }
.chat-toggle-btn { background:none; border:none; color:var(--muted2); cursor:pointer; font-size:1.2rem; padding:0 4px; line-height:1; transition:color 0.2s; margin-left:auto; }
.chat-toggle-btn:hover { color:var(--green); }
.chatbot.minimized .chat-msgs, .chatbot.minimized .chat-input-row { display:none; }
.chatbot.minimized .chat-window { height:auto !important; border-radius:20px; }

/* ───────── EMPTY STATE ───────── */
.empty-state { text-align:center; padding:40px; color:var(--muted); }
.empty-state-icon { font-size:2.5rem; margin-bottom:12px; }
.empty-state-text { font-size:0.9rem; line-height:1.6; }

/* ───────── TABS ───────── */
.tabs { display:flex; gap:4px; background:var(--bg3); border-radius:12px; padding:4px; margin-bottom:20px; }
.tab { flex:1; padding:8px; border:none; background:transparent; color:var(--muted2); border-radius:8px; cursor:pointer; font-family:var(--font-display); font-size:0.82rem; font-weight:500; transition:all 0.2s; }
.tab.active { background:var(--bg2); color:var(--text); box-shadow:0 2px 8px rgba(0,0,0,0.3); }

/* ───────── MODAL ───────── */
.modal-overlay {
  position:fixed; inset:0;
  background:rgba(6,10,16,0.92); backdrop-filter:blur(16px);
  z-index:10000; display:flex; align-items:center; justify-content:center;
}

/* ───────── ADMIN TABLE ───────── */
table { width:100%; border-collapse:collapse; margin-top:8px; }
table th, table td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--border2); }
table th { color:var(--muted2); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; }
table td { color:var(--text2); font-size:0.85rem; }

/* ───────── SCROLLBAR ───────── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:99px; }

.spinner {
  display:inline-block; width:16px; height:16px; border:2px solid rgba(0,229,160,0.3);
  border-top-color:var(--green); border-radius:50%; animation:spin 0.7s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }
</style>
</head>
<body>

<!-- ───── SIDEBAR ───── -->
<nav class="sidebar">
  <div class="logo">💚</div>
  <button class="nav-item active" data-nav="dashboard"><span class="nav-icon">⚡</span><span class="nav-label">Dashboard</span></button>
  <button class="nav-item" data-nav="future"><span class="nav-icon">📌</span><span class="nav-label">Future Expenses</span></button>
  <button class="nav-item" data-nav="insights"><span class="nav-icon">📈</span><span class="nav-label">Insights</span></button>
  <button class="nav-item" data-nav="history"><span class="nav-icon">📋</span><span class="nav-label">History</span></button>
  <button class="nav-item" id="adminNavBtn" data-nav="admin" style="display:none;"><span class="nav-icon">👑</span><span class="nav-label">Admin Panel</span></button>
  <div class="sidebar-sep"></div>
  <button class="nav-item" id="exportBtn"><span class="nav-icon">⬇</span><span class="nav-label">Export CSV</span></button>
</nav>

<!-- ───── MAIN ───── -->
<main class="main">
  <div class="topbar">
    <div class="page-title" id="pageTitle">Dashboard</div>
    <div class="topbar-right">
      <div class="health-pill"><div class="health-dot"></div><span id="topScore" style="font-family:var(--font-mono)">—</span> / 100</div>
      <div class="avatar" id="userAvatar"><span class="avatar-initial">—</span><img src="" style="display:none;"></div>
      <button class="btn-signout" id="signoutBtn">Sign Out</button>
    </div>
  </div>

  <!-- ─── DASHBOARD ─── -->
  <div class="screen active" id="screen-dashboard">
    <div class="income-hero">
      <div class="income-label">Monthly Income & Expenses — Tell ML, it handles the rest</div>
      <div class="income-tool-row">
        <select id="incomeTool" class="tool-picker">
          <option value="manual">📝 Manual Income & Expense</option>
          <option value="auto">📸 Automatic (Image/OCR)</option>
          <option value="profile">👤 Edit Profile</option>
        </select>
        <div style="margin-left:auto;"><span class="ai-badge">ML</span></div>
      </div>

      <!-- Manual Block -->
      <div id="manualBlock">
        <div class="income-input-row" style="align-items:center; gap:16px; margin-bottom:16px;">
          <span style="font-size:0.9rem; color:var(--muted2);">Transaction type:</span>
          <label style="cursor:pointer; display:flex; align-items:center; gap:6px;">
            <input type="radio" name="txMode" value="income" checked onchange="switchTxMode()"> Income
          </label>
          <label style="cursor:pointer; display:flex; align-items:center; gap:6px;">
            <input type="radio" name="txMode" value="expense" onchange="switchTxMode()"> Expense
          </label>
        </div>

        <!-- Income fields -->
        <div id="incomeFields">
          <div class="income-input-row" style="margin-bottom:12px;">
            <label style="font-size:0.8rem; color:var(--muted2);">Date</label>
            <input type="date" id="incomeDate" class="form-input" style="width:160px;">
          </div>
          <div style="display:flex; gap:12px; align-items:stretch; flex-wrap:wrap;">
            <div class="income-peso">₱</div>
            <input class="income-input" id="incomeInput" type="number" placeholder="0.00" step="100" min="0">
            <button class="btn-add" id="addIncomeBtn" style="background:var(--green-dim); color:var(--green); border:1px solid var(--border);">Add Income</button>
          </div>
          <div id="incomeDateWarning" style="display:none; color:var(--red); font-size:0.75rem; margin-top:8px;">Cannot add past income</div>
        </div>

        <!-- Expense fields -->
        <div id="expenseFields" style="display:none;">
          <div class="tx-form" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; align-items:end;">
            <div class="form-group"><label class="form-label">Amount (₱)</label><input class="form-input" id="expenseAmount" type="number" placeholder="0.00" step="0.01" min="1"></div>
            <div class="form-group"><label class="form-label">Category</label>
              <select class="form-select" id="expenseCategory">
                <option>Food & Dining</option><option>Transport</option><option>Groceries</option>
                <option>Entertainment</option><option>Health</option><option>Debt repayment</option>
                <option>Mortgage</option><option>Subscription</option><option>Hobbies</option><option>Other</option>
              </select>
            </div>
            <div class="form-group"><label class="form-label">Note</label><input class="form-input" id="expenseNote" placeholder="Short description"></div>
            <div class="form-group" style="justify-content:flex-end;">
              <div class="needwant-group">
                <label><input type="radio" name="needwant" value="need" checked> Need</label>
                <label><input type="radio" name="needwant" value="want"> Want</label>
              </div>
              <button class="btn-add" id="addExpenseBtn">Add Expense →</button>
            </div>
          </div>
        </div>

        <!-- Spending style -->
        <div class="mindset-row" id="mindsetRow" style="margin-top:20px;">
          <span style="font-size:0.78rem;color:var(--muted);align-self:center;">Spending style:</span>
          <button class="mindset-btn" data-mindset="Saver">🏦 Saver</button>
          <button class="mindset-btn active" data-mindset="Neutral">⚖️ Balanced</button>
          <button class="mindset-btn" data-mindset="Spender">🛍️ Spender</button>
        </div>

        <!-- AI Checklist -->
        <div class="ai-checklist">
          <div style="font-weight:600;margin-bottom:12px;">🧠 ML Autonomous Allocation (100% Sum Rule)</div>
          <div class="checklist-section">
            <div class="checklist-section-title" style="display:flex; align-items:center; gap:10px;">
              Custom Categories
              <button class="add-cat-btn" data-type="want" title="Add custom category">+</button>
            </div>
            <div id="needsChecklist" class="checklist-item"></div>
            <div id="wantsChecklist" class="checklist-item"></div>
          </div>
          <div id="totalWarning" class="total-warning" style="display:none;">⚠️ Total allocation must be 100% – ML will normalise.</div>
          <div style="margin-top:12px;"><button class="btn-analyze" id="analyzeBtn"><span id="analyzeBtnContent">🤖 Let ML Plan</span></button></div>
        </div>
      </div>

      <!-- OCR Upload -->
      <div id="incomeImageUpload" class="income-image-upload" style="display:none;">
        <input type="file" id="incomeImage" accept="image/*" capture="environment">
        <div class="ocr-hint">📸 Take a photo or upload a payslip / budget screenshot. ML will read all income & expenses.</div>
      </div>

      <!-- Profile Edit -->
      <div id="profileBlock" class="profile-block" style="display:none;">
        <div style="font-size:1rem;font-weight:600;margin-bottom:16px;color:var(--green);">Edit Your Profile</div>
        <div class="form-group"><label class="form-label">Name</label><input class="form-input" id="profileName" value=""></div>
        <div class="form-group"><label class="form-label">Email</label><input class="form-input" id="profileEmail" type="email" value=""></div>
        <div class="form-group"><label class="form-label">New Password (leave blank to keep current)</label><input class="form-input" id="profilePass" type="password" placeholder="●●●●●●"></div>
        <div class="form-group">
          <label class="form-label">Profile Picture</label>
          <div style="display:flex; gap:20px; align-items:flex-start;">
            <div style="flex:0 0 100px; height:100px; border-radius:16px; background:var(--bg3); display:flex; align-items:center; justify-content:center; overflow:hidden; border:2px dashed var(--border);">
              <img id="profilePreviewImg" src="" style="width:100%; height:100%; object-fit:cover; display:none;">
              <span id="profilePreviewPlaceholder" style="font-size:2rem; color:var(--muted);">👤</span>
            </div>
            <div style="flex:1; display:flex; flex-direction:column; gap:10px;">
              <label class="btn" style="display:inline-block; width:fit-content; cursor:pointer; font-size:0.8rem; padding:6px 14px;">📁 Upload Photo<input type="file" id="profileFileInput" accept="image/*" style="display:none;"></label>
              <span style="font-size:0.7rem; color:var(--muted2);">or paste a URL below</span>
              <input class="form-input" id="profileAvatar" placeholder="https://example.com/photo.jpg" style="margin-top:4px;">
              <div class="preset-avatars" style="display:flex; gap:8px; margin-top:4px;">
                <div class="preset-avatar" style="background:linear-gradient(135deg,#00E5A0,#009e6a);" data-url="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%2300E5A0'/%3E%3C/svg%3E"></div>
                <div class="preset-avatar" style="background:linear-gradient(135deg,#3B8BFF,#9B59F5);" data-url="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%233B8BFF'/%3E%3C/svg%3E"></div>
                <div class="preset-avatar" style="background:linear-gradient(135deg,#FF3B5C,#F5A623);" data-url="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23FF3B5C'/%3E%3C/svg%3E"></div>
                <div class="preset-avatar" style="background:linear-gradient(135deg,#F5A623,#FF3B5C);" data-url="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23F5A623'/%3E%3C/svg%3E"></div>
              </div>
            </div>
          </div>
        </div>
        <button class="btn-add" id="saveProfileBtn" style="margin-top:8px;">Save Changes</button>
      </div>
    </div>

    <!-- Stats -->
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value" id="sBalance">—</div><div class="stat-label">Balance</div></div>
      <div class="stat-card neg"><div class="stat-value" id="sExpense" style="color:var(--red)">—</div><div class="stat-label">Month Expenses</div></div>
      <div class="stat-card"><div class="stat-value" id="sIncome" style="color:var(--green)">—</div><div class="stat-label">Month Income</div></div>
      <div class="stat-card blue-glow"><div class="stat-value" id="sScore" style="color:var(--blue)">—</div><div class="stat-label">Health Score</div><div class="stat-sub" id="scoreLabel">awaiting data</div></div>
    </div>

    <!-- Income vs Expense Chart -->
    <div class="card chart-card" id="incomeExpenseChartCard" style="display:none;">
      <div class="card-header"><span class="card-title">📊 Income vs Expenses (Current Month)</span><span class="ai-badge">Real-time</span></div>
      <div class="vs-chart-container"><canvas id="incomeExpenseChart"></canvas></div>
    </div>

    <!-- ML Feed -->
    <div class="card ai-feed">
      <div class="ai-feed-header"><div class="ai-pulse"></div><div class="ai-feed-title">ML Activity Feed</div><div class="ai-badge" style="margin-left:auto;">SMART</div></div>
      <div id="aiFeed"><div class="empty-state"><div class="empty-state-icon">🤖</div><div class="empty-state-text">Enter your income above and click <strong style="color:var(--green)">Let ML Plan</strong> — SmartSpend will build your entire financial plan automatically.</div></div></div>
    </div>

    <!-- Plan Toggle -->
    <div id="showPlanToggle" style="display:none; margin-bottom:16px;">
      <button class="btn btn-primary" id="togglePlanBtn">📊 Show Plan Details</button>
    </div>

    <!-- ML Assessment -->
    <div id="financialSummaryBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">ML Assessment</span><span class="ai-badge">ML</span></div>
      <div class="financial-summary-text" id="financialSummaryText"></div>
      <div class="card-title" style="margin-bottom:12px;">Savings Target</div>
      <div class="savings-cards">
        <div class="savings-card"><div class="savings-period">Daily</div><div class="savings-amount" id="saveDaily">—</div></div>
        <div class="savings-card"><div class="savings-period">Weekly</div><div class="savings-amount" id="saveWeekly">—</div></div>
        <div class="savings-card"><div class="savings-period">Monthly</div><div class="savings-amount" id="saveMonthly">—</div></div>
      </div>
      <div id="savingsTip" style="font-size:0.8rem;color:var(--muted2);margin-top:12px;font-style:italic;"></div>
    </div>

    <!-- Allocation -->
    <div id="allocationBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">ML Budget Allocation</span><span class="ai-badge">100% Autonomous</span></div>
      <div class="alloc-grid" id="allocGrid"></div>
    </div>

    <!-- Budgets -->
    <div id="budgetBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">Manage Budgets</span><button class="btn btn-primary" id="resetBudgetsBtn" style="font-size:0.8rem;">⟳ Reset to AI Budgets</button></div>
      <div id="budgetList"></div>
    </div>

    <!-- Advice -->
    <div id="adviceBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">ML Insights</span></div>
      <div class="advice-list" id="adviceList"></div>
    </div>

    <!-- Trend Chart -->
    <div class="card" id="chartBlock" style="display:none">
      <div class="card-header"><span class="card-title">Income vs Expense Trend</span></div>
      <div class="chart-wrapper"><canvas id="trendChart"></canvas></div>
    </div>

    <!-- Forecast -->
    <div class="card" id="forecastBlock" style="display:none">
      <div class="card-header"><span class="card-title">ML Spending Forecast (4 weeks)</span></div>
      <div id="forecastBars"></div>
    </div>
  </div>

  <!-- ─── FUTURE ─── -->
  <div class="screen" id="screen-future">
    <div class="card">
      <div class="card-header"><span class="card-title">Pin Future Expense</span></div>
      <div class="tx-form" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; align-items:end;">
        <div class="form-group" style="flex:2;min-width:200px;"><label class="form-label">Description</label><input class="form-input" id="futureDesc" placeholder="e.g. Rent"></div>
        <div class="form-group"><label class="form-label">Amount (₱)</label><input class="form-input" id="futureAmt" type="number" min="0" step="0.01"></div>
        <div class="form-group"><label class="form-label">Category</label><select class="form-select" id="futureCat"><option>Food & Dining</option><option>Transport</option><option>Groceries</option><option>Health</option><option>Entertainment</option><option>Mortgage</option><option>Debt repayment</option><option>Other</option></select></div>
        <div class="form-group"><label class="form-label">Cycle</label><select class="form-select" id="futureCycle"><option>One-time</option><option>Weekly</option><option>Monthly</option></select></div>
        <div class="form-group"><label class="form-label">Due Date</label><input class="form-input" id="futureDate" type="date"></div>
        <div class="form-group" style="justify-content:flex-end;"><button class="btn-add" id="pinFutureBtn">Pin →</button></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">Pinned Expenses</span><button class="btn btn-primary" id="applyFutureBtn" style="font-size:0.8rem;padding:8px 14px;">⚡ Process Pending</button></div>
      <div id="futureList"></div>
    </div>
  </div>

  <!-- ─── INSIGHTS ─── -->
  <div class="screen" id="screen-insights">
    <div class="card">
      <div class="card-header"><span class="card-title">Budget Longevity (3 Scenarios)</span></div>
      <div id="scenarioBars" style="margin-bottom:20px;"></div>
      <div class="longevity-display" id="longevityDisplay">
        <div><div class="longevity-days" id="longevityDays">—</div><div class="longevity-label">days (current)</div></div>
        <div class="longevity-sep"></div>
        <div class="longevity-stat"><div class="longevity-stat-val" id="longevityBalance">—</div><div class="longevity-stat-label">Balance</div></div>
        <div class="longevity-stat"><div class="longevity-stat-val" id="longevityDaily">—</div><div class="longevity-stat-label">Avg Daily Spend</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">4-Week Spending Forecast <span class="ai-badge" style="margin-left:8px;">ML</span></span></div>
      <div id="forecastBarsInsights"></div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">Category Breakdown (This Month)</span></div>
      <div class="chart-wrapper"><canvas id="catChart"></canvas></div>
    </div>
  </div>

  <!-- ─── HISTORY ─── -->
  <div class="screen" id="screen-history">
    <div class="card">
      <div class="card-header"><span class="card-title">Transaction History</span><button class="btn btn-primary" id="toggleHistoryViewBtn" style="font-size:0.8rem;padding:8px 14px;">🙈 Hide History</button></div>
      <div id="historyContent">
        <div style="display:flex; gap:8px; margin-bottom:12px; align-items:center;">
          <input class="form-input" id="historySearch" placeholder="Search…" style="width:180px;padding:8px 12px;font-size:0.82rem;">
          <label style="font-size:0.78rem; color:var(--muted2); display:flex; align-items:center; gap:6px;"><input type="checkbox" id="showNeedWantBadges" checked> Show Need/Want badges</label>
        </div>
        <div class="tx-list" id="historyList"></div>
      </div>
    </div>
  </div>

  <!-- ─── ADMIN ─── -->
  <div class="screen" id="screen-admin">
    <div class="card">
      <div class="card-header">👑 Admin Dashboard</div>
      <div class="stats-grid" id="adminStats"></div>
      <div class="tabs" id="adminTabs">
        <button class="tab active" data-tab="users">📋 Users</button>
        <button class="tab" data-tab="transactions">💰 All Transactions</button>
      </div>
      <div id="adminUsersPanel">
        <input type="text" id="adminSearchUser" placeholder="Search user..." class="form-input" style="margin-bottom:12px;">
        <table>
          <thead>
            <tr><th>ID</th><th>Name</th><th>Email</th><th>Role</th><th>Created</th></tr>
          </thead>
          <tbody id="adminUserTable"></tbody>
        </table>
      </div>
      <div id="adminTransactionsPanel" style="display:none;">
        <select id="adminUserFilter" class="form-select" style="margin-bottom:12px;"><option value="">All Users</option></select>
        <table>
          <thead>
            <tr><th>Date</th><th>User</th><th>Category</th><th>Type</th><th>Amount</th><th>Need/Want</th></tr>
          </thead>
          <tbody id="adminTxTable"></tbody>
        </table>
      </div>
    </div>
  </div>
</main>

<!-- ───── TOAST ───── -->
<div id="toast"></div>

<!-- ───── AUTH ───── -->
<div id="authOverlay" class="auth-overlay">
  <div class="auth-card">
    <div class="auth-logo">💚</div>
    <div class="auth-title" id="authTitle">Welcome back</div>
    <div class="auth-sub" id="authSub">Sign in to your SmartSpend account</div>
    <input class="auth-input" id="regName" placeholder="Full Name" style="display:none">
    <input class="auth-input" id="authEmail" placeholder="Email address" type="email">
    <input class="auth-input" id="authPass" placeholder="Password" type="password">
    <input class="auth-input" id="authConfirm" placeholder="Confirm Password" type="password" style="display:none">
    <label class="auth-checkbox" id="termsRow" style="display:none"><input type="checkbox" id="termsCheck"> I accept the Terms of Service</label>
    <div class="auth-error" id="authMsg"></div>
    <button class="btn-auth" id="authBtn">Sign In</button>
    <div class="auth-toggle" id="toggleAuth">No account? <span>Register here</span></div>
  </div>
</div>

<!-- ───── CHATBOT ───── -->
<div class="chatbot" id="chatbot">
  <div class="chat-window">
    <div class="chat-header" id="chatHeader"><div class="ai-pulse"></div><div class="chat-header-title">SmartSpend ML <span class="ai-badge">Smart</span></div><button class="chat-toggle-btn" id="chatToggleBtn" title="Minimize">–</button></div>
    <div class="chat-msgs" id="chatMsgs"><div class="msg bot">👋 I'm your ML finance assistant. Ask me anything about your money, budget, or how to save more.</div></div>
    <div class="chat-input-row"><input class="chat-inp" id="chatInp" placeholder="Ask anything…"><button class="chat-send" id="chatSend">→</button></div>
  </div>
</div>

<script>
// ── GLOBAL VARS ──
let currentUser = null;
let allTransactions = [];
let currentMindset = 'Neutral';
let aiPlan = null;
let trendChart = null, catChartInst = null, incomeExpenseChart = null;
let isLogin = true;
let historyVisible = true;
let customCategories = [];

// ── OPTIMISATION FLAGS ──
let futureExpensesApplied = false;
let refreshTimeout = null;
let dashboardLoading = false;
let planAttempted = false;          // ← NEW: only try AI plan once
let isAnalyzing = false;            // ← NEW: prevent concurrent ML calls

const CAT_ICONS = {
  'Food & Dining':'🍜','Transport':'🚗','Groceries':'🛒','Entertainment':'🎬',
  'Health':'💊','Debt repayment':'💳','Mortgage':'🏠','Subscription':'📱',
  'Hobbies':'🎮','Salary':'💰','Savings':'🏦','Other':'📦'
};

const categoryConfig = [
  { name: 'Food & Dining', type: 'need', defaultPct: 0 },
  { name: 'Debt repayment', type: 'need', defaultPct: 0 },
  { name: 'Mortgage', type: 'need', defaultPct: 0 },
  { name: 'Transport', type: 'need', defaultPct: 0 },
  { name: 'Groceries', type: 'need', defaultPct: 0 },
  { name: 'Health', type: 'need', defaultPct: 0 },
  { name: 'Savings', type: 'savings', defaultPct: 0 },
  { name: 'Entertainment', type: 'want', defaultPct: 0 },
  { name: 'Subscription', type: 'want', defaultPct: 0 },
  { name: 'Hobbies', type: 'want', defaultPct: 0 }
];

function fmt(n){ return '₱'+Number(n||0).toLocaleString('en-PH',{minimumFractionDigits:2,maximumFractionDigits:2}); }
function fmtDate(iso){ return new Date(iso).toLocaleDateString('en-PH',{month:'short',day:'numeric',year:'2-digit'}); }
function esc(s){ return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function toast(msg, color=''){
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.borderColor = color || 'var(--border)';
  t.classList.add('show'); setTimeout(()=>t.classList.remove('show'), 2800);
}

async function api(url, opts={}){
  const res = await fetch(url, { ...opts, credentials:'include', headers:{'Content-Type':'application/json',...(opts.headers||{})} });
  if(!res.ok){ const err = await res.json().catch(()=>({error:'Request failed'})); throw new Error(err.error || 'Request failed'); }
  return res.json();
}

// ── HELPER: get selected categories (with fallback) ──
function getSelectedCategories() {
  const selected = [];
  document.querySelectorAll('.cat-checkbox:checked').forEach(chk => selected.push(chk.dataset.cat));
  if (selected.length === 0) {
    // fallback to sensible defaults
    return ['Food & Dining','Transport','Groceries','Health','Entertainment','Debt repayment','Savings'];
  }
  return selected;
}

// ── DEBOUNCED REFRESH (only refreshes data, never retriggers plan) ──
function debouncedRefresh() {
  if (refreshTimeout) clearTimeout(refreshTimeout);
  refreshTimeout = setTimeout(() => {
    loadDashboardData();   // ← calls a non‑plan triggering version
    refreshTimeout = null;
  }, 800);
}

// ── CHECKLIST ──
function renderChecklist() {
  const needsCont = document.getElementById('needsChecklist');
  const wantsCont = document.getElementById('wantsChecklist');
  needsCont.innerHTML = ''; wantsCont.innerHTML = '';
  categoryConfig.forEach(cat => addCategoryCheckbox(cat));
  customCategories.forEach(cat => addCategoryCheckbox(cat));
}

function addCategoryCheckbox(cat) {
  const target = (cat.type === 'need' || cat.type === 'savings') ?
    document.getElementById('needsChecklist') : document.getElementById('wantsChecklist');
  const div = document.createElement('div');
  div.style.display = 'flex'; div.style.alignItems = 'center'; div.style.gap = '6px'; div.style.marginBottom = '6px';
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox'; checkbox.className = 'cat-checkbox'; checkbox.dataset.cat = cat.name; checkbox.checked = true;
  const label = document.createElement('label');
  label.style.fontSize = '0.82rem'; label.style.color = 'var(--text2)'; label.textContent = cat.name;
  const percentSpan = document.createElement('span');
  percentSpan.className = 'cat-percent'; percentSpan.id = 'pct-'+cat.name.replace(/\s/g,''); percentSpan.textContent = '';
  div.appendChild(checkbox); div.appendChild(label); div.appendChild(percentSpan);
  target.appendChild(div);
}

document.querySelectorAll('.add-cat-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    const name = prompt('Enter a custom category name:');
    if (!name || name.trim() === '') return;
    const trimmed = name.trim();
    const exists = categoryConfig.concat(customCategories).some(c => c.name.toLowerCase() === trimmed.toLowerCase());
    if (exists) { toast('Category already exists'); return; }
    customCategories.push({ name: trimmed, type: 'want', defaultPct: 0 });
    renderChecklist();
  });
});

// ── TX MODE ──
function switchTxMode() {
  const mode = document.querySelector('input[name="txMode"]:checked').value;
  document.getElementById('incomeFields').style.display = mode === 'income' ? 'block' : 'none';
  document.getElementById('expenseFields').style.display = mode === 'expense' ? 'block' : 'none';
  document.getElementById('mindsetRow').style.display = mode === 'income' ? 'flex' : 'none';
}

// ── DATE VALIDATION ──
document.getElementById('incomeDate').addEventListener('change', function() {
  const today = new Date().toISOString().slice(0,10);
  const warning = document.getElementById('incomeDateWarning');
  warning.style.display = (this.value < today && this.value !== '') ? 'block' : 'none';
});
document.getElementById('incomeInput').addEventListener('input', function() {
  const income = parseFloat(this.value) || 0;
  if (income <= 10000) setActiveMindset('Saver');
  else if (income <= 50000) setActiveMindset('Neutral');
  else setActiveMindset('Spender');
});
function setActiveMindset(mindset) {
  document.querySelectorAll('.mindset-btn').forEach(b => b.classList.remove('active'));
  const btn = document.querySelector(`.mindset-btn[data-mindset="${mindset}"]`);
  if (btn) btn.classList.add('active');
  currentMindset = mindset;
}
document.querySelectorAll('.mindset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mindset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMindset = btn.dataset.mindset;
  });
});

// ── AUTH ──
const authOverlay = document.getElementById('authOverlay');
const authBtn = document.getElementById('authBtn');
const toggleAuth = document.getElementById('toggleAuth');
const authMsg = document.getElementById('authMsg');
const authEmail = document.getElementById('authEmail');
const authPass = document.getElementById('authPass');
const authConfirm = document.getElementById('authConfirm');
const regName = document.getElementById('regName');
const termsRow = document.getElementById('termsRow');
const termsCheck = document.getElementById('termsCheck');
const authTitle = document.getElementById('authTitle');
const authSub = document.getElementById('authSub');

toggleAuth.addEventListener('click', () => {
  isLogin = !isLogin;
  if(isLogin) {
    authTitle.textContent = 'Welcome back'; authSub.textContent = 'Sign in to your SmartSpend account';
    authBtn.textContent = 'Sign In'; regName.style.display='none'; authConfirm.style.display='none';
    termsRow.style.display='none'; toggleAuth.innerHTML = 'No account? <span>Register here</span>';
  } else {
    authTitle.textContent = 'Create account'; authSub.textContent = 'Start your ML‑powered financial journey';
    authBtn.textContent = 'Register'; regName.style.display='block'; authConfirm.style.display='block';
    termsRow.style.display='flex'; toggleAuth.innerHTML = 'Already registered? <span>Sign in</span>';
  }
  authMsg.textContent = '';
});

authBtn.addEventListener('click', async () => {
  authMsg.textContent = '';
  if(!isLogin && authPass.value !== authConfirm.value) { authMsg.textContent = 'Passwords do not match'; return; }
  if(!isLogin && !termsCheck.checked) { authMsg.textContent = 'You must accept the terms'; return; }
  try {
    const endpoint = isLogin ? '/api/login' : '/api/register';
    const body = isLogin ? { email: authEmail.value, password: authPass.value } :
      { name: regName.value, email: authEmail.value, password: authPass.value };
    const data = await api(endpoint, { method:'POST', body: JSON.stringify(body) });
    currentUser = data; authOverlay.style.display = 'none'; initApp();
  } catch(e) { authMsg.textContent = e.message; }
});

// ── SIGN OUT ──
document.getElementById('signoutBtn').addEventListener('click', async () => {
  await api('/api/logout', { method:'POST' });
  currentUser = null; authOverlay.style.display = 'flex';
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('screen-dashboard').classList.add('active');
  toast('Signed out');
});

// ── NAV ──
document.querySelectorAll('.nav-item[data-nav]').forEach(btn => {
  btn.addEventListener('click', ()=>{
    const nav = btn.dataset.nav;
    document.querySelectorAll('.nav-item').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
    document.getElementById('screen-'+nav).classList.add('active');
    document.getElementById('pageTitle').textContent = btn.querySelector('.nav-label').textContent || 'Dashboard';
    if(nav==='dashboard') loadDashboard();
    else if(nav==='insights') loadInsights();
    else if(nav==='future') loadFutureExpenses();
    else if(nav==='history') loadHistory();
    else if(nav==='admin') loadAdmin();
  });
});

// ── TOOL PICKER ──
document.getElementById('incomeTool').addEventListener('change', function(){
  const val = this.value;
  document.getElementById('manualBlock').style.display = (val === 'manual') ? 'block' : 'none';
  document.getElementById('incomeImageUpload').style.display = (val === 'auto') ? 'block' : 'none';
  document.getElementById('profileBlock').style.display = (val === 'profile') ? 'block' : 'none';
  if (val === 'profile' && currentUser) {
    document.getElementById('profileName').value = currentUser.name || '';
    document.getElementById('profileEmail').value = currentUser.email || '';
    document.getElementById('profilePass').value = '';
    document.getElementById('profileAvatar').value = currentUser.avatar_url || '';
    if (currentUser.avatar_url) {
      document.getElementById('profilePreviewImg').src = currentUser.avatar_url;
      document.getElementById('profilePreviewImg').style.display = 'block';
      document.getElementById('profilePreviewPlaceholder').style.display = 'none';
    } else {
      document.getElementById('profilePreviewImg').style.display = 'none';
      document.getElementById('profilePreviewPlaceholder').style.display = 'block';
    }
  }
});

// ── INIT ──
async function initApp() {
  if(!currentUser) return;
  const avatar = document.getElementById('userAvatar');
  const initialSpan = avatar.querySelector('.avatar-initial');
  const imgEl = avatar.querySelector('img');
  if (currentUser.avatar_url) {
    imgEl.src = currentUser.avatar_url; imgEl.style.display = 'block';
    initialSpan.style.display = 'none';
  } else {
    imgEl.style.display = 'none'; initialSpan.style.display = 'block';
    initialSpan.textContent = currentUser.name?.charAt(0)?.toUpperCase() || '?';
  }
  if(currentUser.role === 'admin') document.getElementById('adminNavBtn').style.display = 'flex';
  else document.getElementById('adminNavBtn').style.display = 'none';
  document.getElementById('incomeDate').value = new Date().toISOString().slice(0,10);
  renderChecklist();
  loadDashboard();
}

// ── ADD INCOME ──
document.getElementById('addIncomeBtn').addEventListener('click', async ()=>{
  const amount = parseFloat(document.getElementById('incomeInput').value);
  if(!amount || amount < 1) { toast('Enter a valid amount (min 1)'); return; }
  const date = document.getElementById('incomeDate').value;
  if (!date) { toast('Please select a date'); return; }
  try {
    await api('/api/transactions', { method:'POST', body: JSON.stringify({
      amount, category:'Salary', tx_type:'income', is_need:true, note:'Manual income',
      tx_date: date
    })});
    toast('Income added!');
    await autoRunPlanAndUpdateDashboard();
  } catch(e) { toast(e.message); }
});

// ── ADD EXPENSE ──
document.getElementById('addExpenseBtn').addEventListener('click', async ()=>{
  const amount = parseFloat(document.getElementById('expenseAmount').value);
  if(!amount || amount < 1) { toast('Amount must be at least 1'); return; }
  const category = document.getElementById('expenseCategory').value;
  const note = document.getElementById('expenseNote').value;
  const isNeed = document.querySelector('input[name="needwant"]:checked').value === 'need';
  try {
    await api('/api/transactions', { method:'POST', body: JSON.stringify({
      amount, category, tx_type:'expense', is_need:isNeed, note
    })});
    toast('Expense added!');
    document.getElementById('expenseAmount').value = '';
    document.getElementById('expenseNote').value = '';
    debouncedRefresh();
  } catch(e) { toast(e.message); }
});

// ── DASHBOARD DATA ONLY (no plan triggering) ──
async function loadDashboardData() {
  if (!currentUser || dashboardLoading) return;
  dashboardLoading = true;
  try {
    if (!futureExpensesApplied) {
      await api('/api/apply_future_expenses', { method: 'POST' });
      futureExpensesApplied = true;
    }
    const summary = await api(`/api/summary/${currentUser.id}`);
    const predict = await api(`/api/predict/${currentUser.id}`);
    const longevity = await api(`/api/longevity/${currentUser.id}`);
    renderStats(summary, predict.score, longevity);
    renderForecast(predict.predictions?.weekly);
    renderTrendChart(summary.monthly);
    renderIncomeExpenseChart(summary.income, summary.expense);
    document.getElementById('incomeExpenseChartCard').style.display = 'block';
    document.getElementById('chartBlock').style.display = Object.keys(summary.monthly).length ? 'block' : 'none';
    document.getElementById('forecastBlock').style.display = Object.keys(predict.predictions?.weekly||{}).length ? 'block' : 'none';
  } catch(e) {} finally { dashboardLoading = false; }
}

// ── FULL DASHBOARD LOAD (only called once, tries plan) ──
async function loadDashboard() {
  await loadDashboardData();
  if (!planAttempted && currentUser && currentUser.monthly_budget_limit > 0) {
    planAttempted = true;
    autoRunPlanAndUpdateDashboard();
  }
}

// ── AUTO PLAN (with lock and proper error handling) ──
async function autoRunPlanAndUpdateDashboard() {
  if (!currentUser || dashboardLoading || isAnalyzing) return;
  const totalIncome = currentUser.monthly_budget_limit;
  if (totalIncome <= 0) return;
  
  isAnalyzing = true;
  dashboardLoading = true;
  addFeedEvent('🤖', 'ML is analyzing your financial profile…');
  
  try {
    const selectedCategories = getSelectedCategories();
    const result = await api('/api/ai/full_setup', {
      method: 'POST',
      body: JSON.stringify({ 
        monthly_income: totalIncome, 
        mindset: currentMindset, 
        selected_categories: selectedCategories 
      })
    });
    if (!result.allocation) throw new Error('No allocation returned from ML');
    aiPlan = result;
    addFeedEvent('✅', `AI categorised ${Object.keys(result.allocation).length} categories`);
    addFeedEvent('💰', `Savings target: ${fmt(result.savings_plan.monthly)}/month`);
    addFeedEvent('🧠', `Summary: "${result.financial_summary.substring(0,60)}…"`);
    addFeedEvent('📋', `${result.advice.length} insights ready`);
    renderAIPlan(result);
  } catch(e) {
    console.error('Auto plan error:', e);
    let errMsg = e.message;
    if (errMsg.includes('selected_categories is not defined')) {
      errMsg = 'Backend error: selected_categories variable missing. Please check your server code.';
    }
    toast('ML analysis failed: ' + errMsg, 'var(--red)');
    addFeedEvent('❌', 'ML error: ' + errMsg);
  } finally {
    dashboardLoading = false;
    isAnalyzing = false;
    loadDashboardData();
  }
}

// ── MANUAL ML PLAN (button click) ──
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  if (isAnalyzing) { toast('ML analysis already running...', 'var(--amber)'); return; }
  const income = parseFloat(document.getElementById('incomeInput').value);
  if(!income || income <= 0){ toast('Enter a valid monthly income first'); return; }
  const selectedCategories = getSelectedCategories();
  if(!selectedCategories.length){ toast('Please select at least one category'); return; }
  
  const btn = document.getElementById('analyzeBtn');
  const btnContent = document.getElementById('analyzeBtnContent');
  btn.disabled = true;
  btnContent.innerHTML = '<div class="spinner"></div> Analyzing…';
  addFeedEvent('🤖','ML is analyzing your financial profile…');
  isAnalyzing = true;
  
  try {
    const result = await api('/api/ai/full_setup', {
      method:'POST',
      body: JSON.stringify({ monthly_income: income, mindset: currentMindset, selected_categories: selectedCategories })
    });
    if (!result.allocation) throw new Error('No allocation returned from ML');
    aiPlan = result;
    addFeedEvent('✅',`AI categorised ${Object.keys(result.allocation).length} categories`);
    addFeedEvent('💰',`Savings target: ${fmt(result.savings_plan.monthly)}/month`);
    addFeedEvent('🧠',`Summary: "${result.financial_summary.substring(0,60)}…"`);
    addFeedEvent('📋',`${result.advice.length} insights ready`);
    renderAIPlan(result);
    toast('ML plan complete! 🎉', 'var(--green)');
  } catch(e){
    console.error('ML plan error:', e);
    let errMsg = e.message;
    if (errMsg.includes('selected_categories is not defined')) {
      errMsg = 'Backend error: selected_categories variable missing. Please check your server code.';
    }
    toast('ML error: '+errMsg);
    addFeedEvent('❌','ML error: '+errMsg);
  } finally {
    isAnalyzing = false;
    btn.disabled = false;
    btnContent.innerHTML = '🔄 Re‑Analyze';
    loadDashboardData();
  }
});

function addFeedEvent(emoji, text) {
  const feedDiv = document.getElementById('aiFeed');
  const existing = feedDiv.querySelector('.empty-state');
  if (existing) existing.remove();
  feedDiv.innerHTML += `<div style="padding:8px 0; border-bottom:1px solid var(--border2); font-size:0.8rem;">${emoji} ${esc(text)}</div>`;
  feedDiv.scrollTop = feedDiv.scrollHeight;
}

function renderStats(summary, score, longevity) {
  document.getElementById('sBalance').innerHTML = fmt(longevity?.balance || 0);
  document.getElementById('sExpense').innerHTML = fmt(summary.expense);
  document.getElementById('sIncome').innerHTML = fmt(summary.income);
  document.getElementById('sScore').innerHTML = score !== undefined ? Math.round(score) : '—';
  document.getElementById('topScore').innerHTML = score !== undefined ? Math.round(score) : '—';
  let label = '';
  if (score >= 80) label = 'Excellent financial health';
  else if (score >= 60) label = 'Good, but room to improve';
  else if (score >= 40) label = 'Needs attention';
  else label = 'Critical – adjust spending';
  document.getElementById('scoreLabel').innerHTML = label;
}

function renderIncomeExpenseChart(income, expense) {
  const ctx = document.getElementById('incomeExpenseChart');
  if (!ctx) return;
  if (incomeExpenseChart) incomeExpenseChart.destroy();
  incomeExpenseChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: ['Income', 'Expense'], datasets: [{ data: [income, expense], backgroundColor: ['#00E5A0', '#FF3B5C'], borderRadius: 8 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
  });
}

function renderTrendChart(monthly) {
  const ctx = document.getElementById('trendChart');
  if(!ctx || !monthly) return;
  if(trendChart) trendChart.destroy();
  const labels = Object.keys(monthly);
  const incomeData = labels.map(m=>monthly[m].income);
  const expenseData = labels.map(m=>monthly[m].expense);
  trendChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [
      { label: 'Income', data: incomeData, borderColor: '#00E5A0', backgroundColor: 'rgba(0,229,160,0.1)', tension:0.3 },
      { label: 'Expense', data: expenseData, borderColor: '#FF3B5C', backgroundColor: 'rgba(255,59,92,0.1)', tension:0.3 }
    ] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#B0C8E0'}}} }
  });
}

function renderForecast(weekly) {
  const container = document.getElementById('forecastBars');
  if(!weekly) { container.innerHTML = ''; return; }
  const maxVal = Math.max(...Object.values(weekly), 1);
  container.innerHTML = Object.entries(weekly).map(([week, val])=>{
    const pct = (val / maxVal * 100).toFixed(0);
    return `<div class="forecast-bar-row"><span class="forecast-week-label">${week}</span><div class="forecast-track"><div class="forecast-fill" style="width:${pct}%"></div></div><span class="forecast-val">${fmt(val)}</span></div>`;
  }).join('');
}

function renderAIPlan(plan) {
  document.getElementById('financialSummaryText').textContent = plan.financial_summary;
  document.getElementById('saveDaily').textContent = fmt(plan.savings_plan.daily);
  document.getElementById('saveWeekly').textContent = fmt(plan.savings_plan.weekly);
  document.getElementById('saveMonthly').textContent = fmt(plan.savings_plan.monthly);
  document.getElementById('savingsTip').textContent = '💡 ' + (plan.savings_plan.tip || '');
  const grid = document.getElementById('allocGrid');
  const needsSet = new Set(['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage']);
  const savingsSet = new Set(['Savings']);
  grid.innerHTML = Object.entries(plan.allocation).map(([cat, pct])=>{
    const type = savingsSet.has(cat) ? 'savings' : (needsSet.has(cat) ? 'need' : 'want');
    const amt = plan.allocation_amounts?.[cat] || (plan.monthly_income * pct / 100);
    return `<div class="alloc-item">
      <div class="alloc-cat">${esc(cat)}</div><div class="alloc-pct">${pct.toFixed(0)}<span style="font-size:1rem;color:var(--muted)">%</span></div>
      <div class="alloc-amount">${fmt(amt)}</div><div class="alloc-type ${type}">${type.toUpperCase()}</div>
      <div class="alloc-item-bar ${type}" style="width:${Math.min(pct,100)}%"></div></div>`;
  }).join('');
  const adviceIcons = { info:'ℹ️', warning:'⚠️', success:'✅' };
  document.getElementById('adviceList').innerHTML = plan.advice.map(a=>
    `<div class="advice-card ${esc(a.type)}"><span class="advice-icon">${adviceIcons[a.type]||'💡'}</span>
      <div><div class="advice-title">${esc(a.title)}</div><div class="advice-body">${esc(a.body)}</div></div></div>`).join('');
  const blocks = ['financialSummaryBlock','allocationBlock','adviceBlock'];
  blocks.forEach(id => { document.getElementById(id).style.display = 'block'; });
  const toggleBtn = document.getElementById('showPlanToggle');
  const togglePlanBtn = document.getElementById('togglePlanBtn');
  toggleBtn.style.display = 'block';
  let detailsVisible = true;
  togglePlanBtn.textContent = '🔽 Hide Plan Details';
  togglePlanBtn.onclick = () => {
    detailsVisible = !detailsVisible;
    blocks.forEach(id => { document.getElementById(id).style.display = detailsVisible ? 'block' : 'none'; });
    togglePlanBtn.textContent = detailsVisible ? '🔽 Hide Plan Details' : '📊 Show Plan Details';
  };
  loadBudgets();
}

// ── BUDGETS ──
async function loadBudgets() {
  if (!currentUser) return;
  try { const budgets = await api(`/api/budgets/${currentUser.id}`); renderBudgetList(budgets); } catch(e) { toast(e.message); }
}
function renderBudgetList(budgets) {
  const container = document.getElementById('budgetList');
  if (!budgets.length) { container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-text">No budgets set yet. Run AI Plan first.</div></div>'; return; }
  container.innerHTML = budgets.map(b => {
    const safeId = b.category.replace(/[^a-zA-Z0-9]/g, '_');
    return `<div class="budget-item" id="budget-${safeId}">
      <div class="budget-cat">${esc(b.category)}</div>
      <input class="budget-input" id="budget-input-${safeId}" type="number" step="0.01" value="${b.limit.toFixed(2)}">
      <button class="budget-save-btn" onclick="saveBudget('${b.category.replace(/'/g, "\\'")}', '${safeId}')">Save</button>
      <span class="budget-amount">${fmt(b.limit)}</span>
    </div>`;
  }).join('');
  document.getElementById('budgetBlock').style.display = 'block';
}
async function saveBudget(category, safeId) {
  const input = document.getElementById('budget-input-'+safeId);
  const newLimit = parseFloat(input.value);
  if (isNaN(newLimit)) return;
  try { await api(`/api/budgets/${currentUser.id}/${encodeURIComponent(category)}`, { method:'PUT', body: JSON.stringify({ limit: newLimit }) }); toast('Budget updated'); loadBudgets(); } catch(e) { toast(e.message); }
}
document.getElementById('resetBudgetsBtn').addEventListener('click', async () => {
  if (!currentUser) return;
  try { await api(`/api/budgets/reset_to_ai/${currentUser.id}`, { method:'POST' }); toast('Budgets reset to AI recommendations'); loadBudgets(); } catch(e) { toast(e.message); }
});

// ── INSIGHTS ──
async function loadInsights() {
  if(!currentUser) return;
  try {
    const longevity = await api(`/api/longevity/${currentUser.id}`);
    const bal = longevity.balance; const avgDaily = longevity.avg_daily_spend;
    const scenarios = [
      { label: '🏦 Saver', daily: avgDaily * 0.8, cssClass: 'saver-fill' },
      { label: '⚖️ Balanced', daily: avgDaily, cssClass: 'neutral-fill' },
      { label: '🛍️ Spender', daily: avgDaily * 1.2, cssClass: 'spender-fill' }
    ];
    const maxDays = Math.max(...scenarios.map(s => bal / s.daily), 1);
    document.getElementById('scenarioBars').innerHTML = scenarios.map(s => {
      const days = Math.floor(bal / s.daily);
      const pct = Math.min((days / maxDays) * 100, 100);
      return `<div class="scenario-bar-row"><span class="scenario-label">${s.label}</span><div class="scenario-track"><div class="scenario-fill ${s.cssClass}" style="width:${pct}%"></div></div><span class="scenario-val">${days} days</span></div>`;
    }).join('');
    document.getElementById('longevityDays').textContent = scenarios[1].daily > 0 ? Math.floor(bal / scenarios[1].daily) : '—';
    document.getElementById('longevityBalance').textContent = fmt(bal);
    document.getElementById('longevityDaily').textContent = fmt(avgDaily);
    const predict = await api(`/api/predict/${currentUser.id}`);
    renderForecastInsights(predict.predictions?.weekly);
    renderCategoryChart(predict.predictions?.categories);
  } catch(e) { toast(e.message); }
}
function renderForecastInsights(weekly) {
  const container = document.getElementById('forecastBarsInsights');
  if(!weekly) { container.innerHTML = ''; return; }
  const maxVal = Math.max(...Object.values(weekly), 1);
  container.innerHTML = Object.entries(weekly).map(([week, val])=>{
    const pct = (val / maxVal * 100).toFixed(0);
    return `<div class="forecast-bar-row"><span class="forecast-week-label">${week}</span><div class="forecast-track"><div class="forecast-fill" style="width:${pct}%"></div></div><span class="forecast-val">${fmt(val)}</span></div>`;
  }).join('');
}
function renderCategoryChart(categories) {
  const ctx = document.getElementById('catChart');
  if(!ctx) return;
  if(catChartInst) catChartInst.destroy();
  const labels = Object.keys(categories||{});
  const data = Object.values(categories||{});
  catChartInst = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: ['#00E5A0','#3B8BFF','#F5A623','#FF3B5C','#9B59F5','#00b87a','#FF8C00'] }] },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#B0C8E0'}}} }
  });
}

// ── HISTORY ──
async function loadHistory() {
  if(!currentUser) return;
  try { const txs = await api('/api/transactions'); allTransactions = txs; renderHistory(txs); } catch(e) { toast(e.message); }
}
function renderHistory(txs) {
  const list = document.getElementById('historyList');
  const search = document.getElementById('historySearch').value.toLowerCase();
  const showBadges = document.getElementById('showNeedWantBadges').checked;
  const filtered = txs.filter(t=> t.category.toLowerCase().includes(search) || (t.note||'').toLowerCase().includes(search));
  if(!filtered.length) { list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><div class="empty-state-text">No matching transactions.</div></div>'; return; }
  list.innerHTML = filtered.map(t=>{
    const badgeHtml = showBadges ? (t.is_need ? '<span class="tx-badge need">Need</span>' : '<span class="tx-badge want">Want</span>') : '';
    return `<div class="tx-item">
      <div class="tx-cat-icon">${CAT_ICONS[t.category]||'📦'}</div>
      <div class="tx-info"><div class="tx-cat">${esc(t.category)} ${badgeHtml}</div><div class="tx-meta">${t.note?esc(t.note)+' · ':''}${fmtDate(t.tx_date)}</div></div>
      <div class="tx-amount ${t.tx_type}">${t.tx_type==='income'?'+':'-'}${fmt(t.amount)}</div>
      <button class="btn-del" onclick="deleteTransaction(${t.id})">🗑</button>
    </div>`;
  }).join('');
}
document.getElementById('historySearch').addEventListener('input', ()=> renderHistory(allTransactions));
document.getElementById('showNeedWantBadges').addEventListener('change', ()=> renderHistory(allTransactions));
document.getElementById('toggleHistoryViewBtn').addEventListener('click', () => {
  const content = document.getElementById('historyContent');
  historyVisible = !historyVisible;
  content.style.display = historyVisible ? 'block' : 'none';
});
async function deleteTransaction(id) {
  if(!confirm('Delete this transaction?')) return;
  try { await api(`/api/transactions/${id}`, { method:'DELETE' }); toast('Deleted'); if(document.getElementById('screen-history').classList.contains('active')) loadHistory(); } catch(e) { toast(e.message); }
}

// ── FUTURE EXPENSES ──
async function loadFutureExpenses() {
  if(!currentUser) return;
  try { const exps = await api('/api/future_expenses'); renderFutureList(exps); } catch(e) { toast(e.message); }
}
function renderFutureList(exps) {
  const list = document.getElementById('futureList');
  if(!exps.length) { list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📌</div><div class="empty-state-text">No pinned expenses yet.</div></div>'; return; }
  list.innerHTML = exps.map(e=>`
    <div class="future-item">
      <div class="future-info"><div class="future-desc">${esc(e.description)}</div><div class="future-meta">${esc(e.category)} · ${e.cycle} · ${e.date}</div></div>
      <div class="future-amount">${fmt(e.amount)}</div>
      <button class="btn-del" onclick="deleteFuture(${e.id})">🗑</button>
    </div>`).join('');
}
document.getElementById('pinFutureBtn').addEventListener('click', async ()=>{
  const desc = document.getElementById('futureDesc').value;
  const amount = parseFloat(document.getElementById('futureAmt').value);
  const category = document.getElementById('futureCat').value;
  const cycle = document.getElementById('futureCycle').value;
  const date = document.getElementById('futureDate').value;
  if(!desc || !amount || !date) { toast('Please fill all fields'); return; }
  try { await api('/api/future_expenses', { method:'POST', body: JSON.stringify({ description:desc, amount, category, cycle, date }) }); toast('Pinned'); document.getElementById('futureDesc').value=''; document.getElementById('futureAmt').value=''; loadFutureExpenses(); } catch(e) { toast(e.message); }
});
async function deleteFuture(id) { if(!confirm('Remove this future expense?')) return; try { await api(`/api/future_expenses/${id}`, { method:'DELETE' }); toast('Removed'); loadFutureExpenses(); } catch(e) { toast(e.message); } }
document.getElementById('applyFutureBtn').addEventListener('click', async ()=>{
  try { const result = await api('/api/apply_future_expenses', { method:'POST' }); toast(`Processed ${result.count} pending expenses`); loadFutureExpenses(); } catch(e) { toast(e.message); }
});

// ── PROFILE ──
document.getElementById('profileAvatar').addEventListener('input', function() {
  const url = this.value.trim();
  const previewImg = document.getElementById('profilePreviewImg');
  const placeholder = document.getElementById('profilePreviewPlaceholder');
  if (url) {
    previewImg.src = url; previewImg.onload = () => { previewImg.style.display = 'block'; placeholder.style.display = 'none'; };
    previewImg.onerror = () => { previewImg.style.display = 'none'; placeholder.style.display = 'block'; };
  } else { previewImg.style.display = 'none'; placeholder.style.display = 'block'; }
});
document.getElementById('profileFileInput').addEventListener('change', function(e) {
  const file = e.target.files[0]; if(!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    document.getElementById('profileAvatar').value = ev.target.result;
    document.getElementById('profileAvatar').dispatchEvent(new Event('input'));
  };
  reader.readAsDataURL(file);
});
document.getElementById('saveProfileBtn').addEventListener('click', async () => {
  const name = document.getElementById('profileName').value.trim();
  const email = document.getElementById('profileEmail').value.trim();
  const password = document.getElementById('profilePass').value;
  const avatar_url = document.getElementById('profileAvatar').value.trim();
  const body = { name, email, avatar_url };
  if (password) body.password = password;
  try { const updatedUser = await api('/api/profile', { method:'PUT', body: JSON.stringify(body) }); currentUser = updatedUser; toast('Profile updated!'); } catch(e) { toast(e.message); }
});

// ── ADMIN ──
async function loadAdmin() {
  if(!currentUser || currentUser.role!=='admin') return;
  try {
    const stats = await api('/api/admin/stats');
    document.getElementById('adminStats').innerHTML = `
      <div class="stat-card"><div class="stat-value">${stats.total_users}</div><div class="stat-label">Users</div></div>
      <div class="stat-card"><div class="stat-value">${stats.total_transactions}</div><div class="stat-label">Transactions</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(stats.total_income)}</div><div class="stat-label">Total Income</div></div>
      <div class="stat-card"><div class="stat-value">${fmt(stats.total_expense)}</div><div class="stat-label">Total Expense</div></div>
      <div class="stat-card blue-glow"><div class="stat-value">${stats.avg_health_score}</div><div class="stat-label">Avg Health Score</div></div>`;
    loadAdminUsers(); loadAdminTransactions();
  } catch(e) { toast(e.message); }
}
document.getElementById('adminSearchUser').addEventListener('input', loadAdminUsers);
document.getElementById('adminUserFilter').addEventListener('change', loadAdminTransactions);
document.querySelectorAll('#adminTabs .tab').forEach(tab=>{
  tab.addEventListener('click', ()=>{
    document.querySelectorAll('#adminTabs .tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    if(tab.dataset.tab==='users'){
      document.getElementById('adminUsersPanel').style.display='block';
      document.getElementById('adminTransactionsPanel').style.display='none';
    } else {
      document.getElementById('adminUsersPanel').style.display='none';
      document.getElementById('adminTransactionsPanel').style.display='block';
    }
  });
});
async function loadAdminUsers() {
  try {
    const users = await api('/api/admin/users');
    const search = document.getElementById('adminSearchUser').value.toLowerCase();
    const filtered = users.filter(u => u.name.toLowerCase().includes(search) || u.email.toLowerCase().includes(search));
    document.getElementById('adminUserTable').innerHTML = filtered.map(u => `
      <tr id="userRow-${u.id}">
        <td>${u.id}</td><td>${esc(u.name)}</td><td>${esc(u.email)}</td><td>${esc(u.role)}</td>
        <td>${fmtDate(u.created_at)}</td>
      </tr>`).join('');
  } catch(e) { toast(e.message); }
}
async function loadAdminTransactions() {
  try {
    const userId = document.getElementById('adminUserFilter').value;
    const url = userId ? `/api/admin/transactions?user_id=${userId}` : '/api/admin/transactions';
    const txs = await api(url);
    document.getElementById('adminTxTable').innerHTML = txs.map(t=>`
      <tr>
        <td>${fmtDate(t.tx_date)}</td><td>User ${t.user_id}</td><td>${esc(t.category)}</td>
        <td>${t.tx_type}</td><td>${fmt(t.amount)}</td><td>${t.is_need?'Need':'Want'}</td>
      </tr>`).join('');
    const users = await api('/api/admin/users');
    document.getElementById('adminUserFilter').innerHTML = '<option value="">All Users</option>'+
      users.map(u=>`<option value="${u.id}">${esc(u.name)} (${u.id})</option>`).join('');
  } catch(e) { toast(e.message); }
}

// ── EXPORT ──
document.getElementById('exportBtn').addEventListener('click', ()=>{ window.open('/api/export/csv', '_blank'); });

// ── CHATBOT ──
let chatDragging = false, chatOffsetX, chatOffsetY;
document.getElementById('chatHeader').addEventListener('mousedown', (e) => {
  if(e.target.id === 'chatToggleBtn') return;
  chatDragging = true; chatOffsetX = e.clientX - chatbot.getBoundingClientRect().left;
  chatOffsetY = e.clientY - chatbot.getBoundingClientRect().top;
  chatbot.style.cursor = 'grabbing'; e.preventDefault();
});
document.addEventListener('mousemove', (e) => {
  if(!chatDragging) return;
  chatbot.style.left = Math.max(0, e.clientX - chatOffsetX) + 'px';
  chatbot.style.top = Math.max(0, e.clientY - chatOffsetY) + 'px';
  chatbot.style.right = 'auto'; chatbot.style.bottom = 'auto';
});
document.addEventListener('mouseup', () => { if(chatDragging){ chatDragging = false; chatbot.style.cursor = ''; } });
document.getElementById('chatToggleBtn').addEventListener('click', () => {
  chatbot.classList.toggle('minimized');
  document.getElementById('chatToggleBtn').textContent = chatbot.classList.contains('minimized') ? '□' : '–';
});
document.getElementById('chatSend').addEventListener('click', sendChat);
document.getElementById('chatInp').addEventListener('keypress', e=>{ if(e.key==='Enter') sendChat(); });
async function sendChat() {
  const inp = document.getElementById('chatInp');
  const msg = inp.value.trim();
  if(!msg) return;
  const msgs = document.getElementById('chatMsgs');
  msgs.innerHTML += `<div class="msg user">${esc(msg)}</div>`;
  inp.value = '';
  try { const resp = await api('/api/ai/chat', { method:'POST', body: JSON.stringify({ message: msg }) }); msgs.innerHTML += `<div class="msg bot">${esc(resp.reply)}</div>`; } catch(e) { msgs.innerHTML += `<div class="msg bot">Sorry, I'm having trouble right now.</div>`; }
  msgs.scrollTop = msgs.scrollHeight;
}

// ── INIT ──
(async ()=>{
  try {
    const user = await api('/api/me');
    currentUser = user;
    authOverlay.style.display = 'none';
    initApp();
  } catch(e) { authOverlay.style.display = 'flex'; }
})();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    response = app.make_response(HTML_PAGE)      # Flask creates a response object
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ----------------------------------------------------------------------
# Init DB
# ----------------------------------------------------------------------
with app.app_context():
    db.create_all()
    ensure_schema()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
