import json
import csv
import io
import os
import requests
import base64
from zoneinfo import ZoneInfo
import google.generativeai as genai
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from functools import wraps
import re
import numpy as np

from flask import Flask, request, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import inspect, text
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise RuntimeError("SECRET_KEY environment variable is not set!")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise RuntimeError("DATABASE_URL environment variable is not set!")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")
genai.configure(api_key=GEMINI_API_KEY)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

from flask_cors import CORS
CORS(app, supports_credentials=True)

# ----------------------------------------------------------------------
# Multi-AI Router
# ----------------------------------------------------------------------
GEMINI_MODELS = ["gemini-2.0-flash"]
FREE_MODELS = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

def route_ai_request(prompt, max_tokens=400):
    if OPENROUTER_API_KEY:
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
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        print(f"✅ Used OpenRouter model: {model}")
                        return data["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                print(f"✅ Used Gemini model: {model_name}")
                return response.text.strip()
        except Exception as e:
            print(f"Gemini {model_name} failed: {e}")
    return "⚠️ All AI services unavailable."

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
    avatar_url = db.Column(db.Text, nullable=True, default=None)

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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'avatar_url': self.avatar_url
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
# Schema migration helper
# ----------------------------------------------------------------------
def ensure_schema():
    inspector = inspect(db.engine)
    dialect = db.engine.dialect.name

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

        if dialect == 'postgresql':
            avatar_col_info = next((col for col in inspector.get_columns('users') if col['name'] == 'avatar_url'), None)
            if avatar_col_info and 'varchar' in str(avatar_col_info['type']).lower():
                with db.engine.connect() as conn:
                    conn.execute(text('ALTER TABLE users ALTER COLUMN avatar_url TYPE TEXT'))
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
    elif admin.role != 'admin':
        admin.role = 'admin'
        db.session.commit()

# ----------------------------------------------------------------------
# Authentication helpers
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
# Transaction validation helper
# ----------------------------------------------------------------------
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

def compute_savings_rate(user_id):
    summary = get_monthly_summary(user_id)
    total_income = summary['income']
    total_expense = summary['expense']
    if total_income == 0:
        return 0
    return round((total_income - total_expense) / total_income * 100, 1)

def compute_net_change(user_id):
    summary = get_monthly_summary(user_id)
    return round(summary['income'] - summary['expense'], 2)

def compute_budget_used(user_id):
    now = datetime.now(timezone.utc)
    first_of_month = datetime(now.year, now.month, 1)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= first_of_month
    ).all()
    total_expense = sum(e.amount for e in expenses)
    budgets = Budget.query.filter_by(user_id=user_id).all()
    total_budget = sum(b.limit_amount for b in budgets)
    if total_budget == 0:
        return 0
    used = (total_expense / total_budget) * 100
    return round(min(used, 100), 1)

def get_score_components(user_id):
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
    need_want_ratio = 1 - (want_expense / total_exp)
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    overspend_penalty = 0.0
    for cat, limit in budgets.items():
        if limit == 0:
            continue
        spent = sum(e.amount for e in expenses if e.category == cat)
        if spent > limit:
            overspend_penalty += (spent - limit) / limit
    budget_adherence = max(0, 1 - min(overspend_penalty, 1))
    monthly_income = defaultdict(float)
    for t in income:
        key = t.tx_date.strftime('%Y-%m')
        monthly_income[key] += t.amount
    values = list(monthly_income.values())
    if len(values) > 1:
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        income_stability = max(0, 1 - cv)
    else:
        income_stability = 0.5
    return {
        'savings_rate': round(savings_rate * 100, 1),
        'need_want_ratio': round(need_want_ratio * 100, 1),
        'budget_adherence': round(budget_adherence * 100, 1),
        'income_stability': round(income_stability * 100, 1),
    }

def get_score_history(user_id):
    now = datetime.now(timezone.utc)
    return [{'month': (now - timedelta(days=30*i)).strftime('%b %Y'), 'score': compute_health_score(user_id)} for i in range(6)]

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
needs_set = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}

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

    try:
        validate_transaction(user.id, amount, tx_type)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    tx_date_str = data.get('tx_date')
    if tx_date_str:
        tx_date = datetime.fromisoformat(tx_date_str)
    else:
        tx_date = datetime.now(timezone.utc)

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

    if tx.tx_type == 'income' and tx.category.lower() == 'salary':
        today = tx_date.date()
        for exp in FutureExpense.query.filter(
            FutureExpense.user_id == user.id,
            FutureExpense.expense_date <= today,
            FutureExpense.cycle == 'One-time'
        ).all():
            try:
                validate_transaction(user.id, exp.amount, 'expense')
                db.session.add(Transaction(
                    user_id=user.id, amount=exp.amount, category=exp.category,
                    tx_type='expense', is_need=(exp.category in needs_set),
                    priority=1, note=f"Auto: {exp.description}",
                    tx_date=tx_date
                ))
                db.session.delete(exp)
            except ValueError as ve:
                print(f"Skipping future expense {exp.description}: {ve}")
                continue
        db.session.commit()

    return jsonify(tx.to_dict()), 201

@app.route('/api/transactions/quick', methods=['POST'])
@login_required
def quick_transaction():
    user = get_current_user()
    data = request.json
    amount = data.get('amount')
    tx_type = data.get('tx_type')
    category = data.get('category')
    note = data.get('note', '')
    is_need = data.get('is_need', True)

    if not category and note:
        prompt = f"Categorize '{note}' as one of: Food & Dining, Transport, Groceries, Entertainment, Health, Debt repayment, Mortgage, Subscription, Hobbies, Salary, Savings, Other. Return only the category name."
        category = route_ai_request(prompt, max_tokens=20).strip()
        if not category or category not in needs_set | {'Salary', 'Savings', 'Subscription', 'Hobbies', 'Entertainment', 'Other'}:
            category = 'Other'
        is_need = category in needs_set

    if not category:
        category = 'Other'

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
        priority=2 if is_need else 1,
        note=note
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
# Summary / Predict / Longevity
# ----------------------------------------------------------------------
@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    base = get_monthly_summary(user_id)
    base['savings_rate'] = compute_savings_rate(user_id)
    base['net_change'] = compute_net_change(user_id)
    base['budget_used'] = compute_budget_used(user_id)
    return jsonify(base)

@app.route('/api/health_score/<int:user_id>')
@login_required
def health_score_detail(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    overall = compute_health_score(user_id)
    components = get_score_components(user_id)
    history = get_score_history(user_id)
    return jsonify({
        'overall': overall,
        'components': components,
        'history': history
    })

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    has_data = Transaction.query.filter_by(user_id=user_id, tx_type='expense').count() > 0
    if not has_data:
        return jsonify({'has_data': False, 'score': None, 'predictions': {'weekly': {}, 'categories': {}}, 'advice': []})
    return jsonify({
        'has_data': True,
        'score': compute_health_score(user_id),
        'predictions': {'weekly': generate_weekly_forecast(user_id), 'categories': get_category_totals(user_id)},
        'advice': []
    })

@app.route('/api/longevity/<int:user_id>')
@login_required
def longevity(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(compute_longevity(user_id))

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

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def admin_update_user(user_id):
    data = request.json
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    if 'role' in data:
        user.role = data['role']
    if 'name' in data:
        user.name = data['name']
    db.session.commit()
    return jsonify(user.to_dict())

@app.route('/api/admin/users/<int:user_id>/full', methods=['PUT'])
@admin_required
def admin_update_user_full(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    data = request.json
    if 'name' in data:
        user.name = data['name']
    if 'email' in data:
        if User.query.filter(User.email == data['email'], User.id != user.id).first():
            return jsonify({'error': 'Email already in use'}), 400
        user.email = data['email']
    if 'password' in data and data['password'].strip():
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        user.set_password(data['password'])
    if 'monthly_budget_limit' in data:
        user.monthly_budget_limit = data['monthly_budget_limit']
    if 'role' in data:
        user.role = data['role']
    if 'avatar_url' in data:
        url = data['avatar_url'].strip()
        if url and not (url.startswith('http://') or url.startswith('https://') or url.startswith('data:image/')):
            return jsonify({'error': 'Invalid avatar URL'}), 400
        user.avatar_url = url if url else None
    db.session.commit()
    return jsonify(user.to_dict()), 200

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    if user.id == session['user_id']:
        return jsonify({'error': 'You cannot delete your own admin account'}), 403
    Transaction.query.filter_by(user_id=user.id).delete()
    Budget.query.filter_by(user_id=user.id).delete()
    FutureExpense.query.filter_by(user_id=user.id).delete()
    UserAllocation.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted'})

@app.route('/api/admin/transactions', methods=['GET'])
@admin_required
def admin_transactions():
    user_id = request.args.get('user_id', type=int)
    query = Transaction.query
    if user_id:
        query = query.filter_by(user_id=user_id)
    transactions = query.order_by(Transaction.tx_date.desc()).all()
    return jsonify([t.to_dict() for t in transactions])

@app.route('/api/admin/allocations/<int:user_id>', methods=['GET'])
@admin_required
def admin_allocations(user_id):
    allocs = UserAllocation.query.filter_by(user_id=user_id).all()
    return jsonify([{'category_name': a.category_name, 'type': a.type, 'percentage': a.percentage} for a in allocs])

@app.route('/api/admin/budgets/<int:user_id>', methods=['GET'])
@admin_required
def admin_budgets(user_id):
    budgets = Budget.query.filter_by(user_id=user_id).all()
    return jsonify([{'category': b.category, 'limit': b.limit_amount} for b in budgets])

@app.route('/api/admin/future_expenses/<int:user_id>', methods=['GET'])
@admin_required
def admin_future_expenses(user_id):
    exps = FutureExpense.query.filter_by(user_id=user_id).all()
    return jsonify([e.to_dict() for e in exps])

# ----------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------
@app.route('/api/anomalies/<int:user_id>')
@login_required
def detect_anomalies(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    transactions = Transaction.query.filter_by(user_id=user_id, tx_type='expense').order_by(Transaction.tx_date.desc()).all()
    if len(transactions) < 5:
        return jsonify({'anomalies': []})
    category_amounts = defaultdict(list)
    for t in transactions:
        category_amounts[t.category].append(t.amount)
    anomalies = []
    for cat, amounts in category_amounts.items():
        if len(amounts) < 3:
            continue
        mean = np.mean(amounts)
        std = np.std(amounts)
        if std == 0:
            continue
        for t in transactions:
            if t.category == cat:
                z = (t.amount - mean) / std
                if abs(z) > 2.5:
                    anomalies.append({
                        'id': t.id,
                        'category': t.category,
                        'amount': t.amount,
                        'note': t.note,
                        'date': t.tx_date.isoformat(),
                        'deviation': round(z, 2)
                    })
    return jsonify({'anomalies': anomalies})

# ----------------------------------------------------------------------
# OCR endpoint (with preview without saving)
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

    save_to_db = request.form.get('save', 'true').lower() != 'false'

    image_bytes = file.read()
    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """You are a budget OCR AI. Extract all income and expense items from the image.
Return ONLY a JSON array of objects. Each object must have:
- type: "income" or "expense"
- amount: number (no currency symbols)
- category: one of: Food & Dining, Transport, Groceries, Health, Entertainment, Debt repayment, Mortgage, Subscription, Hobbies, Salary, Savings, Other
- note: short description (e.g., "Paycheck #1")

Do NOT wrap in markdown. Do NOT add any text before or after the JSON array.
Example:
[{"type":"income","amount":1150,"category":"Salary","note":"Paycheck #1"}]"""

    raw = None
    if OPENROUTER_API_KEY:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "google/gemini-2.0-flash-001",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                            ]
                        }
                    ],
                    "max_tokens": 1000
                },
                timeout=30
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                print("✅ Vision extraction via OpenRouter")
        except Exception as e:
            print("OpenRouter vision failed:", e)

    if raw is None:
        vision_models = ["gemini-2.0-flash"]
        for model_name in vision_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [{'mime_type': 'image/png', 'data': b64_image}, prompt]
                )
                if response and response.text:
                    raw = response.text.strip()
                    print(f"✅ Vision extraction with {model_name}")
                    break
            except Exception as e:
                print(f"Vision model {model_name} failed: {e}")

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
        return jsonify({'error': 'Failed to parse AI output', 'raw_output': raw[:200]}), 500

    if not save_to_db:
        preview = []
        for item in items:
            if not isinstance(item, dict):
                continue
            preview.append({
                'type': item.get('type', 'expense'),
                'amount': abs(float(item.get('amount', 0))),
                'category': item.get('category', 'Other'),
                'note': item.get('note', '')
            })
        return jsonify({'transactions': preview, 'count': len(preview)})

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

@app.route('/api/ocr_multi', methods=['POST'])
@login_required
def ocr_multi():
    user = get_current_user()
    if 'images' not in request.files:
        return jsonify({'error': 'No image files'}), 400
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    all_items = []
    for file in files:
        if file.filename == '':
            continue
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
            except Exception:
                pass
        if raw:
            try:
                items = json.loads(raw.strip())
                if isinstance(items, list):
                    all_items.extend(items)
            except:
                pass
    unique_items = []
    seen = set()
    for item in all_items:
        key = (item.get('type'), item.get('amount'), item.get('category'), item.get('note'))
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return jsonify({'transactions': unique_items, 'count': len(unique_items)})

# ----------------------------------------------------------------------
# AI full setup
# ----------------------------------------------------------------------
@app.route('/api/ai/full_setup', methods=['POST'])
@login_required
def ai_full_setup():
    user = get_current_user()
    data = request.json or {}
    monthly_income = data.get('monthly_income', user.monthly_budget_limit or 0)
    mindset = data.get('mindset', user.spending_mindset)
    social_status = data.get('social_status', user.social_status)
    selected_categories = data.get('selected_categories', [])

    if not selected_categories:
        selected_categories = ['Food & Dining', 'Transport', 'Groceries', 'Health',
                               'Entertainment', 'Debt repayment', 'Savings']

    user.monthly_budget_limit = monthly_income
    user.spending_mindset = mindset
    user.social_status = social_status
    db.session.commit()

    recent_spending = defaultdict(float)
    for t in Transaction.query.filter_by(user_id=user.id, tx_type='expense').order_by(Transaction.tx_date.desc()).limit(30).all():
        recent_spending[t.category] += t.amount

    categories_str = ', '.join(selected_categories)
    prompt = f"""
You are an expert financial planner AI for a Filipino user.

User profile:
- Monthly income: ₱{monthly_income:,.2f}
- Spending mindset: {mindset}
- Social status: {social_status}
- Recent category spending (last 30 days): {dict(recent_spending)}

Selected categories: {categories_str}

Rules:
- Saver mindset: give higher weight to needs and savings.
- Spender mindset: allow more wants.
- All categories sum to exactly 100%.
- Return ONLY a valid JSON object with category names as keys and numeric percentages as values, summing to 100.
- No extra text, no markdown.

Example output: {{"Food & Dining": 45.0, "Transport": 15.0, "Savings": 40.0}}
"""
    raw = route_ai_request(prompt, max_tokens=600)

    try:
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        allocation = json.loads(raw.strip())
    except Exception as e:
        print(f"AI full_setup parse error: {e}")
        needs_count = sum(1 for c in selected_categories if c in {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'})
        wants_count = sum(1 for c in selected_categories if c in {'Entertainment','Hobbies','Subscription'})
        savings_count = sum(1 for c in selected_categories if c == 'Savings')

        if needs_count == 0 and wants_count == 0 and savings_count == 0:
            allocation = {cat: 100.0 / len(selected_categories) for cat in selected_categories}
        else:
            if mindset.lower() == 'saver':
                need_weight, want_weight, saving_weight = 1.5, 0.5, 1.2
            elif mindset.lower() == 'spender':
                need_weight, want_weight, saving_weight = 1.0, 1.5, 0.7
            else:
                need_weight, want_weight, saving_weight = 1.0, 1.0, 1.0

            total_weight = needs_count * need_weight + wants_count * want_weight + savings_count * saving_weight
            allocation = {}
            for cat in selected_categories:
                if cat in {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}:
                    allocation[cat] = round((need_weight / total_weight) * 100, 1)
                elif cat in {'Entertainment','Hobbies','Subscription'}:
                    allocation[cat] = round((want_weight / total_weight) * 100, 1)
                elif cat == 'Savings':
                    allocation[cat] = round((saving_weight / total_weight) * 100, 1)
                else:
                    allocation[cat] = round((1.0 / total_weight) * 100, 1)

            diff = 100.0 - sum(allocation.values())
            if abs(diff) > 0.1:
                max_cat = max(allocation, key=allocation.get)
                allocation[max_cat] = round(allocation[max_cat] + diff, 1)

    allocation = {k: v for k, v in allocation.items() if k in selected_categories}
    total = sum(allocation.values())
    if total > 0 and abs(total - 100) > 0.1:
        factor = 100 / total
        allocation = {k: round(v * factor, 1) for k, v in allocation.items()}

    savings_prompt = f"""
User monthly income: ₱{monthly_income:.2f}, monthly expenses: ₱{sum(recent_spending.values()):.2f}, current balance: ₱{get_monthly_summary(user.id)['balance']:.2f}.
Spending mindset: {mindset}.
Recommend how much they should save per day, per week, and per month.
Return JSON: {{"daily": float, "weekly": float, "monthly": float, "tip": "string"}}.
"""
    try:
        raw_savings = route_ai_request(savings_prompt, max_tokens=200)
        if raw_savings.startswith('```'):
            raw_savings = raw_savings.split('```')[1]
            if raw_savings.startswith('json'):
                raw_savings = raw_savings[4:]
        savings_plan = json.loads(raw_savings.strip())
    except Exception:
        monthly_save = monthly_income * 0.2
        savings_plan = {
            "daily": round(monthly_save / 30, 2),
            "weekly": round(monthly_save / 4, 2),
            "monthly": round(monthly_save, 2),
            "tip": "Automate your savings on payday."
        }

    advice_prompt = f"""
User has budget allocations: {allocation}. Recent spending: {dict(recent_spending)}.
Provide 3 short pieces of financial advice as JSON array:
[{{"title":"...","body":"...","type":"info|warning|success"}}]
"""
    try:
        raw_advice = route_ai_request(advice_prompt, max_tokens=300)
        if raw_advice.startswith('```'):
            raw_advice = raw_advice.split('```')[1]
            if raw_advice.startswith('json'):
                raw_advice = raw_advice[4:]
        advice = json.loads(raw_advice.strip())
        if not isinstance(advice, list):
            advice = []
    except Exception:
        advice = [
            {"title": "Stay Consistent", "body": "Track every expense to improve your score.", "type": "info"},
            {"title": "Savings First", "body": "Transfer savings immediately after receiving income.", "type": "success"},
            {"title": "Review Wants", "body": "Audit subscriptions and entertainment monthly.", "type": "warning"}
        ]

    summary_prompt = f"Based on monthly income ₱{monthly_income}, mindset {mindset}, and allocations {allocation}, give a one‑sentence overall financial health assessment."
    try:
        raw_summary = route_ai_request(summary_prompt, max_tokens=100)
        financial_summary = raw_summary.strip()
    except Exception:
        financial_summary = "Your AI plan is ready. Start by logging your expenses to get personalized insights."

    UserAllocation.query.filter_by(user_id=user.id).delete()
    needs = {'Food & Dining', 'Transport', 'Groceries', 'Health', 'Debt repayment', 'Mortgage'}
    savings_cats = {'Savings'}
    for cat, pct in allocation.items():
        t = 'need' if cat in needs else ('savings' if cat in savings_cats else 'want')
        db.session.add(UserAllocation(user_id=user.id, category_name=cat, type=t, percentage=pct))

    Budget.query.filter_by(user_id=user.id).delete()
    for cat, pct in allocation.items():
        db.session.add(Budget(user_id=user.id, category=cat, limit_amount=round(monthly_income * pct / 100, 2)))

    db.session.commit()

    result = {
        'allocation': allocation,
        'allocation_amounts': {cat: round(monthly_income * pct / 100, 2) for cat, pct in allocation.items()},
        'savings_plan': savings_plan,
        'advice': advice,
        'financial_summary': financial_summary,
        'monthly_income': monthly_income
    }
    return jsonify(result), 200

# ----------------------------------------------------------------------
# Classify transaction / AI chat
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
    future_cnt = FutureExpense.query.filter_by(user_id=user.id).count()
    allocs = UserAllocation.query.filter_by(user_id=user.id).all()
    alloc_str = ', '.join([f"{a.category_name}: {a.percentage}%" for a in allocs]) or 'Not set'
    recent = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).limit(8).all()
    recent_str = '\n'.join([f"{t.category}: ₱{t.amount:.2f} ({t.tx_type}) on {t.tx_date.strftime('%b %d')}" for t in recent]) or 'None'
    prompt = f"""
You are SmartSpend AI for {user.name}.
Balance: ₱{summary['balance']:,.2f}, Income: ₱{summary['income']:,.2f}, Expenses: ₱{summary['expense']:,.2f}, Health: {score}/100.
Allocation: {alloc_str}. Pinned future: {future_cnt}.
Recent: {recent_str}
User asks: "{data.get('message','')}"
Reply in 3-5 sentences, warm, actionable, use ₱.
"""
    reply = route_ai_request(prompt, max_tokens=400)
    return jsonify({'reply': reply})

# ----------------------------------------------------------------------
# Apply future expenses
# ----------------------------------------------------------------------
@app.route('/api/apply_future_expenses', methods=['POST'])
@login_required
def apply_future_expenses():
    user = get_current_user()
    today = datetime.now(ZoneInfo("Asia/Manila")).date()
    applied = []

    onetime = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle == 'One-time'
    ).all()
    for exp in onetime:
        tx = Transaction(
            user_id=user.id,
            amount=exp.amount,
            category=exp.category,
            tx_type='expense',
            is_need=(exp.category in ['Food & Dining', 'Debt repayment', 'Mortgage', 'Transport']),
            priority=1,
            note=f"Auto-deducted future expense: {exp.description}"
        )
        db.session.add(tx)
        applied.append(exp.description)
        db.session.delete(exp)

    recurring = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle.in_(['Weekly', 'Monthly'])
    ).all()
    for exp in recurring:
        tx = Transaction(
            user_id=user.id,
            amount=exp.amount,
            category=exp.category,
            tx_type='expense',
            is_need=(exp.category in ['Food & Dining', 'Debt repayment', 'Mortgage', 'Transport']),
            priority=1,
            note=f"Auto-deducted recurring: {exp.description} ({exp.cycle})"
        )
        db.session.add(tx)
        applied.append(f"{exp.description} (recurring)")

        if exp.cycle == 'Weekly':
            next_date = exp.expense_date + timedelta(weeks=1)
        else:
            next_date = exp.expense_date + timedelta(days=30)

        while next_date <= today:
            if exp.cycle == 'Weekly':
                next_date += timedelta(weeks=1)
            else:
                next_date += timedelta(days=30)

        exp.expense_date = next_date

    db.session.commit()
    return jsonify({'applied': applied, 'count': len(applied)}), 200

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
/* ========== CSS unchanged ========== */
*{margin:0;padding:0;box-sizing:border-box;}
:root{
  --void:#060A10;--bg:#0B1120;--bg2:#111928;--bg3:#17223A;--bg4:#1E2E4A;
  --green:#00E5A0;--green-dim:rgba(0,229,160,0.1);--green-glow:rgba(0,229,160,0.35);
  --red:#FF3B5C;--red-dim:rgba(255,59,92,0.12);
  --amber:#F5A623;--blue:#3B8BFF;--purple:#9B59F5;
  --muted:#4A6080;--muted2:#6B88A8;--text:#D8EAF8;--text2:#B0C8E0;
  --border:rgba(0,229,160,0.15);--border2:rgba(255,255,255,0.06);
  --r:14px;--r2:20px;
  --font-display:'Syne',sans-serif;
  --font-mono:'IBM Plex Mono',monospace;
}
body{font-family:var(--font-display);background:var(--void);color:var(--text);min-height:100vh;overflow-x:hidden;}
::selection{background:var(--green-dim);color:var(--green);}
body::before{
  content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events:none;z-index:0;opacity:0.4;
}
/* ... all other CSS rules unchanged ... */
</style>
</head>
<body>
<!-- ========== Sidebar, topbar, screens (unchanged from last provided version) ========== -->
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
<main class="main">
  <!-- Topbar unchanged -->
  <div class="topbar">
    <div class="page-title" id="pageTitle">Dashboard</div>
    <div class="topbar-right">
      <div class="health-pill"><div class="health-dot"></div><span id="topScore" style="font-family:var(--font-mono)">—</span>&nbsp;/ 100</div>
      <div class="avatar" id="userAvatar">
        <span class="avatar-initial">—</span>
        <img src="" style="display:none;">
      </div>
      <button class="btn-signout" id="signoutBtn">Sign Out</button>
    </div>
  </div>

  <!-- DASHBOARD SCREEN (updated) -->
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

      <!-- Manual Block (Income / Expense switch) -->
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

        <!-- Income fields (no Add Income button) -->
        <div id="incomeFields">
          <div class="income-input-row" style="margin-bottom:12px;">
            <label style="font-size:0.8rem; color:var(--muted2);">Date</label>
            <input type="date" id="incomeDate" class="form-input" style="width:160px;">
          </div>
          <div style="display:flex; gap:12px; align-items:stretch; flex-wrap:wrap;">
            <div class="income-peso">₱</div>
            <input class="income-input" id="incomeInput" type="number" placeholder="0.00" step="100" min="0">
            <!-- Add Income button removed – ML Plan now handles it -->
          </div>
          <div id="incomeDateWarning" style="display:none; color:var(--red); font-size:0.75rem; margin-top:8px;">
            Cannot add past income
          </div>
        </div>

        <!-- Expense fields (no Add Expense button) -->
        <div id="expenseFields" style="display:none;">
          <div class="tx-form">
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
              <!-- Add Expense button removed -->
            </div>
          </div>
        </div>

        <!-- Spending style (only visible when Income mode) -->
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
              Custom ⚙️
              <button class="add-cat-btn" data-type="want" title="Add custom category" style="background:var(--green-dim); border:1px solid var(--border); color:var(--green); border-radius:6px; width:26px; height:26px; display:flex; align-items:center; justify-content:center; cursor:pointer; font-size:1.2rem;">+</button>
            </div>
            <div id="needsChecklist" class="checklist-item"></div>
            <div id="wantsChecklist" class="checklist-item"></div>
          </div>
          <div id="totalWarning" class="total-warning" style="display:none;">⚠️ Total allocation must be 100% – ML will normalise.</div>
          <div style="margin-top:12px;"><button class="btn-analyze" id="analyzeBtn"><span id="analyzeBtnContent">🤖 Let ML Plan</span></button></div>
          <div id="needsWantsSummary" style="margin-top:12px;font-size:0.8rem;color:var(--text2);"></div>
        </div>
      </div>

      <!-- OCR Upload Block -->
      <div id="incomeImageUpload" class="income-image-upload" style="display:none;">
        <input type="file" id="incomeImage" accept="image/*" capture="environment">
        <div class="ocr-hint">📸 Take a photo or upload a payslip / budget screenshot. ML will read all income & expenses.</div>
        <!-- OCR confirmation modal (hidden) -->
        <div id="ocrModal" class="modal-overlay" style="display:none;">
          <div class="auth-card" style="width:600px; max-height:80vh; overflow-y:auto;">
            <div class="auth-logo">📄</div>
            <div style="font-weight:600; margin-bottom:16px;">Review extracted items</div>
            <div id="ocrItemsList"></div>
            <div style="display:flex; gap:12px; margin-top:16px;">
              <button class="btn-add" id="ocrSaveBtn">Save All</button>
              <button class="btn" id="ocrCancelBtn">Cancel</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Profile Edit Block (unchanged) -->
      <div id="profileBlock" class="profile-block" style="display:none;">
        <!-- ... same as before ... -->
      </div>
    </div>

    <!-- Stats grid, Activity Feed, Plan Details, Budgets, Advice, Charts, Forecast – ALL UNCHANGED -->
    <!-- ... (they are exactly as in the last full working version) ... -->

  </div>

  <!-- Future screen, Insights, History, Admin – unchanged -->

</main>

<div id="toast"></div>
<!-- Auth overlay, modals, avatar dropdown, chatbot – unchanged -->

<script>
// ── GLOBAL VARS ──
let currentUser = null;
let allTransactions = [];
let currentMindset = 'Neutral';
let aiPlan = null;
let trendChart = null, catChartInst = null;
let isLogin = true;
let historyVisible = true;

const CAT_ICONS = { /* ... same ... */ };
const categoryConfig = [ /* ... same ... */ ];

function fmt(n){ /* ... same ... */ }
function fmtDate(iso){ /* ... same ... */ }
function esc(s){ /* ... same ... */ }

function toast(msg, color=''){ /* ... same ... */ }

async function api(url, opts={}){ /* ... same ... */ }

// ── CUSTOM CATEGORIES ──
let customCategories = [];

function renderChecklist() { /* ... same ... */ }
function addCategoryCheckbox(cat) { /* ... same ... */ }
function handleAddCategory(e) { /* ... same ... */ }
function updateChecklistPercentages(allocation) { /* ... same ... */ }

// ── TX MODE SWITCH ──
function switchTxMode() { /* ... same ... */ }

// ── DATE & INCOME VALIDATION ──
function updateAddIncomeButtonState() {
  const dateInput = document.getElementById('incomeDate');
  const warning = document.getElementById('incomeDateWarning');
  const today = new Date().toISOString().slice(0,10);
  const isPast = dateInput.value < today;
  warning.style.display = (isPast && dateInput.value !== '') ? 'inline' : 'none';
}

document.getElementById('incomeDate').addEventListener('change', updateAddIncomeButtonState);
document.getElementById('incomeInput').addEventListener('input', function() {
  updateAddIncomeButtonState();
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

// ── AUTH, SIGN OUT, NAVIGATION, INIT, AVATAR (unchanged) ──
// ...

// ── TOOL PICKER TOGGLES ──
document.getElementById('incomeTool').addEventListener('change', function(){
  const val = this.value;
  document.getElementById('manualBlock').style.display = (val === 'manual') ? 'block' : 'none';
  document.getElementById('incomeImageUpload').style.display = (val === 'auto') ? 'block' : 'none';
  document.getElementById('profileBlock').style.display = (val === 'profile') ? 'block' : 'none';

  if (val === 'profile' && currentUser) {
    // ... prefill profile ...
  }
});

// ── ML PLAN BUTTON (now handles transaction creation + AI plan) ──
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const mode = document.querySelector('input[name="txMode"]:checked').value;

  if (mode === 'income') {
    // Create income transaction first
    const amount = parseFloat(document.getElementById('incomeInput').value);
    if (!amount || amount < 1) { toast('Enter a valid income amount (min 1)'); return; }
    const date = document.getElementById('incomeDate').value;
    if (!date) { toast('Please select a date'); return; }
    const today = new Date().toISOString().slice(0,10);
    if (date < today) { toast('Cannot add past income'); return; }

    try {
      await api('/api/transactions', { method:'POST', body: JSON.stringify({
        amount, category:'Salary', tx_type:'income', is_need:true, note:'Manual income',
        tx_date: date
      })});
      toast('Income added! Now running ML Plan…');
      await autoRunPlanAndUpdateDashboard(true);  // true = run plan
    } catch(e) { toast(e.message); }
    return;
  }

  if (mode === 'expense') {
    // Create expense transaction first
    const amount = parseFloat(document.getElementById('expenseAmount').value);
    if (!amount || amount < 1) { toast('Amount must be at least 1'); return; }
    const category = document.getElementById('expenseCategory').value;
    const note = document.getElementById('expenseNote').value;
    const isNeed = document.querySelector('input[name="needwant"]:checked').value === 'need';
    try {
      await api('/api/transactions', { method:'POST', body: JSON.stringify({
        amount, category, tx_type:'expense', is_need:isNeed, note
      })});
      toast('Expense added! Now running ML Plan…');
      await autoRunPlanAndUpdateDashboard(true);
    } catch(e) { toast(e.message); }
    return;
  }
});

// ── AUTO RUN PLAN AFTER TRANSACTION ──
async function autoRunPlanAndUpdateDashboard(runPlan = false) {
  if (!currentUser) return;
  try {
    const summary = await api(`/api/summary/${currentUser.id}`);
    const totalIncome = summary.income || 0;
    document.getElementById('incomeInput').value = totalIncome;
    if (totalIncome <= 0) { loadDashboard(); return; }

    if (runPlan) {
      const selectedCategories = [];
      document.querySelectorAll('.cat-checkbox:checked').forEach(chk => selectedCategories.push(chk.dataset.cat));
      if (selectedCategories.length === 0) {
        ['Food & Dining', 'Transport', 'Groceries', 'Health', 'Entertainment', 'Debt repayment', 'Savings']
          .forEach(c => selectedCategories.push(c));
      }
      const result = await api('/api/ai/full_setup', {
        method: 'POST',
        body: JSON.stringify({ monthly_income: totalIncome, mindset: currentMindset, selected_categories })
      });
      aiPlan = result;
      if (result.allocation) {
        addFeedEvent('✅', `AI categorised ${Object.keys(result.allocation).length} categories into needs & wants`);
        addFeedEvent('💰', `Savings target set: ${fmt(result.savings_plan.monthly)}/month`);
        addFeedEvent('🧠', `Financial summary: "${result.financial_summary.substring(0,60)}…"`);
        addFeedEvent('📋', `${result.advice.length} personalised insights ready`);
        renderAIPlan(result);
      }
    }
  } catch(e) { console.error('Auto plan error:', e); }
  loadDashboard();
}

// ── DASHBOARD LOAD (unchanged) ──
async function loadDashboard() {
  // ... same as before, but also call autoRunPlanAndUpdateDashboard(false) at end
  if (currentUser.monthly_budget_limit > 0 && !aiPlan) {
    autoRunPlanAndUpdateDashboard(true);
  }
}

// ── RENDER AI PLAN (now adds "Return to Checklist" button) ──
function renderAIPlan(plan) {
  // ... financial summary, savings, alloc grid, advice unchanged ...

  // ── REBUILD CHECKLIST WITH PRIORITISED NEEDS/WANTS ──
  const categories = Object.keys(plan.allocation);
  const needsSet = new Set(['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage']);
  const savingsSet = new Set(['Savings']);
  const needs = [], wants = [], savings = [];
  categories.forEach(cat => {
    if (savingsSet.has(cat)) savings.push(cat);
    else if (needsSet.has(cat)) needs.push(cat);
    else wants.push(cat);
  });

  const needsCont = document.getElementById('needsChecklist');
  const wantsCont = document.getElementById('wantsChecklist');
  needsCont.innerHTML = '<div class="checklist-section-title">NEEDS (High Priority)</div>';
  wantsCont.innerHTML = '<div class="checklist-section-title">WANTS (Lower Priority)</div>';

  let idx = 1;
  [...needs, ...savings].forEach(cat => {
    const pct = plan.allocation[cat] || 0;
    const item = document.createElement('div');
    item.className = 'checklist-item';
    item.style.display = 'flex'; item.style.alignItems = 'center'; item.style.justifyContent = 'space-between';
    item.innerHTML = `
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="color:var(--green); font-weight:700;">${idx++}</span>
        <label style="font-size:0.82rem; color:var(--text2);">${esc(cat)}</label>
      </div>
      <span style="font-family:var(--font-mono); font-size:0.85rem; color:var(--green);">${pct.toFixed(1)}%</span>
    `;
    needsCont.appendChild(item);
  });

  wants.forEach(cat => {
    const pct = plan.allocation[cat] || 0;
    const item = document.createElement('div');
    item.className = 'checklist-item';
    item.style.display = 'flex'; item.style.alignItems = 'center'; item.style.justifyContent = 'space-between';
    item.innerHTML = `
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="color:var(--amber); font-weight:700;">${idx++}</span>
        <label style="font-size:0.82rem; color:var(--text2);">${esc(cat)}</label>
      </div>
      <span style="font-family:var(--font-mono); font-size:0.85rem; color:var(--amber);">${pct.toFixed(1)}%</span>
    `;
    wantsCont.appendChild(item);
  });

  // "Return to Checklist" button
  const returnBtn = document.createElement('button');
  returnBtn.textContent = '🔄 Return to Checklist';
  returnBtn.className = 'btn btn-primary';
  returnBtn.style.marginTop = '12px';
  returnBtn.onclick = () => {
    renderChecklist();  // restores checkboxes
    // remove the button itself
    returnBtn.remove();
  };
  wantsCont.appendChild(returnBtn);

  // ... rest of plan details toggle unchanged ...
}

// ── OCR IMAGE UPLOAD (now with confirmation modal) ──
document.getElementById('incomeImage').addEventListener('change', async function(){
  const file = this.files[0];
  if(!file) return;

  const formData = new FormData();
  formData.append('image', file);
  formData.append('save', 'false');   // IMPORTANT: tell backend to only extract, not save

  try {
    const resp = await fetch('/api/ocr_income', { method:'POST', body: formData, credentials:'include' });
    const data = await resp.json();
    if (data.transactions && data.transactions.length > 0) {
      showOCRModal(data.transactions.map(t => ({ ...t, id: Math.random().toString(36) })));
    } else {
      toast('No transactions found in the image');
    }
  } catch(e) {
    toast('OCR failed: '+e.message);
  }
});

function showOCRModal(items) {
  const modal = document.getElementById('ocrModal');
  const list = document.getElementById('ocrItemsList');
  list.innerHTML = '';

  items.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'budget-item';
    row.innerHTML = `
      <span style="flex:1;">${esc(item.category)} – ₱${item.amount.toFixed(2)} – ${esc(item.note)}</span>
      <select class="form-select" style="width:100px;" onchange="updateOCRItemType(${idx}, this.value)">
        <option value="income" ${item.type === 'income' ? 'selected' : ''}>Income</option>
        <option value="expense" ${item.type === 'expense' ? 'selected' : ''}>Expense</option>
      </select>
      <button class="btn-del" onclick="removeOCRItem(${idx})">🗑</button>
    `;
    list.appendChild(row);
  });

  // Store items globally for the modal
  window.ocrItems = items;
  document.getElementById('ocrSaveBtn').onclick = async () => {
    // Save each item via /api/transactions
    for (const item of window.ocrItems) {
      if (item._deleted) continue;
      await api('/api/transactions', {
        method: 'POST',
        body: JSON.stringify({
          amount: Math.abs(item.amount),
          category: item.category,
          tx_type: item.type,
          is_need: item.category in {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'},
          note: item.note
        })
      });
    }
    toast('Transactions saved!');
    modal.style.display = 'none';
    loadDashboard();
  };
  document.getElementById('ocrCancelBtn').onclick = () => {
    modal.style.display = 'none';
  };
  modal.style.display = 'flex';
}

// Helper for OCR modal
function updateOCRItemType(idx, newType) {
  window.ocrItems[idx].type = newType;
}
function removeOCRItem(idx) {
  window.ocrItems[idx]._deleted = true;
  document.getElementById('ocrItemsList').children[idx].style.display = 'none';
}

// ── BUDGET MANAGEMENT ──
async function loadBudgets() {
    if (!currentUser) return;
    try {
        const budgets = await api(`/api/budgets/${currentUser.id}`);
        renderBudgetList(budgets);
    } catch(e) { toast(e.message); }
}

function renderBudgetList(budgets) {
    const container = document.getElementById('budgetList');
    if (!budgets.length) {
        container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-text">No budgets set yet. Run AI Plan first.</div></div>';
        return;
    }
    container.innerHTML = budgets.map(b => {
        const safeId = b.category.replace(/[^a-zA-Z0-9]/g, '_');
        return `
          <div class="budget-item" id="budget-${safeId}">
            <div class="budget-cat">${esc(b.category)}</div>
            <input class="budget-input" id="budget-input-${safeId}" type="number" step="0.01" value="${b.limit.toFixed(2)}">
            <button class="budget-save-btn" onclick="saveBudget('${b.category.replace(/'/g, "\\'")}', '${safeId}')">Save</button>
            <span class="budget-amount">${fmt(b.limit)}</span>
          </div>`;
    }).join('');
    document.getElementById('budgetBlock').style.display = 'block';
}

async function saveBudget(category, safeId) {
    const input = document.getElementById('budget-input-' + safeId);
    const newLimit = parseFloat(input.value);
    if (isNaN(newLimit)) return;
    try {
        await api(`/api/budgets/${currentUser.id}/${encodeURIComponent(category)}`, {
            method: 'PUT',
            body: JSON.stringify({ limit: newLimit })
        });
        toast('Budget updated');
        loadBudgets();
    } catch(e) { toast(e.message); }
}

document.getElementById('resetBudgetsBtn')?.addEventListener('click', async () => {
    if (!currentUser) return;
    try {
        await api(`/api/budgets/reset_to_ai/${currentUser.id}`, { method: 'POST' });
        toast('Budgets reset to AI recommendations');
        loadBudgets();
    } catch(e) { toast(e.message); }
});

function renderForecast(weekly) {
  const container = document.getElementById('forecastBars');
  if(!weekly) { container.innerHTML = ''; return; }
  const maxVal = Math.max(...Object.values(weekly), 1);
  container.innerHTML = Object.entries(weekly).map(([week, val])=>{
    const pct = (val / maxVal * 100).toFixed(0);
    return `<div class="forecast-bar-row">
              <span class="forecast-week-label">${week}</span>
              <div class="forecast-track"><div class="forecast-fill" style="width:${pct}%"></div></div>
              <span class="forecast-val">${fmt(val)}</span>
            </div>`;
  }).join('');
}

function renderTrendChart(monthly) {
  const ctx = document.getElementById('trendChart');
  if(!ctx) return;
  if(trendChart) trendChart.destroy();
  const labels = Object.keys(monthly);
  const incomeData = labels.map(m=>monthly[m].income);
  const expenseData = labels.map(m=>monthly[m].expense);
  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Income', data: incomeData, borderColor: '#00E5A0', backgroundColor: 'rgba(0,229,160,0.1)', tension:0.3 },
        { label: 'Expense', data: expenseData, borderColor: '#FF3B5C', backgroundColor: 'rgba(255,59,92,0.1)', tension:0.3 }
      ]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#B0C8E0'}}} }
  });
}

// ── OCR IMAGE UPLOAD (with confirmation modal) ──
document.getElementById('incomeImage').addEventListener('change', async function(){
  const file = this.files[0];
  if(!file) return;
  const formData = new FormData();
  formData.append('image', file);
  formData.append('save', 'false');   // ask backend to only extract, not save
  try {
    const resp = await fetch('/api/ocr_income', { method:'POST', body: formData, credentials:'include' });
    const data = await resp.json();
    if (data.transactions && data.transactions.length > 0) {
      showOCRModal(data.transactions.map(t => ({ ...t, id: Math.random().toString(36) })));
    } else {
      toast('No transactions found in the image');
    }
  } catch(e) {
    toast('OCR failed: '+e.message);
  }
});

function showOCRModal(items) {
  const modal = document.getElementById('ocrModal');
  const list = document.getElementById('ocrItemsList');
  list.innerHTML = '';

  items.forEach((item, idx) => {
    const row = document.createElement('div');
    row.className = 'budget-item';
    row.innerHTML = `
      <span style="flex:1;">${esc(item.category)} – ₱${item.amount.toFixed(2)} – ${esc(item.note)}</span>
      <select class="form-select" style="width:100px;" onchange="updateOCRItemType(${idx}, this.value)">
        <option value="income" ${item.type === 'income' ? 'selected' : ''}>Income</option>
        <option value="expense" ${item.type === 'expense' ? 'selected' : ''}>Expense</option>
      </select>
      <button class="btn-del" onclick="removeOCRItem(${idx})">🗑</button>
    `;
    list.appendChild(row);
  });

  window.ocrItems = items;
  document.getElementById('ocrSaveBtn').onclick = async () => {
    for (const item of window.ocrItems) {
      if (item._deleted) continue;
      await api('/api/transactions', {
        method: 'POST',
        body: JSON.stringify({
          amount: Math.abs(item.amount),
          category: item.category,
          tx_type: item.type,
          is_need: item.category in {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'},
          note: item.note
        })
      });
    }
    toast('Transactions saved!');
    modal.style.display = 'none';
    loadDashboard();
  };
  document.getElementById('ocrCancelBtn').onclick = () => {
    modal.style.display = 'none';
  };
  modal.style.display = 'flex';
}

function updateOCRItemType(idx, newType) {
  window.ocrItems[idx].type = newType;
}
function removeOCRItem(idx) {
  window.ocrItems[idx]._deleted = true;
  document.getElementById('ocrItemsList').children[idx].style.display = 'none';
}

// ── FUTURE EXPENSES ──
async function loadFutureExpenses() {
  if(!currentUser) return;
  try {
    const exps = await api('/api/future_expenses');
    renderFutureList(exps);
  } catch(e) { toast(e.message); }
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
  try {
    await api('/api/future_expenses', { method:'POST', body: JSON.stringify({ description:desc, amount, category, cycle, date }) });
    toast('Pinned');
    document.getElementById('futureDesc').value=''; document.getElementById('futureAmt').value='';
    loadFutureExpenses();
  } catch(e) { toast(e.message); }
});
async function deleteFuture(id) {
  if(!confirm('Remove this future expense?')) return;
  try {
    await api(`/api/future_expenses/${id}`, { method:'DELETE' });
    toast('Removed'); loadFutureExpenses();
  } catch(e) { toast(e.message); }
}
document.getElementById('applyFutureBtn').addEventListener('click', async ()=>{
  try {
    const result = await api('/api/apply_future_expenses', { method:'POST' });
    toast(`Processed ${result.count} pending expenses`);
    loadFutureExpenses();
  } catch(e) { toast(e.message); }
});

// ── INSIGHTS ──
async function loadInsights() {
  if(!currentUser) return;
  try {
    const longevity = await api(`/api/longevity/${currentUser.id}`);
    const bal = longevity.balance;
    const avgDaily = longevity.avg_daily_spend;

    const scenarios = [
      { label: '🏦 Saver',   daily: avgDaily * 0.8,   cssClass: 'saver-fill' },
      { label: '⚖️ Balanced', daily: avgDaily,         cssClass: 'neutral-fill' },
      { label: '🛍️ Spender', daily: avgDaily * 1.2,   cssClass: 'spender-fill' }
    ];

    const maxDays = Math.max(...scenarios.map(s => bal / s.daily), 1);
    let barsHTML = '';
    scenarios.forEach(s => {
      const days = Math.floor(bal / s.daily);
      const pct = Math.min((days / maxDays) * 100, 100);
      barsHTML += `
        <div class="scenario-bar-row">
          <span class="scenario-label">${s.label}</span>
          <div class="scenario-track"><div class="scenario-fill ${s.cssClass}" style="width:${pct}%"></div></div>
          <span class="scenario-val">${days} days</span>
        </div>`;
    });
    document.getElementById('scenarioBars').innerHTML = barsHTML;

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
    return `<div class="forecast-bar-row">
              <span class="forecast-week-label">${week}</span>
              <div class="forecast-track"><div class="forecast-fill" style="width:${pct}%"></div></div>
              <span class="forecast-val">${fmt(val)}</span>
            </div>`;
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
    data: {
      labels,
      datasets: [{ data, backgroundColor: ['#00E5A0','#3B8BFF','#F5A623','#FF3B5C','#9B59F5','#00b87a','#FF8C00'] }]
    },
    options: { responsive:true, maintainAspectRatio:false, plugins:{legend:{labels:{color:'#B0C8E0'}}} }
  });
}

// ── HISTORY ──
async function loadHistory() {
  if(!currentUser) return;
  try {
    const txs = await api('/api/transactions');
    allTransactions = txs;
    renderHistory(txs);
  } catch(e) { toast(e.message); }
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
      <div class="tx-info">
        <div class="tx-cat">${esc(t.category)} ${badgeHtml}</div>
        <div class="tx-meta">${t.note?esc(t.note)+' · ':''}${fmtDate(t.tx_date)}</div>
      </div>
      <div class="tx-amount ${t.tx_type}">${t.tx_type==='income'?'+':'-'}${fmt(t.amount)}</div>
      <button class="btn-del" onclick="deleteTransaction(${t.id})">🗑</button>
    </div>`;
  }).join('');
}

document.getElementById('historySearch').addEventListener('input', ()=> renderHistory(allTransactions));

document.getElementById('showNeedWantBadges').addEventListener('change', ()=> {
  renderHistory(allTransactions);
});

document.getElementById('toggleHistoryViewBtn').addEventListener('click', () => {
  const content = document.getElementById('historyContent');
  const btn = document.getElementById('toggleHistoryViewBtn');
  historyVisible = !historyVisible;
  if (historyVisible) {
    content.style.display = 'block';
    btn.textContent = '🙈 Hide History';
  } else {
    content.style.display = 'none';
    btn.textContent = '👁 Show History';
  }
});

async function deleteTransaction(id) {
  if(!confirm('Delete this transaction?')) return;
  try {
    await api(`/api/transactions/${id}`, { method:'DELETE' });
    toast('Deleted');
    if(document.getElementById('screen-history').classList.contains('active')) loadHistory();
  } catch(e) { toast(e.message); }
}

// ── PROFILE PREVIEW & UPLOAD HANDLING ──
document.getElementById('profileAvatar').addEventListener('input', function() {
  const url = this.value.trim();
  const previewImg = document.getElementById('profilePreviewImg');
  const placeholder = document.getElementById('profilePreviewPlaceholder');
  if (url) {
    previewImg.src = url;
    previewImg.onerror = () => {
      previewImg.style.display = 'none';
      placeholder.style.display = 'block';
    };
    previewImg.onload = () => {
      previewImg.style.display = 'block';
      placeholder.style.display = 'none';
    };
    previewImg.src = url;
  } else {
    previewImg.style.display = 'none';
    placeholder.style.display = 'block';
  }
});

document.getElementById('profileFileInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    const dataUrl = ev.target.result;
    document.getElementById('profileAvatar').value = dataUrl;
    document.getElementById('profileAvatar').dispatchEvent(new Event('input'));
  };
  reader.readAsDataURL(file);
});

document.querySelectorAll('#profileBlock .preset-avatar').forEach(el => {
  el.addEventListener('click', () => {
    const dataUrl = el.dataset.url;
    document.getElementById('profileAvatar').value = dataUrl;
    document.getElementById('profileAvatar').dispatchEvent(new Event('input'));
  });
});

document.getElementById('saveProfileBtn').addEventListener('click', async () => {
  const name = document.getElementById('profileName').value.trim();
  const email = document.getElementById('profileEmail').value.trim();
  const password = document.getElementById('profilePass').value;
  const avatar_url = document.getElementById('profileAvatar').value.trim();

  const body = { name, email, avatar_url };
  if (password) body.password = password;

  try {
    const updatedUser = await api('/api/profile', { method: 'PUT', body: JSON.stringify(body) });
    currentUser = updatedUser;
    const initial = document.querySelector('.avatar-initial');
    const img = document.querySelector('#userAvatar img');
    if (updatedUser.avatar_url) {
      img.src = updatedUser.avatar_url;
      img.style.display = 'block';
      initial.style.display = 'none';
    } else {
      img.style.display = 'none';
      initial.style.display = 'block';
      initial.textContent = updatedUser.name?.charAt(0)?.toUpperCase() || '?';
    }
    toast('Profile updated!');
  } catch(e) { toast(e.message); }
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
    loadAdminUsers();
    loadAdminTransactions();
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
    const tbody = document.getElementById('adminUserTable');
    tbody.innerHTML = filtered.map(u => `
      <tr id="userRow-${u.id}">
        <td>${u.id}</td>
        <td>${esc(u.name)}</td>
        <td>${esc(u.email)}</td>
        <td>${esc(u.role)}</td>
        <td>${fmtDate(u.created_at)}</td>
        <td>
          <button onclick="adminDeleteUser(${u.id})" class="btn btn-danger" style="padding:4px 8px;font-size:0.7rem;">Del</button>
        </td>
      </tr>
    `).join('');
  } catch(e) { toast(e.message); }
}
async function adminDeleteUser(id){
  if(!confirm('Delete this user and all their data?')) return;
  try {
    await api(`/api/admin/users/${id}`, { method:'DELETE' });
    toast('User deleted'); loadAdminUsers();
  } catch(e) { toast(e.message); }
}
async function loadAdminTransactions(){
  try {
    const userId = document.getElementById('adminUserFilter').value;
    const url = userId ? `/api/admin/transactions?user_id=${userId}` : '/api/admin/transactions';
    const txs = await api(url);
    document.getElementById('adminTxTable').innerHTML = txs.map(t=>`
      <tr>
        <td>${fmtDate(t.tx_date)}</td><td>User ${t.user_id}</td><td>${esc(t.category)}</td><td>${t.tx_type}</td>
        <td>${fmt(t.amount)}</td><td>${t.is_need?'Need':'Want'}</td>
      </tr>`).join('');
    const users = await api('/api/admin/users');
    document.getElementById('adminUserFilter').innerHTML = '<option value="">All Users</option>'+
      users.map(u=>`<option value="${u.id}">${esc(u.name)} (${u.id})</option>`).join('');
  } catch(e) { toast(e.message); }
}

// ── EXPORT ──
document.getElementById('exportBtn').addEventListener('click', ()=>{
  window.open('/api/export/csv', '_blank');
});

// ── CHATBOT DRAGGABLE & MINIMIZABLE ──
(function(){
  const chatbot = document.getElementById('chatbot');
  const header = document.getElementById('chatHeader');
  let offsetX, offsetY, isDragging = false;
  header.addEventListener('mousedown', (e) => {
    if(e.target.id === 'chatToggleBtn') return;
    isDragging = true;
    offsetX = e.clientX - chatbot.getBoundingClientRect().left;
    offsetY = e.clientY - chatbot.getBoundingClientRect().top;
    chatbot.style.cursor = 'grabbing';
    e.preventDefault();
  });
  document.addEventListener('mousemove', (e) => {
    if(!isDragging) return;
    const left = e.clientX - offsetX;
    const top = e.clientY - offsetY;
    const w = window.innerWidth, h = window.innerHeight;
    const bw = chatbot.offsetWidth, bh = chatbot.offsetHeight;
    chatbot.style.left = Math.max(0, Math.min(left, w - bw)) + 'px';
    chatbot.style.top = Math.max(0, Math.min(top, h - bh)) + 'px';
    chatbot.style.right = 'auto'; chatbot.style.bottom = 'auto';
  });
  document.addEventListener('mouseup', () => {
    if(isDragging){ isDragging = false; chatbot.style.cursor = ''; }
  });
  document.getElementById('chatToggleBtn').addEventListener('click', () => {
    chatbot.classList.toggle('minimized');
    document.getElementById('chatToggleBtn').textContent = chatbot.classList.contains('minimized') ? '□' : '–';
  });
})();

document.getElementById('chatSend').addEventListener('click', sendChat);
document.getElementById('chatInp').addEventListener('keypress', e=>{ if(e.key==='Enter') sendChat(); });
async function sendChat() {
  const inp = document.getElementById('chatInp');
  const msg = inp.value.trim();
  if(!msg) return;
  const msgs = document.getElementById('chatMsgs');
  msgs.innerHTML += `<div class="msg user">${esc(msg)}</div>`;
  inp.value = '';
  try {
    const resp = await api('/api/ai/chat', { method:'POST', body: JSON.stringify({ message: msg }) });
    msgs.innerHTML += `<div class="msg bot">${esc(resp.reply)}</div>`;
  } catch(e) {
    msgs.innerHTML += `<div class="msg bot">Sorry, I'm having trouble right now.</div>`;
  }
  msgs.scrollTop = msgs.scrollHeight;
}

// ── INITIAL LOAD ──
(async ()=>{
  try {
    const user = await api('/api/me');
    currentUser = user;
    authOverlay.style.display = 'none';
    initApp();
  } catch(e) { 
    authOverlay.style.display = 'flex'; 
  }
})();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return HTML_PAGE

with app.app_context():
    db.create_all()
    ensure_schema()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
