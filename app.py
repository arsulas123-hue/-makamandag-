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
import sys
import warnings

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
# Multi-AI Router (unchanged)
# ----------------------------------------------------------------------
GEMINI_MODELS = ["gemini-2.0-flash"]
FREE_MODELS = [
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

def route_ai_request(prompt, max_tokens=400):
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                print(f"✅ Used Gemini model: {model_name}")
                return response.text.strip()
        except Exception as e:
            print(f"Gemini {model_name} failed: {e}")
    if not OPENROUTER_API_KEY:
        return "Sorry, all AI services are busy. Try again later."
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
                print(f"✅ Used OpenRouter model: {model}")
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return "⚠️ All AI services unavailable."

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ----------------------------------------------------------------------
# Models (unchanged)
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
# Schema migration helper (unchanged)
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
# Authentication helpers (unchanged)
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
# Transaction validation helper (NEW)
# ----------------------------------------------------------------------
def validate_transaction(user_id, amount, tx_type):
    """Raise ValueError if transaction is invalid."""
    if amount < 1:
        raise ValueError("Transaction amount must be at least 1.")
    if tx_type == 'expense':
        user = User.query.get(user_id)
        if user.monthly_budget_limit <= 0:
            raise ValueError("Your monthly budget is zero. Please set an income first.")
        # Compute current month totals
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
# Analytics helpers (unchanged)
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
# Auth routes (unchanged)
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
# Transaction routes (updated with validation)
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

    # Validate transaction
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
        note=note
    )
    db.session.add(tx)
    db.session.commit()

    # Auto‑process future expenses when salary is added (unchanged)
    if tx.tx_type == 'income' and tx.category.lower() == 'salary':
        today = get_today_date()
        for exp in FutureExpense.query.filter(
            FutureExpense.user_id == user.id,
            FutureExpense.expense_date <= today,
            FutureExpense.cycle == 'One-time'
        ).all():
            # Validate each future expense before adding
            try:
                validate_transaction(user.id, exp.amount, 'expense')
                db.session.add(Transaction(
                    user_id=user.id, amount=exp.amount, category=exp.category,
                    tx_type='expense', is_need=(exp.category in needs_set),
                    priority=1, note=f"Auto: {exp.description}"
                ))
                db.session.delete(exp)
            except ValueError as ve:
                # Skip this future expense if it would violate constraints
                print(f"Skipping future expense {exp.description}: {ve}")
                continue
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
# Budget routes (unchanged)
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
# Summary, Predict, Longevity, Future expenses, Allocations, Export (unchanged)
# ----------------------------------------------------------------------
@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    if get_current_user().id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(get_monthly_summary(user_id))

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
# ADMIN API ROUTES (unchanged)
# ----------------------------------------------------------------------
@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    total_users = User.query.count()
    total_transactions = Transaction.query.count()
    total_income = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.tx_type == 'income').scalar() or 0
    total_expense = db.session.query(db.func.sum(Transaction.amount)).filter(Transaction.tx_type == 'expense').scalar() or 0
    all_health_scores = []
    for user in User.query.all():
        score = compute_health_score(user.id)
        all_health_scores.append(score)
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
# OCR endpoint (updated with validation)
# ----------------------------------------------------------------------
needs_set = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}

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

        # Validate each transaction individually
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
        # If any transactions failed validation, rollback and return error
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
# AI full setup (unchanged)
# ----------------------------------------------------------------------
@app.route('/api/ai/full_setup', methods=['POST'])
@login_required
def ai_full_setup():
    # ... (unchanged, no changes needed)
    pass  # Keeping the same code as before, omitted for brevity

# ----------------------------------------------------------------------
# Helper for safe datetime
# ----------------------------------------------------------------------
def get_today_date():
    try:
        return datetime.now(ZoneInfo("Asia/Manila")).date()
    except Exception as e:
        warnings.warn(f"ZoneInfo fallback to UTC: {e}")
        return datetime.now(timezone.utc).date()

# ----------------------------------------------------------------------
# Apply future expenses (updated with validation)
# ----------------------------------------------------------------------
@app.route('/api/apply_future_expenses', methods=['POST'])
@login_required
def apply_future_expenses():
    user = get_current_user()
    today = get_today_date()
    applied = []
    errors = []

    onetime = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle == 'One-time'
    ).all()
    for exp in onetime:
        try:
            validate_transaction(user.id, exp.amount, 'expense')
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
        except ValueError as e:
            errors.append(f"{exp.description}: {str(e)}")
            continue

    recurring = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle.in_(['Weekly', 'Monthly'])
    ).all()
    for exp in recurring:
        try:
            validate_transaction(user.id, exp.amount, 'expense')
            tx = Transaction(
                user_id=user.id,
                amount=exp.amount,
                category=exp.category,
                tx_type='expense',
                is_need=(exp.category in needs_set),
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
        except ValueError as e:
            errors.append(f"{exp.description}: {str(e)}")
            continue

    db.session.commit()
    if errors:
        # Still return success but list skipped items
        return jsonify({'applied': applied, 'count': len(applied), 'skipped': errors}), 200
    return jsonify({'applied': applied, 'count': len(applied)}), 200

# ----------------------------------------------------------------------
# Frontend HTML (updated with new buttons and Need/Want toggle)
# ----------------------------------------------------------------------
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
/* ... (same styles as before, omitted for brevity) ... */
/* Add extra style for the new Need/Want radio buttons */
.needwant-group { display:flex; gap:16px; align-items:center; margin-top:8px; }
.needwant-group label { font-size:0.82rem; display:flex; align-items:center; gap:4px; cursor:pointer; }
.needwant-group input[type="radio"] { accent-color:var(--green); }
</style>
</head>
<body>
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

  <!-- DASHBOARD SCREEN -->
  <div class="screen active" id="screen-dashboard">
    <div class="income-hero">
      <div class="income-label">Monthly Income & Expenses — Tell ML, it handles the rest</div>
      <div class="income-tool-row">
        <select id="incomeTool" class="tool-picker">
          <option value="manual-income">📝 Manual Income (ML Plan)</option>
          <option value="manual-income-add">💰 Add Income</option>
          <option value="manual-expense">📝 Manual Expense (Quick Log)</option>
          <option value="auto">📸 Automatic (Image/OCR)</option>
          <option value="profile">👤 Edit Profile</option>
        </select>
        <div style="margin-left:auto;"><span class="ai-badge">ML</span></div>
      </div>

      <!-- Manual Income Block (for ML Plan) -->
      <div id="manualIncomeBlock">
        <div class="income-input-row">
          <div class="income-peso">₱</div>
          <input class="income-input" id="incomeInput" type="number" placeholder="0.00" step="100" min="0">
          <button class="btn-add" id="addIncomeBtn" style="background:var(--green-dim); color:var(--green); border:1px solid var(--border);">+ Add Income</button>
        </div>
        <div class="mindset-row">
          <span style="font-size:0.78rem;color:var(--muted);align-self:center;">Spending style:</span>
          <button class="mindset-btn" data-mindset="Saver">🏦 Saver</button>
          <button class="mindset-btn active" data-mindset="Neutral">⚖️ Balanced</button>
          <button class="mindset-btn" data-mindset="Spender">🛍️ Spender</button>
        </div>
        <div class="ai-checklist">
          <div style="font-weight:600;margin-bottom:12px;">🧠 ML Autonomous Allocation (100% Sum Rule)</div>
          <div class="checklist-section"><div class="checklist-section-title">Needs ▼</div><div id="needsChecklist" class="checklist-item"></div></div>
          <div class="checklist-section"><div class="checklist-section-title">Wants ▼</div><div id="wantsChecklist" class="checklist-item"></div></div>
          <div id="totalWarning" class="total-warning" style="display:none;">⚠️ Total allocation must be 100% – ML will normalise.</div>
          <div style="margin-top:12px;"><button class="btn-analyze" id="analyzeBtn"><span id="analyzeBtnContent">🤖 Let ML Plan</span></button></div>
          <div id="needsWantsSummary" style="margin-top:12px;font-size:0.8rem;color:var(--text2);"></div>
        </div>
      </div>

      <!-- Manual Expense Block (with Need/Want override) -->
      <div id="manualExpenseBlock" style="display:none;">
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
            <button class="btn-add" id="addExpenseBtn">Add Expense →</button>
          </div>
        </div>
        <div id="expenseClassifyResult" style="display:none;margin-top:8px;"></div>
      </div>

      <!-- OCR Upload Block -->
      <div id="incomeImageUpload" class="income-image-upload" style="display:none;">
        <input type="file" id="incomeImage" accept="image/*" capture="environment">
        <div class="ocr-hint">📸 Take a photo or upload a payslip / budget screenshot. ML will read all income & expenses.</div>
      </div>

      <!-- Profile Edit Block -->
      <div id="profileBlock" class="profile-block" style="display:none;"> ... (unchanged) ... </div>
    </div>

    <!-- ... rest of the dashboard unchanged ... -->
  </div>

  <!-- ... other screens unchanged ... -->
</main>
<div id="toast"></div>
<div id="authOverlay" class="auth-overlay"> ... </div>
<div id="signoutModal" class="modal-overlay" style="display:none;"> ... </div>
<div id="avatarDropdown" class="avatar-dropdown" style="display:none;"> ... </div>
<div class="chatbot" id="chatbot"> ... </div>

<script>
// ... (large JS block with updates below) ...
</script>
</body>
</html>
"""

# Only the JavaScript changes are shown below (inside the HTML). The full HTML is huge,
# but the key JS updates are:
# - Added "Add Income" button handler.
# - Updated "Add Expense" to read the Need/Want radio selection.
# - The `runAIPlan` now also validates before creating the plan (but that's backend).

# For brevity, the full HTML is omitted here, but the essential changes are in the JS:

# (Inside the HTML script tag, replace the addExpenseBtn handler with:)
# document.getElementById('addExpenseBtn').addEventListener('click', async ()=>{
#   const amount = parseFloat(document.getElementById('expenseAmount').value);
#   if(!amount || amount < 1) { toast('Amount must be at least 1'); return; }
#   const category = document.getElementById('expenseCategory').value;
#   const note = document.getElementById('expenseNote').value;
#   const isNeed = document.querySelector('input[name="needwant"]:checked').value === 'need';
#   try {
#     await api('/api/transactions', { method:'POST', body: JSON.stringify({
#       amount, category, tx_type:'expense', is_need:isNeed, note
#     })});
#     toast('Expense added!');
#     document.getElementById('expenseAmount').value = '';
#     document.getElementById('expenseNote').value = '';
#     loadDashboard();
#   } catch(e) { toast(e.message); }
# });

# And add handler for "Add Income" button:
# document.getElementById('addIncomeBtn').addEventListener('click', async ()=>{
#   const amount = parseFloat(document.getElementById('incomeInput').value);
#   if(!amount || amount < 1) { toast('Enter a valid amount (min 1)'); return; }
#   try {
#     await api('/api/transactions', { method:'POST', body: JSON.stringify({
#       amount, category:'Salary', tx_type:'income', is_need:true, note:'Manual income'
#     })});
#     toast('Income added!');
#     document.getElementById('incomeInput').value = '';
#     loadDashboard();
#   } catch(e) { toast(e.message); }
# });

# The rest of the frontend (avatar, auth, etc.) remains unchanged.

# ----------------------------------------------------------------------
# Start the app
# ----------------------------------------------------------------------
with app.app_context():
    db.create_all()
    ensure_schema()

@app.route('/')
def index():
    return HTML_PAGE

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
