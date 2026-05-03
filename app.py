import json
import csv
import io
import os
import requests
import base64
import google.generativeai as genai
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from functools import wraps
import re

from flask import Flask, request, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import inspect, text
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
app = Flask(__name__)

# Required environment variables (set these in Render dashboard)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise RuntimeError("SECRET_KEY environment variable is not set!")

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise RuntimeError("DATABASE_URL environment variable is not set!")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp'

# Database connection pooling for PostgreSQL
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True,
}

# GEMINI API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")
genai.configure(api_key=GEMINI_API_KEY)

# OpenRouter key (optional – can be empty)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# CORS for frontend-backend communication
from flask_cors import CORS
CORS(app, supports_credentials=True)

# ----------------------------------------------------------------------
# Multi-AI Router (OpenRouter fallback)
# ----------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]
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
    # Fallback to OpenRouter
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
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
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
# Schema migration helper
# ----------------------------------------------------------------------
def ensure_schema():
    inspector = inspect(db.engine)
    # Users table
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
        ]:
            if col not in existing_columns:
                with db.engine.connect() as conn:
                    conn.execute(text(f'ALTER TABLE users ADD COLUMN {col} {defn}'))
                    conn.commit()

    # Transactions table
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

    # Create default admin user if none exists
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
    overspend_penalty = sum(
        (sum(e.amount for e in expenses if e.category == cat) - limit) / limit
        for cat, limit in budgets.items()
        if sum(e.amount for e in expenses if e.category == cat) > limit
    )
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
    tx = Transaction(
        user_id=user.id,
        amount=data['amount'],
        category=data['category'],
        tx_type=data['tx_type'],
        is_need=data.get('is_need', True),
        priority=data.get('priority', 1),
        note=data.get('note', '')
    )
    db.session.add(tx)
    db.session.commit()
    if tx.tx_type == 'income' and tx.category.lower() == 'salary':
        today = datetime.now(timezone.utc).date()
        for exp in FutureExpense.query.filter(
            FutureExpense.user_id == user.id,
            FutureExpense.expense_date <= today,
            FutureExpense.cycle == 'One-time'
        ).all():
            db.session.add(Transaction(
                user_id=user.id, amount=exp.amount, category=exp.category,
                tx_type='expense', is_need=(exp.category in ['Food & Dining', 'Debt repayment', 'Mortgage', 'Transport']),
                priority=1, note=f"Auto: {exp.description}"
            ))
            db.session.delete(exp)
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
# ADMIN API ROUTES
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
# OCR endpoint (optional, for automatic income extraction)
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

Do NOT wrap in markdown. Do NOT add any text before or after the JSON array.
Example:
[{"type":"income","amount":1150,"category":"Salary","note":"Paycheck #1"}]"""

    raw = None

    # Try OpenRouter first (more reliable for vision)
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

    # Fallback to Gemini Vision (only if no raw yet)
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

    # --- Robust JSON extraction ---
    items = None
    try:
        # 1. Strip markdown code fences
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('```')[1]
            if cleaned.startswith('json'):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        # 2. Try direct parse
        items = json.loads(cleaned)
        if not isinstance(items, list):
            items = None
    except Exception:
        # 3. Try to find JSON array with regex
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
        except Exception:
            items = None

    if not items:
        return jsonify({'error': 'Failed to parse AI output', 'raw_output': raw[:200]}), 500

    # Create transactions
    created = []
    for item in items:
        if not isinstance(item, dict):
            continue
        amount = abs(float(item.get('amount', 0)))
        tx_type = item.get('type', 'expense')
        if tx_type not in ('income', 'expense'):
            tx_type = 'expense'
        category = item.get('category', 'Other')
        note = item.get('note', '')
        needs = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}
        is_need = category in needs
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
        created.append(tx.to_dict())
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
        # Clean up markdown if present
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        allocation = json.loads(raw.strip())
    except Exception as e:
        print(f"AI full_setup parse error: {e}")
        # Intelligent fallback
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

    # Savings plan
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

    # Advice
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

    # Financial summary
    summary_prompt = f"Based on monthly income ₱{monthly_income}, mindset {mindset}, and allocations {allocation}, give a one‑sentence overall financial health assessment."
    try:
        raw_summary = route_ai_request(summary_prompt, max_tokens=100)
        financial_summary = raw_summary.strip()
    except Exception:
        financial_summary = "Your AI plan is ready. Start by logging your expenses to get personalized insights."

    # Persist to database
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
        needs = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}
        return jsonify({'is_need': data['category'] in needs, 'priority': 2 if data['category'] in needs else 1, 'suggested_note': data.get('note','')})

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
    today = datetime.now(timezone.utc).date()
    future_expenses = FutureExpense.query.filter(
        FutureExpense.user_id == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle == 'One-time'
    ).all()
    applied = []
    for exp in future_expenses:
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
    db.session.commit()
    return jsonify({'applied': applied, 'count': len(applied)}), 200

# ----------------------------------------------------------------------
# Frontend (single HTML page)
# ----------------------------------------------------------------------
HTML_PAGE = r"""PASTE YOUR EXACT HTML_PAGE STRING HERE"""  # Keep your existing HTML here
@app.route('/')
def index():
    return HTML_PAGE

# ----------------------------------------------------------------------
# Initialize DB
# ----------------------------------------------------------------------
with app.app_context():
    db.create_all()
    ensure_schema()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
