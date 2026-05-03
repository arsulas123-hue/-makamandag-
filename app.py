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
import re  # for OCR stub

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

# 🔥 Fix: Add database connection pooling for PostgreSQL
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,           # Keep 10 connections open
    'max_overflow': 20,        # Allow up to 20 extra connections during spikes
    'pool_timeout': 30,        # Wait 30 sec for a connection before timing out
    'pool_recycle': 3600,      # Recycle connections after 1 hour (prevents stale connections)
    'pool_pre_ping': True,     # Check connection is alive before using it
}

# GEMINI API key (uses environment variable; no fallback to hardcoded)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set!")
genai.configure(api_key=GEMINI_API_KEY)

# OpenRouter key (optional – can be empty)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# CORS for frontend-backend communication (if needed)
from flask_cors import CORS
CORS(app, supports_credentials=True)



# ----------------------------------------------------------------------
# Multi-AI Router (OpenRouter fallback)
# ----------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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
    # Fallback to OpenRouter (unchanged)
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

    # Transactions table: add is_need and priority if missing
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
# Analytics helpers (unchanged)
# ----------------------------------------------------------------------
def compute_health_score(user_id):
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    expenses = Transaction.query.filter(Transaction.user_id == user_id, Transaction.tx_type == 'expense', Transaction.tx_date >= thirty_days_ago).all()
    income = Transaction.query.filter(Transaction.user_id == user_id, Transaction.tx_type == 'income', Transaction.tx_date >= thirty_days_ago).all()
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
    expenses = Transaction.query.filter(Transaction.user_id == user_id, Transaction.tx_type == 'expense', Transaction.tx_date >= thirty_days_ago).all()
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
    expenses = Transaction.query.filter(Transaction.user_id == user_id, Transaction.tx_type == 'expense', Transaction.tx_date >= first_of_month).all()
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
    expenses = Transaction.query.filter(Transaction.user_id == user_id, Transaction.tx_type == 'expense', Transaction.tx_date >= thirty_days_ago).all()
    avg_daily = sum(e.amount for e in expenses) / 30 if expenses else 0
    days = int(summary['balance'] / avg_daily) if avg_daily > 0 else 0
    return {'balance': summary['balance'], 'avg_daily_spend': avg_daily, 'days': days}


# ----------------------------------------------------------------------
# Auth routes (unchanged except role return)
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
# Transaction routes (unchanged)
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
        for exp in FutureExpense.query.filter(FutureExpense.user_id == user.id, FutureExpense.expense_date <= today, FutureExpense.cycle == 'One-time').all():
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
    })# ----------------------------------------------------------------------
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
# Apply future expenses (unchanged)
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


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SmartSpend — AI Finance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
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
.sidebar{
  position:fixed;left:0;top:0;bottom:0;width:72px;
  background:rgba(11,17,32,0.95);backdrop-filter:blur(20px);
  border-right:1px solid var(--border2);
  display:flex;flex-direction:column;align-items:center;
  padding:20px 0;gap:6px;z-index:200;
  transition:width 0.3s cubic-bezier(0.4,0,0.2,1);
}
.sidebar:hover{width:220px;}
.logo{
  width:44px;height:44px;border-radius:12px;margin-bottom:20px;
  background:linear-gradient(135deg,var(--green),#009e6a);
  display:flex;align-items:center;justify-content:center;
  font-size:1.4rem;cursor:pointer;flex-shrink:0;
  box-shadow:0 0 24px var(--green-glow);
}
.nav-item{
  width:calc(100% - 16px);display:flex;align-items:center;gap:14px;
  padding:12px 14px;border-radius:10px;cursor:pointer;
  border:none;background:transparent;color:var(--muted2);
  font-family:var(--font-display);font-size:0.875rem;font-weight:500;
  white-space:nowrap;overflow:hidden;transition:all 0.2s;text-align:left;
}
.nav-item:hover{background:var(--green-dim);color:var(--text);}
.nav-item.active{background:var(--green-dim);color:var(--green);box-shadow:inset 2px 0 0 var(--green);}
.nav-icon{font-size:1.1rem;flex-shrink:0;width:20px;text-align:center;}
.nav-label{opacity:0;transition:opacity 0.2s;font-size:0.85rem;}
.sidebar:hover .nav-label{opacity:1;}
.sidebar-sep{width:40px;height:1px;background:var(--border2);margin:8px 0;}
.sidebar:hover .sidebar-sep{width:calc(100% - 28px);}
.main{margin-left:72px;padding:28px 36px;min-height:100vh;position:relative;z-index:1;}
@media(max-width:768px){.main{margin-left:0;padding:16px;}.sidebar{display:none;}}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:32px;gap:16px;flex-wrap:wrap;}
.page-title{font-size:1.6rem;font-weight:800;letter-spacing:-0.5px;}
.topbar-right{display:flex;align-items:center;gap:12px;}
.health-pill{
  display:flex;align-items:center;gap:8px;
  padding:7px 16px;border-radius:99px;
  background:var(--green-dim);border:1px solid var(--border);
  font-family:var(--font-mono);font-size:0.8rem;color:var(--green);
}
.health-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 var(--green-glow);}50%{opacity:0.8;box-shadow:0 0 0 6px transparent;}}
.avatar{
  width:40px;height:40px;border-radius:10px;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:0.9rem;cursor:pointer;
}
.btn-signout{background:var(--red-dim);border:1px solid rgba(255,59,92,0.2);color:var(--red);padding:8px 16px;border-radius:10px;cursor:pointer;font-family:var(--font-display);font-weight:600;font-size:0.85rem;}
.screen{display:none;animation:fadeIn 0.3s ease;}
.screen.active{display:block;}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.card{background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r2);padding:24px;margin-bottom:20px;transition:border-color 0.2s;}
.card:hover{border-color:var(--border);}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px;}
.card-title{font-size:0.95rem;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;}
.ai-badge{background:linear-gradient(90deg,var(--purple),var(--blue));color:#fff;padding:3px 10px;border-radius:99px;font-size:0.68rem;font-weight:600;letter-spacing:0.5px;}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;}
.stat-card{
  background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r2);padding:22px;
  position:relative;overflow:hidden;transition:all 0.25s;cursor:default;
}
.stat-card::before{
  content:'';position:absolute;top:0;right:0;width:80px;height:80px;
  background:radial-gradient(circle,var(--green-dim),transparent 70%);
  border-radius:50%;transform:translate(30px,-30px);
}
.stat-card.neg::before{background:radial-gradient(circle,var(--red-dim),transparent 70%);}
.stat-card.blue-glow::before{background:radial-gradient(circle,rgba(59,139,255,0.1),transparent 70%);}
.stat-value{font-family:var(--font-mono);font-size:1.7rem;font-weight:600;margin-bottom:6px;letter-spacing:-1px;}
.stat-label{font-size:0.72rem;color:var(--muted2);text-transform:uppercase;letter-spacing:0.5px;}
.stat-sub{font-size:0.75rem;color:var(--muted);font-family:var(--font-mono);margin-top:4px;}
.income-hero{
  background:linear-gradient(135deg,var(--bg2) 0%,rgba(0,229,160,0.05) 100%);
  border:1px solid var(--border);border-radius:var(--r2);padding:32px;
  margin-bottom:24px;position:relative;overflow:hidden;
}
.income-hero::after{
  content:'';position:absolute;top:-60px;right:-60px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,var(--green-glow),transparent 70%);
}
.income-label{font-size:0.8rem;color:var(--muted2);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}
.income-tool-row{
  display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px;
}
.tool-picker{
  background:var(--bg3);border:1px solid var(--border2);border-radius:10px;
  padding:8px 12px;font-family:var(--font-display);font-size:0.9rem;
  cursor:pointer;color:var(--text);
}
.income-input-row{
  display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;
}
.income-peso{
  font-family:var(--font-mono);font-size:2rem;font-weight:500;
  color:var(--green);display:flex;align-items:center;padding:0 8px;
}
.income-input{
  flex:2;min-width:200px;
  font-family:var(--font-mono);font-size:1.8rem;font-weight:500;
  background:transparent;border:none;border-bottom:2px solid var(--border);
  color:var(--text);outline:none;padding:8px 4px;
  transition:border-color 0.2s;
}
.income-input:focus{border-color:var(--green);}
.income-image-upload{display:none;margin-top:12px;}
.income-image-upload input{background:var(--bg3);padding:8px;border-radius:8px;}
.ocr-hint{font-size:0.7rem;color:var(--muted2);margin-top:4px;}
.mindset-row{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;}
.mindset-btn{
  padding:8px 20px;border-radius:99px;border:1px solid var(--border2);
  background:transparent;color:var(--muted2);cursor:pointer;
  font-family:var(--font-display);font-size:0.85rem;font-weight:500;
  transition:all 0.2s;
}
.mindset-btn.active{background:var(--green-dim);border-color:var(--green);color:var(--green);}
.btn-analyze{
  background:linear-gradient(135deg,var(--green),#00b87a);color:#000;
  border:none;border-radius:12px;padding:14px 28px;cursor:pointer;
  font-family:var(--font-display);font-weight:700;font-size:0.95rem;
  display:flex;align-items:center;gap:10px;transition:all 0.2s;flex-shrink:0;
  box-shadow:0 4px 20px var(--green-glow);
}
.btn-analyze:hover{transform:translateY(-2px);box-shadow:0 8px 28px var(--green-glow);}
.btn-analyze:disabled{opacity:0.5;cursor:not-allowed;transform:none;}
.ai-checklist{
  background:var(--bg3);border-radius:12px;padding:20px;margin-top:20px;
}
.checklist-section{margin-bottom:16px;}
.checklist-section-title{font-size:0.85rem;font-weight:600;color:var(--green);margin-bottom:8px;}
.checklist-item{display:flex;align-items:center;gap:12px;margin-bottom:8px;flex-wrap:wrap;}
.checklist-item label{display:flex;align-items:center;gap:6px;cursor:pointer;}
.checklist-item .cat-percent{font-family:var(--font-mono);font-size:0.8rem;color:var(--muted2);min-width:45px;}
.total-warning{color:var(--red);font-size:0.75rem;margin-top:8px;}
.ai-feed{margin-bottom:24px;}
.ai-feed-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.ai-pulse{width:10px;height:10px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite;}
.ai-feed-title{font-family:var(--font-mono);font-size:0.8rem;color:var(--green);text-transform:uppercase;letter-spacing:1px;}
.ai-event{
  display:flex;align-items:flex-start;gap:12px;
  padding:10px 0;border-bottom:1px solid var(--border2);
  animation:slideIn 0.4s ease;
}
@keyframes slideIn{from{opacity:0;transform:translateX(-10px);}to{opacity:1;transform:translateX(0);}}
.ai-event:last-child{border-bottom:none;}
.ai-event-icon{font-size:1rem;flex-shrink:0;margin-top:1px;}
.ai-event-text{font-size:0.83rem;color:var(--text2);line-height:1.5;}
.ai-event-time{font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);margin-left:auto;flex-shrink:0;}
.alloc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;}
.alloc-item{background:var(--bg3);border-radius:12px;padding:16px;position:relative;overflow:hidden;}
.alloc-item-bar{position:absolute;bottom:0;left:0;height:3px;background:var(--green);transition:width 1s ease;}
.alloc-item-bar.want{background:var(--amber);}
.alloc-item-bar.savings{background:var(--blue);}
.alloc-cat{font-size:0.78rem;color:var(--muted2);margin-bottom:6px;text-transform:uppercase;letter-spacing:0.3px;}
.alloc-pct{font-family:var(--font-mono);font-size:1.4rem;font-weight:600;color:var(--text);}
.alloc-amount{font-family:var(--font-mono);font-size:0.8rem;color:var(--muted2);margin-top:4px;}
.alloc-type{font-size:0.65rem;padding:2px 8px;border-radius:99px;display:inline-block;margin-top:6px;font-weight:600;}
.alloc-type.need{background:var(--green-dim);color:var(--green);}
.alloc-type.want{background:rgba(245,166,35,0.12);color:var(--amber);}
.alloc-type.savings{background:rgba(59,139,255,0.12);color:var(--blue);}
.savings-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:16px;}
.savings-card{background:var(--bg3);border-radius:12px;padding:16px;text-align:center;}
.savings-period{font-size:0.72rem;color:var(--muted2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;}
.savings-amount{font-family:var(--font-mono);font-size:1.3rem;font-weight:600;color:var(--green);}
.advice-list{display:flex;flex-direction:column;gap:10px;}
.advice-card{background:var(--bg3);border-radius:12px;padding:16px;display:flex;gap:12px;align-items:flex-start;border-left:3px solid var(--muted);}
.advice-card.info{border-color:var(--blue);}
.advice-card.warning{border-color:var(--amber);}
.advice-card.success{border-color:var(--green);}
.advice-icon{font-size:1.1rem;flex-shrink:0;}
.advice-title{font-size:0.85rem;font-weight:600;margin-bottom:4px;}
.advice-body{font-size:0.8rem;color:var(--text2);line-height:1.5;}
.tx-form{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;align-items:end;}
.form-group{display:flex;flex-direction:column;gap:6px;}
.form-label{font-size:0.72rem;color:var(--muted2);text-transform:uppercase;letter-spacing:0.5px;}
.form-input,.form-select{
  background:var(--bg3);color:var(--text);
  border:1px solid var(--border2);border-radius:10px;
  padding:11px 14px;outline:none;
  font-family:var(--font-display);font-size:0.9rem;
  transition:border-color 0.2s;width:100%;
}
.form-input:focus,.form-select:focus{border-color:var(--green);}
.form-select option{background:var(--bg2);}
.btn-add{
  background:linear-gradient(135deg,var(--green),#00b87a);color:#000;
  border:none;border-radius:10px;padding:12px 20px;cursor:pointer;
  font-family:var(--font-display);font-weight:700;font-size:0.9rem;
  transition:all 0.2s;white-space:nowrap;
}
.btn-add:hover{transform:translateY(-1px);}
.btn-add:disabled{opacity:0.5;cursor:not-allowed;}
.ai-classify-result{
  display:inline-flex;align-items:center;gap:6px;
  background:var(--green-dim);border:1px solid var(--border);
  color:var(--green);border-radius:8px;padding:6px 12px;
  font-family:var(--font-mono);font-size:0.75rem;margin-top:8px;
  animation:fadeIn 0.3s ease;
}
.tx-list{display:flex;flex-direction:column;gap:8px;}
.tx-item{
  display:flex;align-items:center;gap:14px;
  background:var(--bg3);border-radius:12px;padding:14px 16px;
  transition:background 0.15s;
}
.tx-item:hover{background:var(--bg4);}
.tx-cat-icon{width:36px;height:36px;border-radius:10px;background:var(--bg4);display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;}
.tx-info{flex:1;}
.tx-cat{font-size:0.88rem;font-weight:500;}
.tx-meta{font-size:0.72rem;color:var(--muted);margin-top:2px;font-family:var(--font-mono);}
.tx-amount{font-family:var(--font-mono);font-weight:600;font-size:0.95rem;flex-shrink:0;}
.tx-amount.income{color:var(--green);}
.tx-amount.expense{color:var(--red);}
.tx-badge{font-size:0.65rem;padding:2px 7px;border-radius:99px;margin-left:6px;font-weight:600;}
.tx-badge.need{background:var(--green-dim);color:var(--green);}
.tx-badge.want{background:rgba(245,166,35,0.12);color:var(--amber);}
.btn-del{background:none;border:none;color:var(--muted);cursor:pointer;font-size:1rem;padding:4px;border-radius:6px;transition:all 0.15s;}
.btn-del:hover{color:var(--red);background:var(--red-dim);}
.future-item{display:flex;align-items:center;gap:14px;background:var(--bg3);border-radius:12px;padding:14px 16px;margin-bottom:8px;}
.future-info{flex:1;}
.future-desc{font-size:0.88rem;font-weight:500;}
.future-meta{font-size:0.72rem;color:var(--muted);margin-top:2px;font-family:var(--font-mono);}
.future-amount{font-family:var(--font-mono);font-weight:600;font-size:0.9rem;color:var(--amber);}
.forecast-bar-row{display:flex;align-items:center;gap:14px;margin-bottom:14px;}
.forecast-week-label{font-family:var(--font-mono);font-size:0.75rem;color:var(--green);width:60px;flex-shrink:0;}
.forecast-track{flex:1;height:6px;background:var(--bg3);border-radius:99px;overflow:hidden;}
.forecast-fill{height:100%;background:linear-gradient(90deg,var(--green),#00b87a);border-radius:99px;width:0;transition:width 1.2s cubic-bezier(0.4,0,0.2,1);}
.forecast-val{font-family:var(--font-mono);font-size:0.75rem;color:var(--text2);width:90px;text-align:right;flex-shrink:0;}
.longevity-display{display:flex;align-items:center;gap:24px;padding:20px 0;flex-wrap:wrap;}
.longevity-days{font-family:var(--font-mono);font-size:3rem;font-weight:600;color:var(--green);line-height:1;}
.longevity-label{color:var(--muted2);font-size:0.85rem;margin-top:6px;}
.longevity-sep{width:1px;height:60px;background:var(--border2);}
.longevity-stat{text-align:center;}
.longevity-stat-val{font-family:var(--font-mono);font-size:1.1rem;font-weight:600;}
.longevity-stat-label{font-size:0.72rem;color:var(--muted);margin-top:4px;}
#totalWarning, #needsWantsSummary { display: none; }
.btn{
  background:var(--bg3);color:var(--text);border:1px solid var(--border2);
  border-radius:10px;padding:10px 16px;cursor:pointer;
  font-family:var(--font-display);font-weight:600;font-size:0.85rem;
  transition:all 0.2s;
}
.btn:hover{border-color:var(--border);background:var(--bg4);}
.btn-primary{background:var(--green-dim);border-color:var(--border);color:var(--green);}
.btn-danger{background:var(--red-dim);border-color:rgba(255,59,92,0.2);color:var(--red);}
#toast{
  position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(80px);
  background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:12px 24px;font-size:0.875rem;
  opacity:0;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);
  z-index:9999;white-space:nowrap;
  box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
.empty-state{text-align:center;padding:40px;color:var(--muted);}
.empty-state-icon{font-size:2.5rem;margin-bottom:12px;}
.empty-state-text{font-size:0.9rem;line-height:1.6;}
.auth-overlay{
  position:fixed;inset:0;
  background:rgba(6,10,16,0.97);backdrop-filter:blur(20px);
  z-index:9000;display:flex;align-items:center;justify-content:center;
}
.auth-card{
  background:var(--bg2);border:1px solid var(--border);border-radius:24px;
  padding:40px;width:420px;max-width:90%;
  box-shadow:0 24px 80px rgba(0,0,0,0.5);
}
.auth-logo{font-size:2rem;margin-bottom:4px;}
.auth-title{font-size:1.6rem;font-weight:800;margin-bottom:4px;}
.auth-sub{font-size:0.85rem;color:var(--muted2);margin-bottom:28px;}
.auth-input{
  display:block;width:100%;
  background:var(--bg3);color:var(--text);
  border:1px solid var(--border2);border-radius:12px;
  padding:13px 16px;outline:none;font-family:var(--font-display);
  font-size:0.9rem;margin-bottom:12px;transition:border-color 0.2s;
}
.auth-input:focus{border-color:var(--green);}
.btn-auth{
  width:100%;background:linear-gradient(135deg,var(--green),#00b87a);
  color:#000;border:none;border-radius:12px;
  padding:14px;font-family:var(--font-display);font-weight:700;
  font-size:1rem;cursor:pointer;margin-top:8px;
  box-shadow:0 4px 20px var(--green-glow);transition:all 0.2s;
}
.btn-auth:hover{transform:translateY(-2px);}
.auth-toggle{text-align:center;margin-top:16px;font-size:0.85rem;color:var(--muted2);cursor:pointer;}
.auth-toggle span{color:var(--green);font-weight:600;}
.auth-error{color:var(--red);font-size:0.8rem;margin-top:8px;min-height:18px;}
.auth-checkbox{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:0.85rem;color:var(--muted2);cursor:pointer;}
.chatbot{
  position:fixed;bottom:20px;right:20px;
  width:360px;height:460px;
  min-width:280px;min-height:300px;max-width:85vw;max-height:70vh;
  resize:both;overflow:auto;
  z-index:8000;
}
.chat-window{
  width:100%;height:100%;
  background:var(--bg2);border:1px solid var(--border);border-radius:20px;
  display:flex;flex-direction:column;overflow:hidden;
  box-shadow:0 12px 48px rgba(0,0,0,0.4);
}
.chat-header{
  padding:14px 16px;background:var(--bg3);
  border-bottom:1px solid var(--border2);
  display:flex;align-items:center;gap:10px;cursor:move;user-select:none;
  flex-shrink:0;
}
.chat-header-title{font-size:0.875rem;font-weight:600;}
.chat-msgs{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px;}
.msg{max-width:84%;padding:10px 14px;border-radius:16px;font-size:0.82rem;line-height:1.5;}
.msg.user{align-self:flex-end;background:var(--green-dim);color:var(--green);border-bottom-right-radius:4px;}
.msg.bot{align-self:flex-start;background:var(--bg3);color:var(--text2);border-bottom-left-radius:4px;}
.chat-input-row{display:flex;padding:12px;gap:8px;background:var(--bg3);border-top:1px solid var(--border2);flex-shrink:0;}
.chat-inp{flex:1;background:var(--bg4);border:1px solid var(--border2);color:var(--text);border-radius:10px;padding:9px 12px;font-family:var(--font-display);font-size:0.82rem;outline:none;transition:border-color 0.2s;}
.chat-inp:focus{border-color:var(--green);}
.chat-send{background:var(--green-dim);border:1px solid var(--border);color:var(--green);border-radius:10px;padding:8px 14px;cursor:pointer;font-weight:600;font-size:0.82rem;transition:all 0.15s;}
.chat-send:hover{background:var(--green);color:#000;}
.chat-toggle-btn{background:none;border:none;color:var(--muted2);cursor:pointer;font-size:1.2rem;padding:0 4px;line-height:1;transition:color 0.2s;margin-left:auto;}
.chat-toggle-btn:hover{color:var(--green);}
.chatbot.minimized .chat-msgs,
.chatbot.minimized .chat-input-row{display:none;}
.chatbot.minimized .chat-window{height:auto !important;border-radius:20px;}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(0,229,160,0.3);border-top-color:var(--green);border-radius:50%;animation:spin 0.7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.financial-summary-text{font-size:0.875rem;color:var(--text2);line-height:1.6;font-style:italic;border-left:3px solid var(--green);padding-left:14px;margin-bottom:20px;}
.chart-wrapper{position:relative;height:220px;}
.tabs{display:flex;gap:4px;background:var(--bg3);border-radius:12px;padding:4px;margin-bottom:20px;}
.tab{flex:1;padding:8px;border:none;background:transparent;color:var(--muted2);border-radius:8px;cursor:pointer;font-family:var(--font-display);font-size:0.82rem;font-weight:500;transition:all 0.2s;}
.tab.active{background:var(--bg2);color:var(--text);box-shadow:0 2px 8px rgba(0,0,0,0.3);}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:99px;}

.scenario-bar-row {
  display:flex; align-items:center; gap:14px; margin-bottom:14px;
}
.scenario-label {
  font-family:var(--font-mono); font-size:0.75rem; color:var(--green); width:70px; flex-shrink:0;
}
.scenario-track {
  flex:1; height:6px; background:var(--bg3); border-radius:99px; overflow:hidden;
}
.scenario-fill {
  height:100%; border-radius:99px; width:0; transition:width 1.2s ease;
}
.saver-fill { background:linear-gradient(90deg, var(--blue), var(--green)); }
.neutral-fill { background:linear-gradient(90deg, var(--green), #00b87a); }
.spender-fill { background:linear-gradient(90deg, var(--amber), var(--red)); }
.scenario-val {
  font-family:var(--font-mono); font-size:0.75rem; color:var(--text2); width:90px; text-align:right; flex-shrink:0;
}

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
      <div class="avatar" id="userAvatar">—</div>
      <button class="btn-signout" id="signoutBtn">Sign Out</button>
    </div>
  </div>

  <div class="screen active" id="screen-dashboard">
    <div class="income-hero">
      <div class="income-label">Monthly Income & Expenses — Tell AI, it handles the rest</div>
      <div class="income-tool-row">
        <select id="incomeTool" class="tool-picker">
          <option value="manual-income">📝 Manual Income (AI Plan)</option>
          <option value="manual-expense">📝 Manual Expense (Quick Log)</option>
          <option value="auto">📸 Automatic (Image/OCR)</option>
        </select>
        <div style="margin-left:auto;"><span class="ai-badge">GEMINI</span></div>
      </div>

      <!-- Manual Income Mode -->
      <div id="manualIncomeBlock">
        <div class="income-input-row">
          <div class="income-peso">₱</div>
          <input class="income-input" id="incomeInput" type="number" placeholder="0.00" step="100" min="0">
        </div>
        <div class="mindset-row">
          <span style="font-size:0.78rem;color:var(--muted);align-self:center;">Spending style:</span>
          <button class="mindset-btn" data-mindset="Saver">🏦 Saver</button>
          <button class="mindset-btn active" data-mindset="Neutral">⚖️ Balanced</button>
          <button class="mindset-btn" data-mindset="Spender">🛍️ Spender</button>
        </div>
        <div class="ai-checklist">
          <div style="font-weight:600;margin-bottom:12px;">🧠 AI Autonomous Allocation (100% Sum Rule)</div>
          <div class="checklist-section"><div class="checklist-section-title">Needs ▼</div><div id="needsChecklist" class="checklist-item"></div></div>
          <div class="checklist-section"><div class="checklist-section-title">Wants ▼</div><div id="wantsChecklist" class="checklist-item"></div></div>
          <div id="totalWarning" class="total-warning" style="display:none;">⚠️ Total allocation must be 100% – AI will normalise.</div>
          <div style="margin-top:12px;"><button class="btn-analyze" id="analyzeBtn"><span id="analyzeBtnContent">🤖 Let AI Plan</span></button></div>
          <div id="needsWantsSummary" style="margin-top:12px;font-size:0.8rem;color:var(--text2);"></div>
        </div>
      </div>

      <!-- Manual Expense Mode (hidden by default) -->
      <div id="manualExpenseBlock" style="display:none;">
        <div class="tx-form">
          <div class="form-group"><label class="form-label">Amount (₱)</label><input class="form-input" id="expenseAmount" type="number" placeholder="0.00" step="0.01" min="0"></div>
          <div class="form-group"><label class="form-label">Category</label>
            <select class="form-select" id="expenseCategory">
              <option>Food & Dining</option><option>Transport</option><option>Groceries</option>
              <option>Entertainment</option><option>Health</option><option>Debt repayment</option>
              <option>Mortgage</option><option>Subscription</option><option>Hobbies</option><option>Other</option>
            </select>
          </div>
          <div class="form-group"><label class="form-label">Note</label><input class="form-input" id="expenseNote" placeholder="Short description"></div>
          <div class="form-group" style="justify-content:flex-end;"><button class="btn-add" id="addExpenseBtn">Add Expense →</button></div>
        </div>
        <div id="expenseClassifyResult" style="display:none;margin-top:8px;"></div>
      </div>

      <!-- OCR Upload Block -->
      <div id="incomeImageUpload" class="income-image-upload">
        <input type="file" id="incomeImage" accept="image/*" capture="environment">
        <div class="ocr-hint">📸 Take a photo or upload a payslip / budget screenshot. AI will read all income & expenses.</div>
      </div>
    </div>

    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value" id="sBalance">—</div><div class="stat-label">Balance</div></div>
      <div class="stat-card neg"><div class="stat-value" id="sExpense" style="color:var(--red)">—</div><div class="stat-label">Month Expenses</div></div>
      <div class="stat-card"><div class="stat-value" id="sIncome" style="color:var(--green)">—</div><div class="stat-label">Month Income</div></div>
      <div class="stat-card blue-glow"><div class="stat-value" id="sScore" style="color:var(--blue)">—</div><div class="stat-label">AI Health Score</div><div class="stat-sub" id="scoreLabel">awaiting data</div></div>
    </div>

    <div class="card ai-feed">
      <div class="ai-feed-header"><div class="ai-pulse"></div><div class="ai-feed-title">AI Activity Feed</div><div class="ai-badge" style="margin-left:auto;">GEMINI</div></div>
      <div id="aiFeed"><div class="empty-state"><div class="empty-state-icon">🤖</div><div class="empty-state-text">Enter your income above and click<br><strong style="color:var(--green)">Let AI Plan</strong> — Gemini will build your entire financial plan automatically.</div></div></div>
    </div>

    <div id="financialSummaryBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">AI Assessment</span><span class="ai-badge">Gemini</span></div>
      <div class="financial-summary-text" id="financialSummaryText"></div>
      <div class="card-title" style="margin-bottom:12px;">Savings Target</div>
      <div class="savings-cards">
        <div class="savings-card"><div class="savings-period">Daily</div><div class="savings-amount" id="saveDaily">—</div></div>
        <div class="savings-card"><div class="savings-period">Weekly</div><div class="savings-amount" id="saveWeekly">—</div></div>
        <div class="savings-card"><div class="savings-period">Monthly</div><div class="savings-amount" id="saveMonthly">—</div></div>
      </div>
      <div id="savingsTip" style="font-size:0.8rem;color:var(--muted2);margin-top:12px;font-style:italic;"></div>
    </div>

    <div id="allocationBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">AI Budget Allocation</span><span class="ai-badge">100% Autonomous</span></div>
      <div class="alloc-grid" id="allocGrid"></div>
    </div>

    <div id="adviceBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">AI Insights</span></div>
      <div class="advice-list" id="adviceList"></div>
    </div>

    <div class="card" id="chartBlock" style="display:none">
      <div class="card-header"><span class="card-title">Income vs Expense Trend</span></div>
      <div class="chart-wrapper"><canvas id="trendChart"></canvas></div>
    </div>

    <div class="card" id="forecastBlock" style="display:none">
      <div class="card-header"><span class="card-title">ML Spending Forecast (4 weeks)</span></div>
      <div id="forecastBars"></div>
    </div>
  </div>

  <!-- Other screens (future, insights, history, admin) kept the same -->
  <div class="screen" id="screen-future">
    <div class="card">
      <div class="card-header"><span class="card-title">Pin Future Expense</span></div>
      <div class="tx-form">
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

   <div class="screen" id="screen-insights">
    <div class="card">
      <div class="card-header"><span class="card-title">Budget Longevity (3 Scenarios)</span></div>
      <!-- Scenario bars -->
      <div id="scenarioBars" style="margin-bottom:20px;"></div>
      <!-- Summary stats -->
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


  <div class="screen" id="screen-history">
    <div class="card">
      <div class="card-header"><span class="card-title">Transaction History</span><div style="display:flex;gap:8px;"><input class="form-input" id="historySearch" placeholder="Search…" style="width:180px;padding:8px 12px;font-size:0.82rem;"></div></div>
      <div class="tx-list" id="historyList"></div>
    </div>
  </div>

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
        <table style="width:100%; border-collapse:collapse;"><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Role</th><th>Created</th><th>Actions</th></tr></thead><tbody id="adminUserTable"></tbody></table>
      </div>
      <div id="adminTransactionsPanel" style="display:none;">
        <select id="adminUserFilter" class="form-select" style="margin-bottom:12px;"><option value="">All Users</option></select>
        <table style="width:100%; border-collapse:collapse;"><thead><tr><th>Date</th><th>User</th><th>Category</th><th>Type</th><th>Amount</th><th>Need/Want</th></tr></thead><tbody id="adminTxTable"></tbody></table>
      </div>
    </div>
  </div>
</main>

<div id="toast"></div>
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

<div class="chatbot" id="chatbot">
  <div class="chat-window">
    <div class="chat-header" id="chatHeader"><div class="ai-pulse"></div><div class="chat-header-title">SmartSpend AI <span class="ai-badge">Gemini</span></div><button class="chat-toggle-btn" id="chatToggleBtn" title="Minimize">–</button></div>
    <div class="chat-msgs" id="chatMsgs"><div class="msg bot">👋 I'm your AI finance assistant. Ask me anything about your money, budget, or how to save more.</div></div>
    <div class="chat-input-row"><input class="chat-inp" id="chatInp" placeholder="Ask anything…"><button class="chat-send" id="chatSend">→</button></div>
  </div>
</div>

<script>
// ── GLOBAL VARS ──
let currentUser = null;
let allTransactions = [];
let currentMindset = 'Neutral';
let aiPlan = null;
let trendChart = null, catChartInst = null;
let isLogin = true;

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
  t.textContent = msg;
  t.style.borderColor = color || 'var(--border)';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 2800);
}

async function api(url, opts={}){
  const res = await fetch(url, {
    ...opts,
    credentials:'include',
    headers:{'Content-Type':'application/json',...(opts.headers||{})}
  });
  if(!res.ok){
    const err = await res.json().catch(()=>({error:'Request failed'}));
    throw new Error(err.error || 'Request failed');
  }
  return res.json();
}

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
    authTitle.textContent = 'Welcome back';
    authSub.textContent = 'Sign in to your SmartSpend account';
    authBtn.textContent = 'Sign In';
    regName.style.display='none';
    authConfirm.style.display='none';
    termsRow.style.display='none';
    toggleAuth.innerHTML = 'No account? <span>Register here</span>';
  } else {
    authTitle.textContent = 'Create account';
    authSub.textContent = 'Start your AI-powered financial journey';
    authBtn.textContent = 'Register';
    regName.style.display='block';
    authConfirm.style.display='block';
    termsRow.style.display='flex';
    toggleAuth.innerHTML = 'Already registered? <span>Sign in</span>';
  }
  authMsg.textContent = '';
});

authBtn.addEventListener('click', async () => {
  authMsg.textContent = '';
  if(!isLogin && authPass.value !== authConfirm.value) {
    authMsg.textContent = 'Passwords do not match';
    return;
  }
  if(!isLogin && !termsCheck.checked) {
    authMsg.textContent = 'You must accept the terms';
    return;
  }
  try {
    const endpoint = isLogin ? '/api/login' : '/api/register';
    const body = isLogin ? { email: authEmail.value, password: authPass.value } :
      { name: regName.value, email: authEmail.value, password: authPass.value };
    const data = await api(endpoint, { method:'POST', body: JSON.stringify(body) });
    currentUser = data;
    authOverlay.style.display = 'none';
    initApp();
  } catch(e) {
    authMsg.textContent = e.message;
  }
});

document.getElementById('signoutBtn').addEventListener('click', async ()=>{
  await api('/api/logout', { method:'POST' });
  currentUser = null;
  authOverlay.style.display = 'flex';
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('screen-dashboard').classList.add('active');
  toast('Signed out');
});

// ── NAVIGATION ──
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

// ── INIT ──
async function initApp() {
  if(!currentUser) return;
  document.getElementById('userAvatar').textContent = currentUser.name?.charAt(0)?.toUpperCase() || '?';
  if(currentUser.role === 'admin') document.getElementById('adminNavBtn').style.display = 'flex';
  else document.getElementById('adminNavBtn').style.display = 'none';
  renderChecklist();
  loadDashboard();
}

function renderChecklist() {
  const needsDiv = document.getElementById('needsChecklist');
  const wantsDiv = document.getElementById('wantsChecklist');
  let needsHtml = '', wantsHtml = '';
  const selectedCategories = aiPlan?.allocation || {};
  categoryConfig.forEach((cat, idx) => {
    const pct = selectedCategories[cat.name] !== undefined ? selectedCategories[cat.name] : cat.defaultPct;
    const itemHtml = `<div class="checklist-item">
      <label><input type="checkbox" class="cat-checkbox" data-cat="${cat.name}" data-idx="${idx}" ${pct>0?'checked':''}> ${cat.name}</label>
      <span class="cat-percent" id="pct-${idx}">${pct.toFixed(0)}%</span>
    </div>`;
    if(cat.type === 'need') needsHtml += itemHtml;
    else if(cat.type === 'want') wantsHtml += itemHtml;
  });
  needsDiv.innerHTML = needsHtml;
  wantsDiv.innerHTML = wantsHtml;
  document.querySelectorAll('.cat-checkbox').forEach(chk => {
    chk.addEventListener('change', () => updateChecklistPercentages());
  });
  updateChecklistPercentages();
}

function updateChecklistPercentages(allocations = null) {
  if(allocations) {
    for(let i=0; i<categoryConfig.length; i++) {
      const cat = categoryConfig[i];
      const pct = allocations[cat.name] !== undefined ? allocations[cat.name] : 0;
      const pctSpan = document.getElementById(`pct-${i}`);
      if(pctSpan) pctSpan.textContent = pct.toFixed(0) + '%';
      categoryConfig[i].defaultPct = pct;
    }
  }
  let total = 0;
  document.querySelectorAll('.cat-checkbox:checked').forEach(chk => {
    const catName = chk.dataset.cat;
    const idx = categoryConfig.findIndex(c => c.name === catName);
    if(idx !== -1) total += categoryConfig[idx].defaultPct;
  });
  const warning = document.getElementById('totalWarning');
  if(Math.abs(total - 100) > 0.01) {
    warning.style.display = 'block';
    warning.textContent = `⚠️ Selected categories total ${total}% – must be 100%. AI will normalise.`;
  } else {
    warning.style.display = 'none';
  }
}

// ── TOOL PICKER TOGGLES (Manual Income / Manual Expense / Auto) ──
document.getElementById('incomeTool').addEventListener('change', function(){
  const val = this.value;
  document.getElementById('manualIncomeBlock').style.display = (val === 'manual-income') ? 'block' : 'none';
  document.getElementById('manualExpenseBlock').style.display = (val === 'manual-expense') ? 'block' : 'none';
  document.getElementById('incomeImageUpload').style.display = (val === 'auto') ? 'block' : 'none';
});

// ── MANUAL EXPENSE LOGIC ──
document.getElementById('addExpenseBtn').addEventListener('click', async ()=>{
  const amount = parseFloat(document.getElementById('expenseAmount').value);
  if(!amount || amount<=0) { toast('Enter a valid amount'); return; }
  const category = document.getElementById('expenseCategory').value;
  const note = document.getElementById('expenseNote').value;
  try {
    let classify = { is_need: true, priority: 1, suggested_note: note };
    try {
      const cl = await api('/api/ai/classify_transaction', { method:'POST', body: JSON.stringify({ amount, category, note }) });
      classify = cl;
      document.getElementById('expenseClassifyResult').innerHTML = `<div class="ai-classify-result">🤖 AI: ${classify.is_need?'Need':'Want'} · Priority ${classify.priority} · Note: ${esc(classify.suggested_note)}</div>`;
      document.getElementById('expenseClassifyResult').style.display = 'block';
    } catch(e) {}
    await api('/api/transactions', { method:'POST', body: JSON.stringify({ amount, category, tx_type: 'expense', is_need: classify.is_need, priority: classify.priority, note: classify.suggested_note||note }) });
    toast('Expense added!');
    document.getElementById('expenseAmount').value = '';
    document.getElementById('expenseNote').value = '';
    loadDashboard(); // refresh stats
  } catch(e) { toast(e.message); }
});

// ── DASHBOARD LOAD ──
async function loadDashboard() {
  if(!currentUser) return;
  try {
    await api('/api/apply_future_expenses', { method:'POST' }); // auto‑process pending
    const summary = await api(`/api/summary/${currentUser.id}`);
    const predict = await api(`/api/predict/${currentUser.id}`);
    const longevity = await api(`/api/longevity/${currentUser.id}`);
    renderStats(summary, predict.score, longevity);
    renderForecast(predict.predictions?.weekly);
    renderTrendChart(summary.monthly);
    document.getElementById('chartBlock').style.display = Object.keys(summary.monthly).length ? 'block' : 'none';
    document.getElementById('forecastBlock').style.display = Object.keys(predict.predictions?.weekly||{}).length ? 'block' : 'none';
  } catch(e) { toast(e.message); }
}

function renderStats(summary, score, longevity) {
  document.getElementById('sBalance').textContent = fmt(summary.balance);
  document.getElementById('sExpense').textContent = fmt(summary.expense);
  document.getElementById('sIncome').textContent = fmt(summary.income);
  document.getElementById('sScore').textContent = score ?? '—';
  document.getElementById('scoreLabel').textContent = score ? 'AI Health Score' : 'awaiting data';
  document.getElementById('topScore').textContent = score ?? '—';
}

function addFeedEvent(icon, text) {
  const feed = document.getElementById('aiFeed');
  const empty = feed.querySelector('.empty-state');
  if (empty) empty.remove();
  const time = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});
  const div = document.createElement('div');
  div.className = 'ai-event';
  div.innerHTML = `<span class="ai-event-icon">${icon}</span><span class="ai-event-text">${esc(text)}</span><span class="ai-event-time">${time}</span>`;
  feed.prepend(div);
}

// ── AI PLAN ──
document.getElementById('analyzeBtn').addEventListener('click', runAIPlan);

async function runAIPlan() {
  const income = parseFloat(document.getElementById('incomeInput').value);
  if(!income || income <= 0){ toast('Enter a valid monthly income first'); return; }
  const selectedCategories = [];
  document.querySelectorAll('.cat-checkbox:checked').forEach(chk => selectedCategories.push(chk.dataset.cat));
  if(!selectedCategories.length){ toast('Please select at least one category'); return; }
  const btn = document.getElementById('analyzeBtn');
  const btnContent = document.getElementById('analyzeBtnContent');
  btn.disabled = true;
  btnContent.innerHTML = '<div class="spinner"></div> Analyzing…';
  addFeedEvent('🤖','AI is analyzing your financial profile…');
  try {
    const result = await api('/api/ai/full_setup', {
      method:'POST',
      body: JSON.stringify({ monthly_income: income, mindset: currentMindset, selected_categories: selectedCategories })
    });
    aiPlan = result;
    if(result.allocation) updateChecklistPercentages(result.allocation);
    addFeedEvent('✅',`Budget allocated across ${Object.keys(result.allocation).length} categories`);
    addFeedEvent('💰',`Savings target set: ${fmt(result.savings_plan.monthly)}/month`);
    addFeedEvent('🧠',`Financial summary: "${result.financial_summary.substring(0,60)}…"`);
    addFeedEvent('📋',`${result.advice.length} personalized insights ready`);
    renderAIPlan(result);
    toast('AI plan complete! 🎉', 'var(--green)');
  } catch(e){
    toast('AI error: '+e.message);
    addFeedEvent('❌','AI error: '+e.message);
  }
  btn.disabled = false;
  btnContent.innerHTML = '🔄 Re‑Analyze';
}

function renderAIPlan(plan) {
  document.getElementById('financialSummaryText').textContent = plan.financial_summary;
  document.getElementById('saveDaily').textContent = fmt(plan.savings_plan.daily);
  document.getElementById('saveWeekly').textContent = fmt(plan.savings_plan.weekly);
  document.getElementById('saveMonthly').textContent = fmt(plan.savings_plan.monthly);
  document.getElementById('savingsTip').textContent = '💡 ' + (plan.savings_plan.tip || '');
  document.getElementById('financialSummaryBlock').style.display = 'block';

  const grid = document.getElementById('allocGrid');
  const needs = new Set(['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage']);
  const savings = new Set(['Savings']);
  grid.innerHTML = Object.entries(plan.allocation).map(([cat, pct])=>{
    const type = savings.has(cat) ? 'savings' : (needs.has(cat) ? 'need' : 'want');
    const amt = plan.allocation_amounts?.[cat] || (plan.monthly_income * pct / 100);
    return `<div class="alloc-item">
      <div class="alloc-cat">${esc(cat)}</div>
      <div class="alloc-pct">${pct.toFixed(0)}<span style="font-size:1rem;color:var(--muted)">%</span></div>
      <div class="alloc-amount">${fmt(amt)}</div>
      <div class="alloc-type ${type}">${type.toUpperCase()}</div>
      <div class="alloc-item-bar ${type}" style="width:${Math.min(pct,100)}%"></div>
    </div>`;
  }).join('');
  document.getElementById('allocationBlock').style.display = 'block';

  const adviceIcons = { info:'ℹ️', warning:'⚠️', success:'✅' };
  document.getElementById('adviceList').innerHTML = plan.advice.map(a=>
    `<div class="advice-card ${esc(a.type)}">
      <span class="advice-icon">${adviceIcons[a.type]||'💡'}</span>
      <div><div class="advice-title">${esc(a.title)}</div><div class="advice-body">${esc(a.body)}</div></div>
    </div>`).join('');
  document.getElementById('adviceBlock').style.display = 'block';
}

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

// ── OCR IMAGE UPLOAD ──
document.getElementById('incomeImage').addEventListener('change', async function(){
  const file = this.files[0];
  if(!file) return;
  try {
    const formData = new FormData();
    formData.append('image', file);
    const resp = await fetch('/api/ocr_income', { method:'POST', body: formData, credentials:'include' });
    const data = await resp.json();
    
if(data.transactions){
  toast(`Extracted ${data.count} transactions`);
  loadDashboard();
}

  } catch(e) { toast('OCR failed: '+e.message); }
});

// Mindset buttons
document.querySelectorAll('.mindset-btn').forEach(btn => {
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('.mindset-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    currentMindset = btn.dataset.mindset;
  });
});

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
    // Fetch longevity data
    const longevity = await api(`/api/longevity/${currentUser.id}`);
    const bal = longevity.balance;
    const avgDaily = longevity.avg_daily_spend;

    // Compute scenarios
    const scenarios = [
      { label: '🏦 Saver',   daily: avgDaily * 0.8,   cssClass: 'saver-fill' },
      { label: '⚖️ Balanced', daily: avgDaily,         cssClass: 'neutral-fill' },
      { label: '🛍️ Spender', daily: avgDaily * 1.2,   cssClass: 'spender-fill' }
    ];

    // Calculate days for each scenario
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

    // Update summary numbers (current = balanced)
    document.getElementById('longevityDays').textContent = scenarios[1].daily > 0 ? Math.floor(bal / scenarios[1].daily) : '—';
    document.getElementById('longevityBalance').textContent = fmt(bal);
    document.getElementById('longevityDaily').textContent = fmt(avgDaily);

    // Load forecast and category chart
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
  const filtered = txs.filter(t=> t.category.toLowerCase().includes(search) || (t.note||'').toLowerCase().includes(search));
  if(!filtered.length) { list.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🔍</div><div class="empty-state-text">No matching transactions.</div></div>'; return; }
  list.innerHTML = filtered.map(t=>`
    <div class="tx-item">
      <div class="tx-cat-icon">${CAT_ICONS[t.category]||'📦'}</div>
      <div class="tx-info">
        <div class="tx-cat">${esc(t.category)} ${t.is_need?'<span class="tx-badge need">Need</span>':'<span class="tx-badge want">Want</span>'}</div>
        <div class="tx-meta">${t.note?esc(t.note)+' · ':''}${fmtDate(t.tx_date)}</div>
      </div>
      <div class="tx-amount ${t.tx_type}">${t.tx_type==='income'?'+':'-'}${fmt(t.amount)}</div>
      <button class="btn-del" onclick="deleteTransaction(${t.id})">🗑</button>
    </div>`).join('');
}
document.getElementById('historySearch').addEventListener('input', ()=> renderHistory(allTransactions));
async function deleteTransaction(id) {
  if(!confirm('Delete this transaction?')) return;
  try {
    await api(`/api/transactions/${id}`, { method:'DELETE' });
    toast('Deleted');
    if(document.getElementById('screen-history').classList.contains('active')) loadHistory();
  } catch(e) { toast(e.message); }
}

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
async function loadAdminUsers(){
  try {
    const users = await api('/api/admin/users');
    const search = document.getElementById('adminSearchUser').value.toLowerCase();
    const filtered = users.filter(u=> u.name.toLowerCase().includes(search) || u.email.toLowerCase().includes(search));
    document.getElementById('adminUserTable').innerHTML = filtered.map(u=>`
      <tr>
        <td>${u.id}</td><td>${esc(u.name)}</td><td>${esc(u.email)}</td><td>${esc(u.role)}</td>
        <td>${fmtDate(u.created_at)}</td>
        <td>
          <button onclick="adminUpdateRole(${u.id},'${u.role==='admin'?'user':'admin'}')" class="btn" style="padding:4px 8px;font-size:0.7rem;">Make ${u.role==='admin'?'User':'Admin'}</button>
          <button onclick="adminDeleteUser(${u.id})" class="btn btn-danger" style="padding:4px 8px;font-size:0.7rem;">Del</button>
        </td>
      </tr>`).join('');
  } catch(e) { toast(e.message); }
}
async function adminUpdateRole(id, newRole){
  try {
    await api(`/api/admin/users/${id}`, { method:'PUT', body: JSON.stringify({ role: newRole }) });
    toast(`Role updated to ${newRole}`); loadAdminUsers();
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


# ----------------------------------------------------------------------
# Initialize DB
# ----------------------------------------------------------------------
with app.app_context():
    db.create_all()
    ensure_schema()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
