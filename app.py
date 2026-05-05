
import json, csv, io, os, re, base64, requests
import google.generativeai as genai

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from functools import wraps

from flask import Flask, request, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from sqlalchemy import inspect, text

# ── APP SETUP ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── FIX 1: Never hardcode secrets. Raise clearly if missing. ──────────────────
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set!\n"
        "Set it in Render dashboard → Environment → SECRET_KEY"
    )
app.config['SECRET_KEY'] = _secret

# ── FIX 2: DATABASE_URL from env, never hardcoded ─────────────────────────────
_db_url = os.environ.get('DATABASE_URL', '')
if not _db_url:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set!\n"
        "In Render, link a PostgreSQL database and it auto-sets DATABASE_URL."
    )
# ── FIX 4: Render gives postgres://, SQLAlchemy 2.x needs postgresql:// ───────
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/tmp'

# Safe pool settings for Render free-tier Postgres
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size':     5,
    'max_overflow':  10,
    'pool_timeout':  30,
    'pool_recycle':  1800,
    'pool_pre_ping': True,
}

# ── FIX 5: Session cookies — required for login to survive on Render HTTPS ────
app.config['SESSION_COOKIE_SECURE']   = True   # send cookie only over HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # allow cross-page navigation
app.config['SESSION_COOKIE_HTTPONLY'] = True   # JS cannot read the cookie

# ── FIX 6: CORS — so the browser sends cookies with API requests ──────────────
CORS(app, supports_credentials=True)

# ── FIX 3: GEMINI_API_KEY from env, never hardcoded ──────────────────────────
_gemini_key = os.environ.get('GEMINI_API_KEY', '')
if not _gemini_key:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set!\n"
        "Set it in Render dashboard → Environment → GEMINI_API_KEY"
    )
genai.configure(api_key=_gemini_key)

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_URL     = 'https://openrouter.ai/api/v1/chat/completions'

# ── AI ROUTER ─────────────────────────────────────────────────────────────────

GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
OPENROUTER_FALLBACKS = [
    'google/gemini-2.0-flash-001',
    'meta-llama/llama-3.2-3b-instruct:free',
    'mistralai/mistral-7b-instruct:free',
]

def route_ai_request(prompt, max_tokens=400):
    """Try Gemini models first, fall back to OpenRouter free models."""
    for name in GEMINI_MODELS:
        try:
            m   = genai.GenerativeModel(name)
            res = m.generate_content(prompt)
            if res and res.text:
                return res.text.strip()
        except Exception as e:
            print(f'Gemini {name} failed: {e}')

    if not OPENROUTER_API_KEY:
        return 'Sorry, all AI services are busy. Try again later.'

    for model in OPENROUTER_FALLBACKS:
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers={'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                         'Content-Type':  'application/json'},
                json={'model': model,
                      'messages': [{'role': 'user', 'content': prompt}],
                      'max_tokens': max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'OpenRouter {model} failed: {e}')

    return '⚠️ All AI services unavailable.'

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from AI JSON output."""
    raw = raw.strip()
    if raw.startswith('```'):
        parts = raw.split('```')
        raw   = parts[1] if len(parts) > 1 else raw
        if raw.startswith('json'):
            raw = raw[4:]
    return raw.strip()

# ── DATABASE ──────────────────────────────────────────────────────────────────

db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ── MODELS ────────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id                   = db.Column(db.Integer,      primary_key=True)
    name                 = db.Column(db.String(100),  nullable=False)
    email                = db.Column(db.String(120),  unique=True, nullable=False)
    password_hash        = db.Column(db.String(256),  nullable=False)
    social_status        = db.Column(db.String(20),   default='Middle')
    spending_mindset     = db.Column(db.String(20),   default='Neutral')
    monthly_budget_limit = db.Column(db.Float,        default=0.0)
    created_at           = db.Column(db.DateTime,     default=lambda: datetime.now(timezone.utc))
    role                 = db.Column(db.String(20),   default='user')
    avatar_url           = db.Column(db.String(1000), nullable=True)

    transactions    = db.relationship('Transaction',   backref='user', lazy=True)
    budgets         = db.relationship('Budget',        backref='user', lazy=True)
    future_expenses = db.relationship('FutureExpense', backref='user', lazy=True)
    allocations     = db.relationship('UserAllocation',backref='user', lazy=True)

    def set_password(self, pw):
        self.password_hash = bcrypt.generate_password_hash(pw).decode('utf-8')

    def check_password(self, pw):
        try:
            return bcrypt.check_password_hash(self.password_hash, pw)
        except Exception:
            return False

    def to_dict(self):
        return {
            'id':                   self.id,
            'name':                 self.name,
            'email':                self.email,
            'social_status':        self.social_status,
            'spending_mindset':     self.spending_mindset,
            'monthly_budget_limit': self.monthly_budget_limit,
            'role':                 self.role,
            'avatar_url':           self.avatar_url,
            'created_at':           self.created_at.isoformat() if self.created_at else None,
        }


class Transaction(db.Model):
    __tablename__ = 'transactions'
    id        = db.Column(db.Integer,     primary_key=True)
    user_id   = db.Column(db.Integer,     db.ForeignKey('users.id'), nullable=False)
    amount    = db.Column(db.Float,       nullable=False)
    category  = db.Column(db.String(50),  nullable=False)
    tx_type   = db.Column(db.String(10),  nullable=False)
    is_need   = db.Column(db.Boolean,     default=True)
    priority  = db.Column(db.Integer,     default=1)
    note      = db.Column(db.String(200), default='')
    tx_date   = db.Column(db.DateTime,    default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id':       self.id,
            'user_id':  self.user_id,
            'amount':   self.amount,
            'category': self.category,
            'tx_type':  self.tx_type,
            'is_need':  self.is_need,
            'priority': self.priority,
            'note':     self.note or '',
            'tx_date':  self.tx_date.isoformat(),
        }


class Budget(db.Model):
    __tablename__ = 'budgets'
    id           = db.Column(db.Integer,    primary_key=True)
    user_id      = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    category     = db.Column(db.String(50), nullable=False)
    limit_amount = db.Column(db.Float,      nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category', name='uq_user_cat'),)


class FutureExpense(db.Model):
    __tablename__ = 'future_expenses'
    id           = db.Column(db.Integer,      primary_key=True)
    user_id      = db.Column(db.Integer,      db.ForeignKey('users.id'), nullable=False)
    description  = db.Column(db.String(200),  nullable=False)
    amount       = db.Column(db.Float,        nullable=False)
    category     = db.Column(db.String(50),   nullable=False)
    cycle        = db.Column(db.String(20),   default='One-time')
    expense_date = db.Column(db.Date,         nullable=False)
    created_at   = db.Column(db.DateTime,     default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id':          self.id,
            'description': self.description,
            'amount':      self.amount,
            'category':    self.category,
            'cycle':       self.cycle,
            'date':        self.expense_date.isoformat(),
        }


class UserAllocation(db.Model):
    __tablename__ = 'user_allocations'
    id            = db.Column(db.Integer,    primary_key=True)
    user_id       = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    category_name = db.Column(db.String(50), nullable=False)
    type          = db.Column(db.String(10), nullable=False)
    percentage    = db.Column(db.Float,      nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category_name', name='uq_user_cat_alloc'),)

# ── SCHEMA MIGRATION ──────────────────────────────────────────────────────────

def ensure_schema():
    """
    FIX 7 & 8: Every ALTER wrapped in its own try/except.
    Safe on fresh DB, re-deploy, and existing DBs.
    """
    inspector = inspect(db.engine)

    # ── users table ──────────────────────────────────────────────────────────
    if inspector.has_table('users'):
        cols = [c['name'] for c in inspector.get_columns('users')]

        # FIX 8: Only attempt password migration if both columns exist safely
        if 'password' in cols and 'password_hash' in cols:
            try:
                with db.engine.connect() as conn:
                    rows = conn.execute(
                        text("SELECT id, password FROM users WHERE password IS NOT NULL AND password != ''")
                    ).fetchall()
                    for uid, plain_pw in rows:
                        try:
                            hashed = bcrypt.generate_password_hash(plain_pw).decode('utf-8')
                            conn.execute(
                                text("UPDATE users SET password_hash = :h WHERE id = :uid"),
                                {'h': hashed, 'uid': uid}
                            )
                        except Exception:
                            pass  # skip rows that already have bcrypt hashes
                    conn.execute(text('ALTER TABLE users DROP COLUMN password'))
                    conn.commit()
            except Exception as e:
                print(f'Password migration warning (safe to ignore): {e}')

        # Add missing columns — each in its own try/except
        _add_col(cols, 'password_hash',        "VARCHAR(256) NOT NULL DEFAULT ''")
        _add_col(cols, 'social_status',         "VARCHAR(20)  DEFAULT 'Middle'")
        _add_col(cols, 'spending_mindset',       "VARCHAR(20)  DEFAULT 'Neutral'")
        _add_col(cols, 'monthly_budget_limit',   'FLOAT        DEFAULT 0.0')
        _add_col(cols, 'role',                   "VARCHAR(20)  DEFAULT 'user'")
        _add_col(cols, 'avatar_url',             'VARCHAR(1000)')

    # ── transactions table ───────────────────────────────────────────────────
    if inspector.has_table('transactions'):
        tx_cols = [c['name'] for c in inspector.get_columns('transactions')]
        _add_col(tx_cols, 'is_need',  'BOOLEAN DEFAULT TRUE',  'transactions')
        _add_col(tx_cols, 'priority', 'INTEGER DEFAULT 1',     'transactions')

    # Create tables that don't exist yet
    db.create_all()

    # Seed default admin if none exists
    try:
        admin = User.query.filter_by(email='admin@smartspend.com').first()
        if not admin:
            admin = User(name='Admin', email='admin@smartspend.com', role='admin')
            admin.set_password('Admin@2026!')
            db.session.add(admin)
            db.session.commit()
            print('✅ Admin seeded: admin@smartspend.com / Admin@2026!')
        elif admin.role != 'admin':
            admin.role = 'admin'
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f'Admin seed warning: {e}')


def _add_col(existing_cols, col_name, definition, table='users'):
    """Helper: add a column only if it doesn't exist, silently skip if it does."""
    if col_name not in existing_cols:
        try:
            with db.engine.connect() as conn:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col_name} {definition}'))
                conn.commit()
        except Exception as e:
            print(f'Could not add {table}.{col_name} (safe to ignore): {e}')

# ── AUTH DECORATORS ───────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized — please log in'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized — please log in'}), 401
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    return User.query.get(session['user_id']) if 'user_id' in session else None

# ── ANALYTICS HELPERS ─────────────────────────────────────────────────────────

def compute_health_score(user_id):
    ago30 = datetime.now(timezone.utc) - timedelta(days=30)
    exps  = Transaction.query.filter(Transaction.user_id == user_id,
                                     Transaction.tx_type == 'expense',
                                     Transaction.tx_date >= ago30).all()
    incs  = Transaction.query.filter(Transaction.user_id == user_id,
                                     Transaction.tx_type == 'income',
                                     Transaction.tx_date >= ago30).all()
    total_exp = sum(e.amount for e in exps)
    total_inc = sum(i.amount for i in incs)
    savings_rate = max(0, (total_inc - total_exp) / total_inc) if total_inc > 0 else 0
    want_ratio   = sum(e.amount for e in exps if not e.is_need) / (total_exp or 1)
    budgets      = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    overspend    = sum(
        (sum(e.amount for e in exps if e.category == cat) - lim) / lim
        for cat, lim in budgets.items()
        if lim > 0 and sum(e.amount for e in exps if e.category == cat) > lim
    )
    return max(0, min(100, int(70 + savings_rate * 20 - want_ratio * 15 - min(20, overspend * 10))))


def generate_weekly_forecast(user_id):
    ago30 = datetime.now(timezone.utc) - timedelta(days=30)
    exps  = Transaction.query.filter(Transaction.user_id == user_id,
                                     Transaction.tx_type == 'expense',
                                     Transaction.tx_date >= ago30).all()
    if not exps:
        return {f'Week {i}': 0 for i in range(1, 5)}
    daily = defaultdict(float)
    for e in exps:
        daily[e.tx_date.date()] += e.amount
    values = [daily[d] for d in sorted(daily)]
    n = len(values)
    if n > 1:
        xm = (n - 1) / 2
        ym = sum(values) / n
        num = sum((i - xm) * (values[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n)) or 1
        slope = num / den
        intercept = ym - slope * xm
        weeks = [0, 0, 0, 0]
        for day in range(1, 29):
            weeks[(day - 1) // 7] += max(0, intercept + slope * (n + day))
        return {f'Week {i+1}': round(weeks[i], 2) for i in range(4)}
    avg = sum(values) / n * 7
    return {f'Week {i+1}': round(avg, 2) for i in range(4)}


def get_category_totals(user_id):
    now   = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    exps  = Transaction.query.filter(Transaction.user_id == user_id,
                                     Transaction.tx_type == 'expense',
                                     Transaction.tx_date >= start).all()
    totals = defaultdict(float)
    for e in exps:
        totals[e.category] += e.amount
    return dict(totals)


def get_monthly_summary(user_id):
    all_tx  = Transaction.query.filter_by(user_id=user_id).all()
    balance = sum(t.amount if t.tx_type == 'income' else -t.amount for t in all_tx)
    now     = datetime.now(timezone.utc)
    start   = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    this_exp = sum(t.amount for t in all_tx if t.tx_type == 'expense' and t.tx_date >= start)
    this_inc = sum(t.amount for t in all_tx if t.tx_type == 'income'  and t.tx_date >= start)
    monthly  = defaultdict(lambda: {'income': 0.0, 'expense': 0.0})
    for t in all_tx:
        k = t.tx_date.strftime('%Y-%m')
        monthly[k]['income' if t.tx_type == 'income' else 'expense'] += t.amount
    last6 = sorted(monthly)[-6:]
    return {
        'balance': round(balance, 2),
        'expense': round(this_exp, 2),
        'income':  round(this_inc, 2),
        'monthly': {m: monthly[m] for m in last6},
    }


def compute_longevity(user_id):
    s      = get_monthly_summary(user_id)
    ago30  = datetime.now(timezone.utc) - timedelta(days=30)
    exps   = Transaction.query.filter(Transaction.user_id == user_id,
                                      Transaction.tx_type == 'expense',
                                      Transaction.tx_date >= ago30).all()
    avg    = sum(e.amount for e in exps) / 30 if exps else 0
    days   = int(s['balance'] / avg) if avg > 0 else 0
    return {'balance': s['balance'], 'avg_daily_spend': round(avg, 2), 'days': days}

# ── AUTH ROUTES ───────────────────────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    email= (data.get('email') or '').strip().lower()
    pw   = data.get('password', '')
    if not name or not email or not pw:
        return jsonify({'error': 'Name, email and password are required'}), 400
    if len(pw) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    user = User(name=name, email=email, role='user')
    user.set_password(pw)
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify(user.to_dict()), 201


@app.route('/api/login', methods=['POST'])
def login():
    """
    FIX: Proper session assignment. Returns user dict on success so the
    frontend can update currentUser immediately without an extra /api/me call.
    """
    data  = request.json or {}
    email = (data.get('email') or '').strip().lower()
    pw    = data.get('password', '')
    if not email or not pw:
        return jsonify({'error': 'Email and password are required'}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(pw):
        return jsonify({'error': 'Invalid email or password'}), 401
    # Write session — this is what keeps the user logged in
    session.permanent = True          # survive browser restart
    session['user_id'] = user.id
    return jsonify(user.to_dict()), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'}), 200


@app.route('/api/me')
def me():
    """
    Called on every page load to restore session. Returns 401 if not logged in
    so the frontend shows the login screen.
    """
    uid = session.get('user_id')
    if not uid:
        return jsonify({'error': 'Not logged in'}), 401
    user = User.query.get(uid)
    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 401
    return jsonify(user.to_dict())

# ── PROFILE ROUTES ────────────────────────────────────────────────────────────

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile():
    user = get_current_user()
    data = request.json or {}
    if 'name'       in data: user.name       = data['name'].strip()
    if 'email'      in data: user.email      = data['email'].strip().lower()
    if 'avatar_url' in data: user.avatar_url = data['avatar_url']
    if data.get('password'):
        if len(data['password']) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        user.set_password(data['password'])
    db.session.commit()
    return jsonify(user.to_dict())


@app.route('/api/me/avatar', methods=['PUT'])
@login_required
def update_avatar():
    user           = get_current_user()
    user.avatar_url= (request.json or {}).get('avatar_url', '')
    db.session.commit()
    return jsonify({'avatar_url': user.avatar_url})

# ── TRANSACTION ROUTES ────────────────────────────────────────────────────────

@app.route('/api/transactions', methods=['GET'])
@login_required
def list_transactions():
    user = get_current_user()
    txs  = Transaction.query.filter_by(user_id=user.id)\
                            .order_by(Transaction.tx_date.desc()).all()
    return jsonify([t.to_dict() for t in txs])


@app.route('/api/transactions', methods=['POST'])
@login_required
def create_transaction():
    """
    FIX 9 & 10:
    - Amount must be >= 1 (no negative or zero values)
    - Expenses blocked if budget for category is zero
    - Expenses blocked if they would exceed remaining monthly income
    """
    user = get_current_user()
    data = request.json or {}

    if not all(k in data for k in ('amount', 'category', 'tx_type')):
        return jsonify({'error': 'amount, category and tx_type are required'}), 400

    amount = float(data['amount'])

    # ── FIX 9: Enforce minimum amount of 1 ────────────────────────────────
    if amount < 1:
        return jsonify({'error': 'Amount must be at least ₱1'}), 400

    category = data['category']
    tx_type  = data['tx_type']

    if tx_type == 'expense':
        summary = get_monthly_summary(user.id)

        # ── FIX 10a: Block if budget for this category is zero ─────────────
        budget = Budget.query.filter_by(user_id=user.id, category=category).first()
        if budget is not None and budget.limit_amount <= 0:
            return jsonify({
                'error': f'Budget for "{category}" is ₱0. '
                         'Set a budget limit first before adding expenses.'
            }), 400

        # ── FIX 10b: Block if expense exceeds available monthly income ──────
        remaining = summary['income'] - summary['expense']
        if amount > remaining and summary['income'] > 0:
            return jsonify({
                'error': f'This expense (₱{amount:,.2f}) exceeds your remaining '
                         f'monthly balance (₱{remaining:,.2f}). '
                         'Add more income or reduce the amount.'
            }), 400

    # Parse optional custom date
    tx_date = datetime.now(timezone.utc)
    if data.get('tx_date'):
        try:
            tx_date = datetime.fromisoformat(data['tx_date'])
            if tx_date.tzinfo is None:
                tx_date = tx_date.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    tx = Transaction(
        user_id  = user.id,
        amount   = amount,
        category = category,
        tx_type  = tx_type,
        is_need  = data.get('is_need', True),
        priority = data.get('priority', 1),
        note     = data.get('note', ''),
        tx_date  = tx_date,
    )
    db.session.add(tx)
    db.session.commit()

    # Auto-apply past-due one-time future expenses when salary income is logged
    if tx.tx_type == 'income' and tx.category.lower() == 'salary':
        today = datetime.now(timezone.utc).date()
        needs = {'Food & Dining', 'Debt repayment', 'Mortgage', 'Transport'}
        for exp in FutureExpense.query.filter(
            FutureExpense.user_id     == user.id,
            FutureExpense.expense_date<= today,
            FutureExpense.cycle       == 'One-time'
        ).all():
            db.session.add(Transaction(
                user_id  = user.id, amount = exp.amount, category = exp.category,
                tx_type  = 'expense', is_need = exp.category in needs,
                priority = 1,        note    = f'Auto: {exp.description}'
            ))
            db.session.delete(exp)
        db.session.commit()

    return jsonify(tx.to_dict()), 201


@app.route('/api/transactions/<int:tx_id>', methods=['DELETE'])
@login_required
def delete_transaction(tx_id):
    user = get_current_user()
    tx   = Transaction.query.filter_by(id=tx_id, user_id=user.id).first()
    if not tx:
        return jsonify({'error': 'Transaction not found'}), 404
    db.session.delete(tx)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200

# ── BUDGET ROUTES ─────────────────────────────────────────────────────────────

@app.route('/api/budgets/<int:uid>', methods=['GET'])
@login_required
def get_budgets(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify([{'category': b.category, 'limit': b.limit_amount}
                    for b in Budget.query.filter_by(user_id=uid).all()])


@app.route('/api/budgets/<int:uid>', methods=['POST'])
@login_required
def upsert_budget(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    data     = request.json or {}
    category = data.get('category', '').strip()
    limit    = data.get('limit')
    if not category or limit is None:
        return jsonify({'error': 'category and limit are required'}), 400
    b = Budget.query.filter_by(user_id=uid, category=category).first()
    if b:
        b.limit_amount = float(limit)
    else:
        db.session.add(Budget(user_id=uid, category=category, limit_amount=float(limit)))
    db.session.commit()
    return jsonify({'message': 'Budget saved'})


@app.route('/api/budgets/<int:uid>/<path:category>', methods=['PUT'])
@login_required
def update_budget(uid, category):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    data  = request.json or {}
    limit = data.get('limit')
    if limit is None:
        return jsonify({'error': 'limit is required'}), 400
    b = Budget.query.filter_by(user_id=uid, category=category).first()
    if b:
        b.limit_amount = float(limit)
    else:
        db.session.add(Budget(user_id=uid, category=category, limit_amount=float(limit)))
    db.session.commit()
    return jsonify({'message': 'Budget updated'})


@app.route('/api/budgets/reset_to_ai/<int:uid>', methods=['POST'])
@login_required
def reset_budgets_to_ai(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    user   = get_current_user()
    allocs = UserAllocation.query.filter_by(user_id=uid).all()
    if not allocs or user.monthly_budget_limit <= 0:
        return jsonify({'error': 'No AI allocation found. Run AI Plan first.'}), 400
    Budget.query.filter_by(user_id=uid).delete()
    for a in allocs:
        db.session.add(Budget(
            user_id      = uid,
            category     = a.category_name,
            limit_amount = round(user.monthly_budget_limit * a.percentage / 100, 2)
        ))
    db.session.commit()
    return jsonify({'message': 'Budgets reset to AI recommendations'})

# ── SUMMARY / PREDICT / LONGEVITY ─────────────────────────────────────────────

@app.route('/api/summary/<int:uid>')
@login_required
def summary(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(get_monthly_summary(uid))


@app.route('/api/predict/<int:uid>')
@login_required
def predict(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    if not Transaction.query.filter_by(user_id=uid, tx_type='expense').count():
        return jsonify({'has_data': False, 'score': None,
                        'predictions': {'weekly': {}, 'categories': {}}, 'advice': []})
    return jsonify({
        'has_data':    True,
        'score':       compute_health_score(uid),
        'predictions': {
            'weekly':     generate_weekly_forecast(uid),
            'categories': get_category_totals(uid),
        },
        'advice': [],
    })


@app.route('/api/longevity/<int:uid>')
@login_required
def longevity(uid):
    if get_current_user().id != uid:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(compute_longevity(uid))

# ── FUTURE EXPENSES ───────────────────────────────────────────────────────────

@app.route('/api/future_expenses', methods=['GET'])
@login_required
def get_future_expenses():
    user = get_current_user()
    return jsonify([e.to_dict() for e in
                    FutureExpense.query.filter_by(user_id=user.id)
                                 .order_by(FutureExpense.expense_date).all()])


@app.route('/api/future_expenses', methods=['POST'])
@login_required
def create_future_expense():
    user = get_current_user()
    data = request.json or {}
    if not all(k in data for k in ('description', 'amount', 'category', 'date')):
        return jsonify({'error': 'Missing fields'}), 400
    # FIX: enforce min amount of 1
    if float(data['amount']) < 1:
        return jsonify({'error': 'Amount must be at least ₱1'}), 400
    exp = FutureExpense(
        user_id      = user.id,
        description  = data['description'],
        amount       = float(data['amount']),
        category     = data['category'],
        cycle        = data.get('cycle', 'One-time'),
        expense_date = datetime.fromisoformat(data['date']).date(),
    )
    db.session.add(exp)
    db.session.commit()
    return jsonify(exp.to_dict()), 201


@app.route('/api/future_expenses/<int:exp_id>', methods=['DELETE'])
@login_required
def delete_future_expense(exp_id):
    user = get_current_user()
    exp  = FutureExpense.query.filter_by(id=exp_id, user_id=user.id).first()
    if not exp:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(exp)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200


@app.route('/api/apply_future_expenses', methods=['POST'])
@login_required
def apply_future_expenses():
    user  = get_current_user()
    today = datetime.now(timezone.utc).date()
    needs = {'Food & Dining', 'Debt repayment', 'Mortgage', 'Transport'}
    applied = []

    # One-time expenses: apply then delete
    for exp in FutureExpense.query.filter(
        FutureExpense.user_id      == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle        == 'One-time'
    ).all():
        db.session.add(Transaction(
            user_id=user.id, amount=exp.amount, category=exp.category,
            tx_type='expense', is_need=exp.category in needs,
            priority=1, note=f'Auto: {exp.description}'
        ))
        applied.append(exp.description)
        db.session.delete(exp)

    # Recurring expenses: apply then advance date
    for exp in FutureExpense.query.filter(
        FutureExpense.user_id      == user.id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle.in_(['Weekly', 'Monthly'])
    ).all():
        db.session.add(Transaction(
            user_id=user.id, amount=exp.amount, category=exp.category,
            tx_type='expense', is_need=exp.category in needs,
            priority=1, note=f'Recurring: {exp.description} ({exp.cycle})'
        ))
        applied.append(f'{exp.description} ({exp.cycle})')
        delta = timedelta(weeks=1) if exp.cycle == 'Weekly' else timedelta(days=30)
        next_date = exp.expense_date + delta
        while next_date <= today:
            next_date += delta
        exp.expense_date = next_date

    db.session.commit()
    return jsonify({'applied': applied, 'count': len(applied)}), 200

# ── ALLOCATIONS ───────────────────────────────────────────────────────────────

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
        return jsonify({'error': 'Expected a list'}), 400
    UserAllocation.query.filter_by(user_id=user.id).delete()
    for item in data:
        db.session.add(UserAllocation(
            user_id=user.id, category_name=item['category_name'],
            type=item['type'], percentage=item['percentage']
        ))
    db.session.commit()
    return jsonify({'message': 'Saved'}), 200

# ── EXPORT ────────────────────────────────────────────────────────────────────

@app.route('/api/export/csv')
@login_required
def export_csv():
    user   = get_current_user()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Category', 'Type', 'Need/Want', 'Priority', 'Note', 'Amount'])
    for t in Transaction.query.filter_by(user_id=user.id)\
                               .order_by(Transaction.tx_date.desc()).all():
        writer.writerow([
            t.tx_date.strftime('%Y-%m-%d %H:%M'),
            t.category, t.tx_type,
            'Need' if t.is_need else 'Want',
            t.priority, t.note or '', t.amount,
        ])
    resp = Response(output.getvalue(), mimetype='text/csv')
    resp.headers.set('Content-Disposition', 'attachment', filename='transactions.csv')
    return resp

# ── OCR ───────────────────────────────────────────────────────────────────────

@app.route('/api/ocr_income', methods=['POST'])
@login_required
def ocr_income():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    img_b64   = base64.b64encode(file.read()).decode('utf-8')
    mime_type = file.mimetype or 'image/jpeg'

    prompt = """You are a financial OCR assistant. Analyse the image and extract all financial items.
Return ONLY a JSON array. Each object must have:
- "type": "income" or "expense"
- "amount": positive number
- "category": one of Food & Dining, Transport, Groceries, Health, Entertainment, Debt repayment, Mortgage, Subscription, Hobbies, Salary, Other
- "note": short description (max 40 chars)
No markdown. No extra text."""

    raw = None
    try:
        m   = genai.GenerativeModel('gemini-1.5-flash')
        res = m.generate_content([prompt, {'mime_type': mime_type, 'data': img_b64}])
        if res and res.text:
            raw = res.text.strip()
    except Exception as e:
        print(f'OCR error: {e}')

    if not raw:
        return jsonify({'error': 'Could not read image'}), 500

    try:
        items = json.loads(_strip_fences(raw))
        if not isinstance(items, list):
            items = []
    except Exception:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        items = json.loads(match.group(0)) if match else []

    # Validate amounts
    items = [i for i in items if isinstance(i, dict) and float(i.get('amount', 0)) >= 1]
    return jsonify({'transactions': items}), 200

# ── AI ROUTES ─────────────────────────────────────────────────────────────────

@app.route('/api/ai/full_setup', methods=['POST'])
@login_required
def ai_full_setup():
    user = get_current_user()
    data = request.json or {}

    monthly_income     = float(data.get('monthly_income', user.monthly_budget_limit or 0))
    mindset            = data.get('mindset', user.spending_mindset)
    social_status      = data.get('social_status', user.social_status)
    selected_categories= data.get('selected_categories', [])

    if not selected_categories:
        selected_categories = ['Food & Dining', 'Transport', 'Groceries',
                               'Health', 'Entertainment', 'Savings']

    user.monthly_budget_limit = monthly_income
    user.spending_mindset     = mindset
    user.social_status        = social_status
    db.session.commit()

    recent = defaultdict(float)
    for t in Transaction.query.filter_by(user_id=user.id, tx_type='expense')\
                               .order_by(Transaction.tx_date.desc()).limit(30).all():
        recent[t.category] += t.amount

    # ── Allocation ────────────────────────────────────────────────────────
    prompt = f"""You are a Filipino financial planner.
Monthly income: ₱{monthly_income:,.2f}, mindset: {mindset}, status: {social_status}.
Recent spending: {dict(recent)}.
Selected categories: {', '.join(selected_categories)}

Rules:
- Allocate ONLY the selected categories. Total must equal exactly 100.
- Needs (Food & Dining, Transport, Groceries, Health, Debt repayment, Mortgage): 50-70% total.
- Savings: 10-20%.
- Wants (Entertainment, Subscription, Hobbies, etc.): remainder.
Return ONLY a JSON object, no markdown.
Example: {{"Food & Dining": 30.0, "Transport": 10.0, "Savings": 15.0, "Entertainment": 15.0}}"""

    allocation = None
    try:
        raw        = _strip_fences(route_ai_request(prompt, 600))
        allocation = json.loads(raw)
        if not isinstance(allocation, dict):
            allocation = None
    except Exception as e:
        print(f'Allocation parse error: {e}')

    if not allocation:
        # Smart fallback
        needs_set   = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}
        savings_set = {'Savings'}
        wts = {'Saver': (1.5,0.5,1.2), 'Spender': (1.0,1.5,0.7)}.get(mindset, (1.0,1.0,1.0))
        nw, ww, sw = wts
        total_w = sum(
            nw if c in needs_set else sw if c in savings_set else ww
            for c in selected_categories
        ) or 1
        allocation = {
            c: round((nw if c in needs_set else sw if c in savings_set else ww) / total_w * 100, 1)
            for c in selected_categories
        }

    # Keep only selected, normalise to 100
    allocation = {k: v for k, v in allocation.items() if k in selected_categories}
    total = sum(allocation.values()) or 1
    if abs(total - 100) > 0.1:
        allocation = {k: round(v / total * 100, 1) for k, v in allocation.items()}

    # ── Savings plan ──────────────────────────────────────────────────────
    savings_plan = None
    try:
        sp_prompt = (
            f"Monthly income ₱{monthly_income:.2f}, expenses ₱{sum(recent.values()):.2f}, "
            f"mindset {mindset}. Return ONLY JSON: "
            '{"daily":float,"weekly":float,"monthly":float,"tip":"string"}'
        )
        savings_plan = json.loads(_strip_fences(route_ai_request(sp_prompt, 200)))
    except Exception:
        pass

    if not savings_plan:
        ms = monthly_income * 0.2
        savings_plan = {
            'daily': round(ms/30,2), 'weekly': round(ms/4,2),
            'monthly': round(ms,2),  'tip': 'Automate your savings on payday.'
        }

    # ── Advice ────────────────────────────────────────────────────────────
    advice = []
    try:
        adv_prompt = (
            f"Allocations: {allocation}. Spending: {dict(recent)}. "
            "Give 3 short financial advice items. "
            'Return ONLY JSON array: [{"title":"...","body":"...","type":"info|warning|success"}]'
        )
        advice = json.loads(_strip_fences(route_ai_request(adv_prompt, 300)))
        if not isinstance(advice, list):
            advice = []
    except Exception:
        advice = [
            {'title':'Stay Consistent','body':'Track every expense daily.','type':'info'},
            {'title':'Save First','body':'Transfer savings immediately on payday.','type':'success'},
            {'title':'Review Wants','body':'Audit subscriptions every month.','type':'warning'},
        ]

    # ── Financial summary ─────────────────────────────────────────────────
    fs = 'Your AI plan is ready. Start logging expenses for personalized insights.'
    try:
        fs = route_ai_request(
            f"One sentence: financial outlook for Filipino, income ₱{monthly_income}, mindset {mindset}.",
            100
        ).strip()
    except Exception:
        pass

    # ── Persist ───────────────────────────────────────────────────────────
    needs_set   = {'Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'}
    savings_set = {'Savings'}

    UserAllocation.query.filter_by(user_id=user.id).delete()
    for cat, pct in allocation.items():
        t = 'need' if cat in needs_set else 'savings' if cat in savings_set else 'want'
        db.session.add(UserAllocation(user_id=user.id, category_name=cat, type=t, percentage=pct))

    Budget.query.filter_by(user_id=user.id).delete()
    for cat, pct in allocation.items():
        db.session.add(Budget(user_id=user.id, category=cat,
                              limit_amount=round(monthly_income * pct / 100, 2)))
    db.session.commit()

    return jsonify({
        'allocation':        allocation,
        'allocation_amounts':{c: round(monthly_income * p / 100, 2) for c, p in allocation.items()},
        'savings_plan':      savings_plan,
        'advice':            advice,
        'financial_summary': fs,
        'monthly_income':    monthly_income,
    }), 200


@app.route('/api/ai/classify_transaction', methods=['POST'])
@login_required
def ai_classify_transaction():
    data = request.json or {}
    note = (data.get('note') or '')[:100]
    if not note:
        return jsonify({'error': 'note is required'}), 400
    prompt = (
        f'Classify this expense: "{note}". '
        'Return ONLY JSON: {"category":"string","is_need":bool}'
    )
    try:
        result = json.loads(_strip_fences(route_ai_request(prompt, 100)))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/chat', methods=['POST'])
@login_required
def ai_chat():
    data = request.json or {}
    msg  = data.get('message', '').strip()
    if not msg:
        return jsonify({'error': 'No message'}), 400
    user    = get_current_user()
    summary = get_monthly_summary(user.id)
    prompt  = (
        f"You are a friendly Filipino financial assistant for {user.name}. "
        f"Balance: ₱{summary['balance']:,.2f}, income: ₱{summary['income']:,.2f}, "
        f"expenses: ₱{summary['expense']:,.2f}. "
        f"User: {msg}\nAI:"
    )
    return jsonify({'reply': route_ai_request(prompt, 300)}), 200

# ── ADMIN ROUTES ──────────────────────────────────────────────────────────────

@app.route('/api/admin/stats')
@admin_required
def admin_stats():
    users = User.query.all()
    return jsonify({
        'total_users':        len(users),
        'total_transactions': Transaction.query.count(),
        'total_income':       db.session.query(db.func.sum(Transaction.amount)).filter_by(tx_type='income').scalar()  or 0,
        'total_expense':      db.session.query(db.func.sum(Transaction.amount)).filter_by(tx_type='expense').scalar() or 0,
        'avg_health_score':   round(sum(compute_health_score(u.id) for u in users) / len(users), 1) if users else 0,
    })


@app.route('/api/admin/users')
@admin_required
def admin_users():
    return jsonify([u.to_dict() for u in User.query.all()])


@app.route('/api/admin/users/<int:uid>', methods=['PUT'])
@admin_required
def admin_update_user(uid):
    data = request.json or {}
    user = User.query.get_or_404(uid)
    if 'role' in data: user.role = data['role']
    if 'name' in data: user.name = data['name']
    db.session.commit()
    return jsonify(user.to_dict())


@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@admin_required
def admin_delete_user(uid):
    if uid == session.get('user_id'):
        return jsonify({'error': 'Cannot delete your own account'}), 403
    user = User.query.get_or_404(uid)
    for Model in (Transaction, Budget, FutureExpense, UserAllocation):
        Model.query.filter_by(user_id=uid).delete()
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted'})


@app.route('/api/admin/transactions')
@admin_required
def admin_transactions():
    uid   = request.args.get('user_id', type=int)
    query = Transaction.query
    if uid:
        query = query.filter_by(user_id=uid)
    return jsonify([t.to_dict() for t in query.order_by(Transaction.tx_date.desc()).all()])


@app.route('/api/admin/allocations/<int:uid>')
@admin_required
def admin_allocations(uid):
    return jsonify([{'category_name': a.category_name, 'type': a.type, 'percentage': a.percentage}
                    for a in UserAllocation.query.filter_by(user_id=uid).all()])


@app.route('/api/admin/budgets/<int:uid>')
@admin_required
def admin_budgets(uid):
    return jsonify([{'category': b.category, 'limit': b.limit_amount}
                    for b in Budget.query.filter_by(user_id=uid).all()])


@app.route('/api/admin/future_expenses/<int:uid>')
@admin_required
def admin_future_expenses(uid):
    return jsonify([e.to_dict() for e in FutureExpense.query.filter_by(user_id=uid).all()])

# ── MAIN ROUTE ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return HTML_PAGE

# ── STARTUP ───────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()
    ensure_schema()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


# ═════════════════════════════════════════════════════════════════════════════
# HTML PAGE (kept intact from your original — only the Python backend was fixed)
# ═════════════════════════════════════════════════════════════════════════════
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
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");pointer-events:none;z-index:0;opacity:0.4;}
.sidebar{position:fixed;left:0;top:0;bottom:0;width:72px;background:rgba(11,17,32,0.95);backdrop-filter:blur(20px);border-right:1px solid var(--border2);display:flex;flex-direction:column;align-items:center;padding:20px 0;gap:6px;z-index:200;transition:width 0.3s cubic-bezier(0.4,0,0.2,1);}
.sidebar:hover{width:220px;}
.logo{width:44px;height:44px;border-radius:12px;margin-bottom:20px;background:linear-gradient(135deg,var(--green),#009e6a);display:flex;align-items:center;justify-content:center;font-size:1.4rem;cursor:pointer;flex-shrink:0;box-shadow:0 0 24px var(--green-glow);}
.nav-item{width:calc(100% - 16px);display:flex;align-items:center;gap:14px;padding:12px 14px;border-radius:10px;cursor:pointer;border:none;background:transparent;color:var(--muted2);font-family:var(--font-display);font-size:0.875rem;font-weight:500;white-space:nowrap;overflow:hidden;transition:all 0.2s;text-align:left;}
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
.health-pill{display:flex;align-items:center;gap:8px;padding:7px 16px;border-radius:99px;background:var(--green-dim);border:1px solid var(--border);font-family:var(--font-mono);font-size:0.8rem;color:var(--green);}
.health-dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 var(--green-glow);}50%{opacity:0.8;box-shadow:0 0 0 6px transparent;}}
.avatar{width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.9rem;cursor:pointer;position:relative;}
.avatar img{width:100%;height:100%;border-radius:10px;object-fit:cover;}
.avatar-dropdown{position:absolute;top:50px;right:0;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:16px;width:240px;z-index:300;box-shadow:0 12px 32px rgba(0,0,0,0.4);}
.dropdown-user-info{margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--border2);}
.dropdown-section-title{font-size:0.75rem;color:var(--muted2);margin-bottom:6px;}
.preset-avatar{cursor:pointer;transition:transform 0.2s;width:36px;height:36px;border-radius:50%;display:inline-block;}
.preset-avatar:hover{transform:scale(1.15);}
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
.stat-card{background:var(--bg2);border:1px solid var(--border2);border-radius:var(--r2);padding:22px;position:relative;overflow:hidden;transition:all 0.25s;cursor:default;}
.stat-card::before{content:'';position:absolute;top:0;right:0;width:80px;height:80px;background:radial-gradient(circle,var(--green-dim),transparent 70%);border-radius:50%;transform:translate(30px,-30px);}
.stat-card.neg::before{background:radial-gradient(circle,var(--red-dim),transparent 70%);}
.stat-card.blue-glow::before{background:radial-gradient(circle,rgba(59,139,255,0.1),transparent 70%);}
.stat-value{font-family:var(--font-mono);font-size:1.7rem;font-weight:600;margin-bottom:6px;letter-spacing:-1px;}
.stat-label{font-size:0.72rem;color:var(--muted2);text-transform:uppercase;letter-spacing:0.5px;}
.stat-sub{font-size:0.75rem;color:var(--muted);font-family:var(--font-mono);margin-top:4px;}
.income-hero{background:linear-gradient(135deg,var(--bg2) 0%,rgba(0,229,160,0.05) 100%);border:1px solid var(--border);border-radius:var(--r2);padding:32px;margin-bottom:24px;position:relative;overflow:hidden;}
.income-hero::after{content:'';position:absolute;top:-60px;right:-60px;width:200px;height:200px;border-radius:50%;background:radial-gradient(circle,var(--green-glow),transparent 70%);}
.income-label{font-size:0.8rem;color:var(--muted2);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}
.income-tool-row{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:16px;}
.tool-picker{background:var(--bg3);border:1px solid var(--border2);border-radius:10px;padding:8px 12px;font-family:var(--font-display);font-size:0.9rem;cursor:pointer;color:var(--text);}
.income-input-row{display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;}
.income-peso{font-family:var(--font-mono);font-size:2rem;font-weight:500;color:var(--green);display:flex;align-items:center;padding:0 8px;}
.income-input{flex:2;min-width:200px;font-family:var(--font-mono);font-size:1.8rem;font-weight:500;background:transparent;border:none;border-bottom:2px solid var(--border);color:var(--text);outline:none;padding:8px 4px;transition:border-color 0.2s;}
.income-input:focus{border-color:var(--green);}
.income-image-upload{display:none;margin-top:12px;}
.income-image-upload input{background:var(--bg3);padding:8px;border-radius:8px;}
.ocr-hint{font-size:0.7rem;color:var(--muted2);margin-top:4px;}
.mindset-row{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap;}
.mindset-btn{padding:8px 20px;border-radius:99px;border:1px solid var(--border2);background:transparent;color:var(--muted2);cursor:pointer;font-family:var(--font-display);font-size:0.85rem;font-weight:500;transition:all 0.2s;}
.mindset-btn.active{background:var(--green-dim);border-color:var(--green);color:var(--green);}
.btn-analyze{background:linear-gradient(135deg,var(--green),#00b87a);color:#000;border:none;border-radius:12px;padding:14px 28px;cursor:pointer;font-family:var(--font-display);font-weight:700;font-size:0.95rem;display:flex;align-items:center;gap:10px;transition:all 0.2s;flex-shrink:0;box-shadow:0 4px 20px var(--green-glow);}
.btn-analyze:hover{transform:translateY(-2px);box-shadow:0 8px 28px var(--green-glow);}
.btn-analyze:disabled{opacity:0.5;cursor:not-allowed;transform:none;}
.ai-checklist{background:var(--bg3);border-radius:12px;padding:20px;margin-top:20px;}
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
.ai-event{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid var(--border2);animation:slideIn 0.4s ease;}
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
.form-input,.form-select{background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:10px;padding:11px 14px;outline:none;font-family:var(--font-display);font-size:0.9rem;transition:border-color 0.2s;width:100%;}
.form-input:focus,.form-select:focus{border-color:var(--green);}
.form-select option{background:var(--bg2);}
.btn-add{background:linear-gradient(135deg,var(--green),#00b87a);color:#000;border:none;border-radius:10px;padding:12px 20px;cursor:pointer;font-family:var(--font-display);font-weight:700;font-size:0.9rem;transition:all 0.2s;white-space:nowrap;}
.btn-add:hover{transform:translateY(-1px);}
.btn-add:disabled{opacity:0.5;cursor:not-allowed;}
.ai-classify-result{display:inline-flex;align-items:center;gap:6px;background:var(--green-dim);border:1px solid var(--border);color:var(--green);border-radius:8px;padding:6px 12px;font-family:var(--font-mono);font-size:0.75rem;margin-top:8px;animation:fadeIn 0.3s ease;}
.tx-list{display:flex;flex-direction:column;gap:8px;}
.tx-item{display:flex;align-items:center;gap:14px;background:var(--bg3);border-radius:12px;padding:14px 16px;transition:background 0.15s;}
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
#totalWarning,#needsWantsSummary{display:none;}
.btn{background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:10px;padding:10px 16px;cursor:pointer;font-family:var(--font-display);font-weight:600;font-size:0.85rem;transition:all 0.2s;}
.btn:hover{border-color:var(--border);background:var(--bg4);}
.btn-primary{background:var(--green-dim);border-color:var(--border);color:var(--green);}
.btn-danger{background:var(--red-dim);border-color:rgba(255,59,92,0.2);color:var(--red);}
#toast{position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(80px);background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:12px 24px;font-size:0.875rem;opacity:0;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);z-index:9999;white-space:nowrap;box-shadow:0 8px 32px rgba(0,0,0,0.4);}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
.empty-state{text-align:center;padding:40px;color:var(--muted);}
.empty-state-icon{font-size:2.5rem;margin-bottom:12px;}
.empty-state-text{font-size:0.9rem;line-height:1.6;}
.auth-overlay{position:fixed;inset:0;background:rgba(6,10,16,0.97);backdrop-filter:blur(20px);z-index:9000;display:flex;align-items:center;justify-content:center;}
.auth-card{background:var(--bg2);border:1px solid var(--border);border-radius:24px;padding:40px;width:420px;max-width:90%;box-shadow:0 24px 80px rgba(0,0,0,0.5);}
.auth-logo{font-size:2rem;margin-bottom:4px;}
.auth-title{font-size:1.6rem;font-weight:800;margin-bottom:4px;}
.auth-sub{font-size:0.85rem;color:var(--muted2);margin-bottom:28px;}
.auth-input{display:block;width:100%;background:var(--bg3);color:var(--text);border:1px solid var(--border2);border-radius:12px;padding:13px 16px;outline:none;font-family:var(--font-display);font-size:0.9rem;margin-bottom:12px;transition:border-color 0.2s;}
.auth-input:focus{border-color:var(--green);}
.btn-auth{width:100%;background:linear-gradient(135deg,var(--green),#00b87a);color:#000;border:none;border-radius:12px;padding:14px;font-family:var(--font-display);font-weight:700;font-size:1rem;cursor:pointer;margin-top:8px;box-shadow:0 4px 20px var(--green-glow);transition:all 0.2s;}
.btn-auth:hover{transform:translateY(-2px);}
.auth-toggle{text-align:center;margin-top:16px;font-size:0.85rem;color:var(--muted2);cursor:pointer;}
.auth-toggle span{color:var(--green);font-weight:600;}
.auth-error{color:var(--red);font-size:0.8rem;margin-top:8px;min-height:18px;}
.auth-checkbox{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:0.85rem;color:var(--muted2);cursor:pointer;}
.chatbot{position:fixed;bottom:20px;right:20px;width:360px;height:460px;min-width:280px;min-height:300px;max-width:85vw;max-height:70vh;resize:both;overflow:auto;z-index:8000;}
.chat-window{width:100%;height:100%;background:var(--bg2);border:1px solid var(--border);border-radius:20px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 12px 48px rgba(0,0,0,0.4);}
.chat-header{padding:14px 16px;background:var(--bg3);border-bottom:1px solid var(--border2);display:flex;align-items:center;gap:10px;cursor:move;user-select:none;flex-shrink:0;}
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
.chatbot.minimized .chat-msgs,.chatbot.minimized .chat-input-row{display:none;}
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
.modal-overlay{position:fixed;inset:0;background:rgba(6,10,16,0.92);backdrop-filter:blur(16px);z-index:10000;display:flex;align-items:center;justify-content:center;}
.scenario-bar-row{display:flex;align-items:center;gap:14px;margin-bottom:14px;}
.scenario-label{font-family:var(--font-mono);font-size:0.75rem;color:var(--green);width:70px;flex-shrink:0;}
.scenario-track{flex:1;height:6px;background:var(--bg3);border-radius:99px;overflow:hidden;}
.scenario-fill{height:100%;border-radius:99px;width:0;transition:width 1.2s ease;}
.saver-fill{background:linear-gradient(90deg,var(--blue),var(--green));}
.neutral-fill{background:linear-gradient(90deg,var(--green),#00b87a);}
.spender-fill{background:linear-gradient(90deg,var(--amber),var(--red));}
.scenario-val{font-family:var(--font-mono);font-size:0.75rem;color:var(--text2);width:90px;text-align:right;flex-shrink:0;}
.budget-item{display:flex;align-items:center;gap:12px;background:var(--bg3);border-radius:10px;padding:10px 14px;margin-bottom:8px;}
.budget-cat{font-size:0.82rem;width:130px;}
.budget-input{width:80px;background:var(--bg4);border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:6px 8px;font-family:var(--font-mono);font-size:0.8rem;}
.budget-save-btn{background:var(--green-dim);border:1px solid var(--border);color:var(--green);border-radius:6px;padding:6px 10px;cursor:pointer;font-size:0.75rem;font-weight:600;}
.budget-amount{font-family:var(--font-mono);font-size:0.8rem;color:var(--muted2);margin-left:auto;}
.needwant-group{display:flex;gap:16px;align-items:center;margin-top:8px;}
.needwant-group label{font-size:0.82rem;display:flex;align-items:center;gap:4px;cursor:pointer;}
.needwant-group input[type="radio"]{accent-color:var(--green);}
/* error toast variant */
#toast.err{border-color:var(--red);color:var(--red);}
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

  <!-- DASHBOARD -->
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

      <div id="manualBlock">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;flex-wrap:wrap;">
          <span style="font-size:0.9rem;color:var(--muted2);">Transaction type:</span>
          <label style="cursor:pointer;display:flex;align-items:center;gap:6px;"><input type="radio" name="txMode" value="income" checked onchange="switchTxMode()"> Income</label>
          <label style="cursor:pointer;display:flex;align-items:center;gap:6px;"><input type="radio" name="txMode" value="expense" onchange="switchTxMode()"> Expense</label>
        </div>

        <div id="incomeFields">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <label style="font-size:0.8rem;color:var(--muted2);">Date</label>
            <input type="date" id="incomeDate" class="form-input" style="width:160px;">
          </div>
          <div style="display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;">
            <div class="income-peso">₱</div>
            <input class="income-input" id="incomeInput" type="number" placeholder="0.00" step="100" min="1">
          </div>
          <div id="incomeDateWarning" style="display:none;color:var(--red);font-size:0.75rem;margin-top:8px;">⚠️ Cannot add past-dated income</div>
        </div>

        <div id="expenseFields" style="display:none;">
          <div class="tx-form">
            <div class="form-group"><label class="form-label">Amount (₱ — min 1)</label><input class="form-input" id="expenseAmount" type="number" placeholder="0.00" step="0.01" min="1"></div>
            <div class="form-group"><label class="form-label">Category</label>
              <select class="form-select" id="expenseCategory">
                <option>Food & Dining</option><option>Transport</option><option>Groceries</option>
                <option>Entertainment</option><option>Health</option><option>Debt repayment</option>
                <option>Mortgage</option><option>Subscription</option><option>Hobbies</option><option>Other</option>
              </select>
            </div>
            <div class="form-group"><label class="form-label">Note</label><input class="form-input" id="expenseNote" placeholder="Short description"></div>
            <div class="form-group">
              <div class="needwant-group">
                <label><input type="radio" name="needwant" value="need" checked> Need</label>
                <label><input type="radio" name="needwant" value="want"> Want</label>
              </div>
            </div>
          </div>
          <div id="expenseError" style="color:var(--red);font-size:0.8rem;margin-top:8px;display:none;"></div>
        </div>

        <div class="mindset-row" id="mindsetRow" style="margin-top:20px;">
          <span style="font-size:0.78rem;color:var(--muted);align-self:center;">Spending style:</span>
          <button class="mindset-btn" data-mindset="Saver">🏦 Saver</button>
          <button class="mindset-btn active" data-mindset="Neutral">⚖️ Balanced</button>
          <button class="mindset-btn" data-mindset="Spender">🛍️ Spender</button>
        </div>

        <div class="ai-checklist">
          <div style="font-weight:600;margin-bottom:12px;">🧠 ML Allocation Checklist — select categories to plan</div>
          <div class="checklist-section">
            <div class="checklist-section-title">Needs / Savings</div>
            <div id="needsChecklist"></div>
          </div>
          <div class="checklist-section">
            <div class="checklist-section-title">Wants</div>
            <div id="wantsChecklist"></div>
          </div>
          <div id="totalWarning" class="total-warning" style="display:none;"></div>
          <div style="margin-top:12px;">
            <button class="btn-analyze" id="analyzeBtn"><span id="analyzeBtnContent">🤖 Add &amp; Run ML Plan</span></button>
          </div>
        </div>
      </div>

      <div id="incomeImageUpload" class="income-image-upload" style="display:none;">
        <input type="file" id="incomeImage" accept="image/*" capture="environment">
        <div class="ocr-hint">📸 Upload a payslip or receipt — ML will extract all income & expenses.</div>
        <div id="ocrModal" class="modal-overlay" style="display:none;">
          <div class="auth-card" style="width:600px;max-height:80vh;overflow-y:auto;">
            <div class="auth-logo">📄</div>
            <div style="font-weight:600;margin-bottom:16px;">Review extracted items — edit type/amount, then save</div>
            <div id="ocrItemsList"></div>
            <div style="display:flex;gap:12px;margin-top:16px;">
              <button class="btn-add" id="ocrSaveBtn">Save All</button>
              <button class="btn" id="ocrCancelBtn">Cancel</button>
            </div>
          </div>
        </div>
      </div>

      <div id="profileBlock" style="display:none;">
        <div style="font-size:1rem;font-weight:600;margin-bottom:16px;color:var(--green);">Edit Profile</div>
        <div style="display:flex;flex-direction:column;gap:12px;max-width:400px;">
          <div class="form-group"><label class="form-label">Name</label><input class="form-input" id="profileName"></div>
          <div class="form-group"><label class="form-label">Email</label><input class="form-input" id="profileEmail" type="email"></div>
          <div class="form-group"><label class="form-label">New Password (blank = keep current)</label><input class="form-input" id="profilePass" type="password" placeholder="••••••"></div>
          <div class="form-group"><label class="form-label">Avatar URL</label><input class="form-input" id="profileAvatar" placeholder="https://…">
            <label class="btn" style="display:inline-block;width:fit-content;cursor:pointer;font-size:0.8rem;padding:6px 14px;margin-top:6px;">📁 Upload <input type="file" id="profileFileInput" accept="image/*" style="display:none;"></label>
          </div>
          <button class="btn-add" id="saveProfileBtn">Save Changes</button>
        </div>
      </div>
    </div>

    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value" id="sBalance">—</div><div class="stat-label">Balance</div></div>
      <div class="stat-card neg"><div class="stat-value" id="sExpense" style="color:var(--red)">—</div><div class="stat-label">Month Expenses</div></div>
      <div class="stat-card"><div class="stat-value" id="sIncome" style="color:var(--green)">—</div><div class="stat-label">Month Income</div></div>
      <div class="stat-card blue-glow"><div class="stat-value" id="sScore" style="color:var(--blue)">—</div><div class="stat-label">Health Score</div><div class="stat-sub" id="scoreLabel">awaiting data</div></div>
    </div>

    <div class="card ai-feed">
      <div class="ai-feed-header"><div class="ai-pulse"></div><div class="ai-feed-title">ML Activity Feed</div><div class="ai-badge" style="margin-left:auto;">SMART</div></div>
      <div id="aiFeed"><div class="empty-state"><div class="empty-state-icon">🤖</div><div class="empty-state-text">Enter income or an expense above and click<br><strong style="color:var(--green)">Add & Run ML Plan</strong></div></div></div>
    </div>

    <div id="showPlanToggle" style="display:none;margin-bottom:16px;">
      <button class="btn btn-primary" id="togglePlanBtn">📊 Show Plan Details</button>
    </div>

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

    <div id="allocationBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">ML Budget Allocation</span><span class="ai-badge">100% Autonomous</span></div>
      <div class="alloc-grid" id="allocGrid"></div>
    </div>

    <div id="budgetBlock" style="display:none" class="card">
      <div class="card-header">
        <span class="card-title">Manage Budgets</span>
        <button class="btn btn-primary" id="resetBudgetsBtn" style="font-size:0.8rem;">⟳ Reset to AI Budgets</button>
      </div>
      <div id="budgetList"></div>
    </div>

    <div id="adviceBlock" style="display:none" class="card">
      <div class="card-header"><span class="card-title">ML Insights</span></div>
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

  <!-- FUTURE EXPENSES -->
  <div class="screen" id="screen-future">
    <div class="card">
      <div class="card-header"><span class="card-title">Pin Future Expense</span></div>
      <div class="tx-form">
        <div class="form-group" style="flex:2;min-width:200px;"><label class="form-label">Description</label><input class="form-input" id="futureDesc" placeholder="e.g. Rent"></div>
        <div class="form-group"><label class="form-label">Amount (₱ — min 1)</label><input class="form-input" id="futureAmt" type="number" min="1" step="0.01"></div>
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

  <!-- INSIGHTS -->
  <div class="screen" id="screen-insights">
    <div class="card">
      <div class="card-header"><span class="card-title">Budget Longevity (3 Scenarios)</span></div>
      <div id="scenarioBars" style="margin-bottom:20px;"></div>
      <div class="longevity-display">
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

  <!-- HISTORY -->
  <div class="screen" id="screen-history">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Transaction History</span>
        <button class="btn btn-primary" id="toggleHistoryViewBtn" style="font-size:0.8rem;padding:8px 14px;">🙈 Hide</button>
      </div>
      <div id="historyContent">
        <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">
          <input class="form-input" id="historySearch" placeholder="Search…" style="width:180px;padding:8px 12px;font-size:0.82rem;">
          <label style="font-size:0.78rem;color:var(--muted2);display:flex;align-items:center;gap:6px;"><input type="checkbox" id="showNeedWantBadges" checked> Show Need/Want badges</label>
        </div>
        <div class="tx-list" id="historyList"></div>
      </div>
    </div>
  </div>

  <!-- ADMIN -->
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
        <table style="width:100%;border-collapse:collapse;"><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Role</th><th>Created</th><th>Actions</th></tr></thead><tbody id="adminUserTable"></tbody></table>
      </div>
      <div id="adminTransactionsPanel" style="display:none;">
        <select id="adminUserFilter" class="form-select" style="margin-bottom:12px;"><option value="">All Users</option></select>
        <table style="width:100%;border-collapse:collapse;"><thead><tr><th>Date</th><th>User</th><th>Category</th><th>Type</th><th>Amount</th><th>Need/Want</th></tr></thead><tbody id="adminTxTable"></tbody></table>
      </div>
    </div>
  </div>
</main>

<div id="toast"></div>

<!-- AUTH -->
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

<!-- SIGN OUT MODAL -->
<div id="signoutModal" class="modal-overlay" style="display:none;">
  <div class="auth-card" style="width:320px;padding:28px;">
    <div style="text-align:center;margin-bottom:20px;font-size:1rem;font-weight:600;">Sign out?</div>
    <button class="btn-auth" id="confirmLogoutBtn" style="margin-bottom:10px;">Sign Out</button>
    <button class="btn-auth" id="cancelSignoutBtn" style="background:transparent;color:var(--muted2);box-shadow:none;border:1px solid var(--border2);">Cancel</button>
  </div>
</div>

<!-- AVATAR DROPDOWN -->
<div id="avatarDropdown" class="avatar-dropdown" style="display:none;">
  <div class="dropdown-user-info">
    <div id="dropdownUserName" style="font-weight:600;"></div>
    <div id="dropdownUserEmail" style="font-size:0.8rem;color:var(--muted2);"></div>
  </div>
  <div class="dropdown-section-title">Change Avatar</div>
  <input type="text" id="avatarUrlInput" class="form-input" placeholder="Paste image URL…" style="width:100%;margin-bottom:8px;">
  <button class="btn btn-primary" id="saveAvatarBtn" style="width:100%;">Save Avatar</button>
</div>

<!-- CHATBOT -->
<div class="chatbot" id="chatbot">
  <div class="chat-window">
    <div class="chat-header" id="chatHeader"><div class="ai-pulse"></div><div class="chat-header-title">SmartSpend ML <span class="ai-badge">Smart</span></div><button class="chat-toggle-btn" id="chatToggleBtn">–</button></div>
    <div class="chat-msgs" id="chatMsgs"><div class="msg bot">👋 I'm your ML finance assistant. Ask me anything about your money or budget.</div></div>
    <div class="chat-input-row"><input class="chat-inp" id="chatInp" placeholder="Ask anything…"><button class="chat-send" id="chatSend">→</button></div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════
let currentUser=null, allTransactions=[], currentMindset='Neutral', aiPlan=null;
let trendChart=null, catChartInst=null, isLogin=true, historyVisible=true;

const CAT_ICONS={'Food & Dining':'🍜','Transport':'🚗','Groceries':'🛒','Entertainment':'🎬','Health':'💊','Debt repayment':'💳','Mortgage':'🏠','Subscription':'📱','Hobbies':'🎮','Salary':'💰','Savings':'🏦','Other':'📦'};
const NEED_CATS=['Food & Dining','Debt repayment','Mortgage','Transport','Groceries','Health','Savings'];
const WANT_CATS=['Entertainment','Subscription','Hobbies'];
const ALL_CATS=[...NEED_CATS,...WANT_CATS];

function fmt(n){return '₱'+Number(n||0).toLocaleString('en-PH',{minimumFractionDigits:2,maximumFractionDigits:2});}
function fmtDate(iso){return new Date(iso).toLocaleDateString('en-PH',{month:'short',day:'numeric',year:'2-digit'});}
function esc(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function toast(msg,isErr=false){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.classList.toggle('err',isErr);
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

async function api(url,opts={}){
  const res=await fetch(url,{...opts,credentials:'include',headers:{'Content-Type':'application/json',...(opts.headers||{})}});
  if(!res.ok){const e=await res.json().catch(()=>({error:'Request failed'}));throw new Error(e.error||'Request failed');}
  return res.json();
}

// ═══════════════════════════════════════════════
// CHECKLIST
// ═══════════════════════════════════════════════
function renderChecklist(){
  const nc=document.getElementById('needsChecklist'),wc=document.getElementById('wantsChecklist');
  nc.innerHTML=''; wc.innerHTML='';
  ALL_CATS.forEach(cat=>{
    const isNeed=NEED_CATS.includes(cat);
    const row=document.createElement('div');
    row.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px;';
    row.innerHTML=`<input type="checkbox" class="cat-checkbox" data-cat="${esc(cat)}" checked style="accent-color:var(--green);width:15px;height:15px;"><label style="font-size:0.82rem;color:var(--text2);cursor:pointer;">${esc(cat)}</label><span class="cat-percent" id="pct-${cat.replace(/\s/g,'')}" style="font-family:var(--font-mono);font-size:0.78rem;color:var(--muted2);margin-left:auto;"></span>`;
    (isNeed?nc:wc).appendChild(row);
  });
}

function updateChecklistPct(allocation){
  ALL_CATS.forEach(cat=>{
    const span=document.getElementById('pct-'+cat.replace(/\s/g,''));
    if(span) span.textContent=allocation[cat]!=null?`${allocation[cat].toFixed(1)}%`:'';
  });
}

// ═══════════════════════════════════════════════
// TX MODE SWITCH
// ═══════════════════════════════════════════════
function switchTxMode(){
  const mode=document.querySelector('input[name="txMode"]:checked').value;
  document.getElementById('incomeFields').style.display=mode==='income'?'block':'none';
  document.getElementById('expenseFields').style.display=mode==='expense'?'block':'none';
  document.getElementById('mindsetRow').style.display=mode==='income'?'flex':'none';
  document.getElementById('expenseError').style.display='none';
}

// ═══════════════════════════════════════════════
// DATE VALIDATION
// ═══════════════════════════════════════════════
document.getElementById('incomeDate').addEventListener('change',function(){
  const today=new Date().toISOString().slice(0,10);
  document.getElementById('incomeDateWarning').style.display=this.value&&this.value<today?'block':'none';
});

document.getElementById('incomeInput').addEventListener('input',function(){
  const v=parseFloat(this.value)||0;
  const m=v<=10000?'Saver':v<=50000?'Neutral':'Spender';
  document.querySelectorAll('.mindset-btn').forEach(b=>b.classList.toggle('active',b.dataset.mindset===m));
  currentMindset=m;
});

document.querySelectorAll('.mindset-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.mindset-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active'); currentMindset=btn.dataset.mindset;
  });
});

// ═══════════════════════════════════════════════
// AUTH
// ═══════════════════════════════════════════════
const authOverlay=document.getElementById('authOverlay');
const authMsg=document.getElementById('authMsg');

document.getElementById('toggleAuth').addEventListener('click',()=>{
  isLogin=!isLogin;
  if(isLogin){
    document.getElementById('authTitle').textContent='Welcome back';
    document.getElementById('authSub').textContent='Sign in to your SmartSpend account';
    document.getElementById('authBtn').textContent='Sign In';
    document.getElementById('regName').style.display='none';
    document.getElementById('authConfirm').style.display='none';
    document.getElementById('termsRow').style.display='none';
    document.getElementById('toggleAuth').innerHTML='No account? <span>Register here</span>';
  }else{
    document.getElementById('authTitle').textContent='Create account';
    document.getElementById('authSub').textContent='Start your ML‑powered journey';
    document.getElementById('authBtn').textContent='Register';
    document.getElementById('regName').style.display='block';
    document.getElementById('authConfirm').style.display='block';
    document.getElementById('termsRow').style.display='flex';
    document.getElementById('toggleAuth').innerHTML='Already registered? <span>Sign in</span>';
  }
  authMsg.textContent='';
});

document.getElementById('authBtn').addEventListener('click',async()=>{
  authMsg.textContent='';
  const email=document.getElementById('authEmail').value.trim();
  const pw=document.getElementById('authPass').value;
  if(!isLogin){
    const name=document.getElementById('regName').value.trim();
    const confirm=document.getElementById('authConfirm').value;
    const terms=document.getElementById('termsCheck').checked;
    if(!name){authMsg.textContent='Name required';return;}
    if(pw!==confirm){authMsg.textContent='Passwords do not match';return;}
    if(!terms){authMsg.textContent='Accept the terms';return;}
  }
  try{
    const body=isLogin?{email,password:pw}:{name:document.getElementById('regName').value.trim(),email,password:pw};
    const data=await api(isLogin?'/api/login':'/api/register',{method:'POST',body:JSON.stringify(body)});
    currentUser=data; authOverlay.style.display='none'; initApp();
  }catch(e){authMsg.textContent=e.message;}
});

// ═══════════════════════════════════════════════
// SIGN OUT
// ═══════════════════════════════════════════════
document.getElementById('signoutBtn').addEventListener('click',()=>{document.getElementById('signoutModal').style.display='flex';});
document.getElementById('confirmLogoutBtn').addEventListener('click',async()=>{
  await api('/api/logout',{method:'POST'}); currentUser=null;
  document.getElementById('signoutModal').style.display='none';
  authOverlay.style.display='flex';
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('screen-dashboard').classList.add('active');
  toast('Signed out');
});
document.getElementById('cancelSignoutBtn').addEventListener('click',()=>{document.getElementById('signoutModal').style.display='none';});

// ═══════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════
document.querySelectorAll('.nav-item[data-nav]').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const nav=btn.dataset.nav;
    document.querySelectorAll('.nav-item').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
    document.getElementById('screen-'+nav).classList.add('active');
    document.getElementById('pageTitle').textContent=btn.querySelector('.nav-label').textContent;
    if(nav==='dashboard')loadDashboard();
    else if(nav==='insights')loadInsights();
    else if(nav==='future')loadFutureExpenses();
    else if(nav==='history')loadHistory();
    else if(nav==='admin')loadAdmin();
  });
});

// ═══════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════
async function initApp(){
  if(!currentUser)return;
  const av=document.getElementById('userAvatar');
  const initial=av.querySelector('.avatar-initial');
  const img=av.querySelector('img');
  if(currentUser.avatar_url){img.src=currentUser.avatar_url;img.style.display='block';initial.style.display='none';}
  else{img.style.display='none';initial.style.display='block';initial.textContent=currentUser.name?.charAt(0)?.toUpperCase()||'?';}
  document.getElementById('adminNavBtn').style.display=currentUser.role==='admin'?'flex':'none';
  document.getElementById('incomeDate').value=new Date().toISOString().slice(0,10);
  renderChecklist();
  loadDashboard();
}

// ═══════════════════════════════════════════════
// AVATAR DROPDOWN
// ═══════════════════════════════════════════════
(function(){
  const av=document.getElementById('userAvatar'),dd=document.getElementById('avatarDropdown');
  let vis=false;
  av.addEventListener('click',e=>{
    e.stopPropagation(); vis=!vis; dd.style.display=vis?'block':'none';
    if(vis&&currentUser){document.getElementById('dropdownUserName').textContent=currentUser.name;document.getElementById('dropdownUserEmail').textContent=currentUser.email;document.getElementById('avatarUrlInput').value=currentUser.avatar_url||'';}
  });
  document.addEventListener('click',()=>{vis=false;dd.style.display='none';});
})();

document.getElementById('saveAvatarBtn').addEventListener('click',async()=>{
  const url=document.getElementById('avatarUrlInput').value.trim();
  try{const r=await api('/api/me/avatar',{method:'PUT',body:JSON.stringify({avatar_url:url})});currentUser.avatar_url=r.avatar_url;
    const av=document.getElementById('userAvatar'),initial=av.querySelector('.avatar-initial'),img=av.querySelector('img');
    if(r.avatar_url){img.src=r.avatar_url;img.style.display='block';initial.style.display='none';}
    else{img.style.display='none';initial.style.display='block';initial.textContent=currentUser.name?.charAt(0)?.toUpperCase()||'?';}
    toast('Avatar updated');
  }catch(e){toast(e.message,true);}
  document.getElementById('avatarDropdown').style.display='none';
});

// ═══════════════════════════════════════════════
// TOOL PICKER
// ═══════════════════════════════════════════════
document.getElementById('incomeTool').addEventListener('change',function(){
  const v=this.value;
  document.getElementById('manualBlock').style.display=v==='manual'?'block':'none';
  document.getElementById('incomeImageUpload').style.display=v==='auto'?'block':'none';
  document.getElementById('profileBlock').style.display=v==='profile'?'block':'none';
  if(v==='profile'&&currentUser){
    document.getElementById('profileName').value=currentUser.name||'';
    document.getElementById('profileEmail').value=currentUser.email||'';
    document.getElementById('profilePass').value='';
    document.getElementById('profileAvatar').value=currentUser.avatar_url||'';
  }
});

// ═══════════════════════════════════════════════
// MAIN ACTION BUTTON — Add & Run ML Plan
// ═══════════════════════════════════════════════
document.getElementById('analyzeBtn').addEventListener('click',async()=>{
  const mode=document.querySelector('input[name="txMode"]:checked').value;
  const errEl=document.getElementById('expenseError');
  errEl.style.display='none';

  // ── Income mode ────────────────────────────────────────────────────────
  if(mode==='income'){
    const amount=parseFloat(document.getElementById('incomeInput').value);
    const date=document.getElementById('incomeDate').value;
    const today=new Date().toISOString().slice(0,10);
    if(!amount||amount<1){toast('Income amount must be at least ₱1',true);return;}
    if(!date){toast('Please select a date',true);return;}
    if(date<today){toast('Cannot add past-dated income',true);return;}
    const btn=document.getElementById('analyzeBtn');
    const bc=document.getElementById('analyzeBtnContent');
    btn.disabled=true; bc.innerHTML='<div class="spinner"></div> Saving & Planning…';
    try{
      await api('/api/transactions',{method:'POST',body:JSON.stringify({amount,category:'Salary',tx_type:'income',is_need:true,note:'Manual income',tx_date:date})});
      addFeedEvent('💰',`Income ₱${amount.toLocaleString()} added`);
      await runMLPlan();
    }catch(e){toast(e.message,true);addFeedEvent('❌',e.message);}
    btn.disabled=false; bc.innerHTML='🤖 Add &amp; Run ML Plan';
    return;
  }

  // ── Expense mode ───────────────────────────────────────────────────────
  if(mode==='expense'){
    const amount=parseFloat(document.getElementById('expenseAmount').value);
    const category=document.getElementById('expenseCategory').value;
    const note=document.getElementById('expenseNote').value;
    const isNeed=document.querySelector('input[name="needwant"]:checked').value==='need';
    errEl.style.display='none';

    if(!amount||amount<1){errEl.textContent='Amount must be at least ₱1';errEl.style.display='block';return;}

    // Budget zero check (client-side pre-flight)
    const btn=document.getElementById('analyzeBtn');
    const bc=document.getElementById('analyzeBtnContent');
    btn.disabled=true; bc.innerHTML='<div class="spinner"></div> Saving…';
    try{
      await api('/api/transactions',{method:'POST',body:JSON.stringify({amount,category,tx_type:'expense',is_need:isNeed,note})});
      addFeedEvent('💸',`Expense ₱${amount.toLocaleString()} (${category}) added`);
      document.getElementById('expenseAmount').value='';
      document.getElementById('expenseNote').value='';
      await runMLPlan();
    }catch(e){errEl.textContent=e.message;errEl.style.display='block';toast(e.message,true);addFeedEvent('❌',e.message);}
    btn.disabled=false; bc.innerHTML='🤖 Add &amp; Run ML Plan';
  }
});

async function runMLPlan(){
  const summary=await api(`/api/summary/${currentUser.id}`);
  const totalIncome=summary.income||0;
  document.getElementById('incomeInput').value=totalIncome>0?totalIncome:'';
  if(totalIncome<=0){loadDashboard();return;}
  const selectedCats=[];
  document.querySelectorAll('.cat-checkbox:checked').forEach(c=>selectedCats.push(c.dataset.cat));
  if(!selectedCats.length) ALL_CATS.forEach(c=>selectedCats.push(c));
  addFeedEvent('🤖','ML is building your financial plan…');
  const result=await api('/api/ai/full_setup',{method:'POST',body:JSON.stringify({monthly_income:totalIncome,mindset:currentMindset,selected_categories:selectedCats})});
  aiPlan=result;
  if(result.allocation){
    updateChecklistPct(result.allocation);
    addFeedEvent('✅',`Allocated across ${Object.keys(result.allocation).length} categories`);
    addFeedEvent('💰',`Savings target: ${fmt(result.savings_plan.monthly)}/month`);
    addFeedEvent('🧠',result.financial_summary.substring(0,80)+'…');
    addFeedEvent('📋',`${result.advice.length} insights ready`);
    renderAIPlan(result);
  }
  loadDashboard();
}

// ═══════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════
async function loadDashboard(){
  if(!currentUser)return;
  try{
    await api('/api/apply_future_expenses',{method:'POST'});
    const[summary,predict]=await Promise.all([api(`/api/summary/${currentUser.id}`),api(`/api/predict/${currentUser.id}`)]);
    document.getElementById('sBalance').textContent=fmt(summary.balance);
    document.getElementById('sExpense').textContent=fmt(summary.expense);
    document.getElementById('sIncome').textContent=fmt(summary.income);
    document.getElementById('sScore').textContent=predict.score??'—';
    document.getElementById('scoreLabel').textContent=predict.score?'Health Score':'awaiting data';
    document.getElementById('topScore').textContent=predict.score??'—';
    renderForecast(predict.predictions?.weekly);
    renderTrendChart(summary.monthly);
    document.getElementById('chartBlock').style.display=Object.keys(summary.monthly).length?'block':'none';
    document.getElementById('forecastBlock').style.display=Object.keys(predict.predictions?.weekly||{}).length?'block':'none';
    // Auto-run plan if income exists and no plan yet
    if(summary.income>0&&!aiPlan) runMLPlan();
  }catch(e){toast(e.message,true);}
}

function addFeedEvent(icon,text){
  const feed=document.getElementById('aiFeed');
  const empty=feed.querySelector('.empty-state');if(empty)empty.remove();
  const time=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
  const div=document.createElement('div'); div.className='ai-event';
  div.innerHTML=`<span class="ai-event-icon">${icon}</span><span class="ai-event-text">${esc(text)}</span><span class="ai-event-time">${time}</span>`;
  feed.prepend(div);
}

// ═══════════════════════════════════════════════
// RENDER AI PLAN
// ═══════════════════════════════════════════════
function renderAIPlan(plan){
  document.getElementById('financialSummaryText').textContent=plan.financial_summary;
  document.getElementById('saveDaily').textContent=fmt(plan.savings_plan.daily);
  document.getElementById('saveWeekly').textContent=fmt(plan.savings_plan.weekly);
  document.getElementById('saveMonthly').textContent=fmt(plan.savings_plan.monthly);
  document.getElementById('savingsTip').textContent='💡 '+(plan.savings_plan.tip||'');

  const needs=new Set(['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage']);
  const savings=new Set(['Savings']);
  document.getElementById('allocGrid').innerHTML=Object.entries(plan.allocation).map(([cat,pct])=>{
    const type=savings.has(cat)?'savings':needs.has(cat)?'need':'want';
    const amt=plan.allocation_amounts?.[cat]||(plan.monthly_income*pct/100);
    return `<div class="alloc-item"><div class="alloc-cat">${esc(cat)}</div><div class="alloc-pct">${pct.toFixed(0)}<span style="font-size:1rem;color:var(--muted)">%</span></div><div class="alloc-amount">${fmt(amt)}</div><div class="alloc-type ${type}">${type.toUpperCase()}</div><div class="alloc-item-bar ${type}" style="width:${Math.min(pct,100)}%"></div></div>`;
  }).join('');

  const icons={info:'ℹ️',warning:'⚠️',success:'✅'};
  document.getElementById('adviceList').innerHTML=plan.advice.map(a=>
    `<div class="advice-card ${esc(a.type)}"><span class="advice-icon">${icons[a.type]||'💡'}</span><div><div class="advice-title">${esc(a.title)}</div><div class="advice-body">${esc(a.body)}</div></div></div>`).join('');

  ['financialSummaryBlock','allocationBlock','adviceBlock'].forEach(id=>document.getElementById(id).style.display='block');
  const tog=document.getElementById('showPlanToggle'),togBtn=document.getElementById('togglePlanBtn');
  tog.style.display='block'; let vis=true;
  togBtn.textContent='🔽 Hide Plan Details';
  togBtn.onclick=()=>{vis=!vis;['financialSummaryBlock','allocationBlock','adviceBlock'].forEach(id=>document.getElementById(id).style.display=vis?'block':'none');togBtn.textContent=vis?'🔽 Hide Plan Details':'📊 Show Plan Details';};
  loadBudgets();
}

// ═══════════════════════════════════════════════
// BUDGETS
// ═══════════════════════════════════════════════
async function loadBudgets(){
  if(!currentUser)return;
  try{const b=await api(`/api/budgets/${currentUser.id}`);renderBudgetList(b);}catch(e){toast(e.message,true);}
}
function renderBudgetList(budgets){
  const c=document.getElementById('budgetList');
  if(!budgets.length){c.innerHTML='<div class="empty-state"><div class="empty-state-icon">📊</div><div class="empty-state-text">No budgets yet. Run ML Plan first.</div></div>';return;}
  c.innerHTML=budgets.map(b=>{
    const sid=b.category.replace(/[^a-zA-Z0-9]/g,'_');
    const esc_cat=b.category.replace(/'/g,"\\'");
    return `<div class="budget-item"><div class="budget-cat">${esc(b.category)}</div><input class="budget-input" id="bi-${sid}" type="number" step="0.01" min="0" value="${b.limit.toFixed(2)}"><button class="budget-save-btn" onclick="saveBudget('${esc_cat}','${sid}')">Save</button><span class="budget-amount">${fmt(b.limit)}</span></div>`;
  }).join('');
  document.getElementById('budgetBlock').style.display='block';
}
async function saveBudget(cat,sid){
  const v=parseFloat(document.getElementById('bi-'+sid)?.value);
  if(isNaN(v))return;
  try{await api(`/api/budgets/${currentUser.id}/${encodeURIComponent(cat)}`,{method:'PUT',body:JSON.stringify({limit:v})});toast('Budget updated');loadBudgets();}catch(e){toast(e.message,true);}
}
document.getElementById('resetBudgetsBtn')?.addEventListener('click',async()=>{
  try{await api(`/api/budgets/reset_to_ai/${currentUser.id}`,{method:'POST'});toast('Budgets reset');loadBudgets();}catch(e){toast(e.message,true);}
});

// ═══════════════════════════════════════════════
// CHARTS
// ═══════════════════════════════════════════════
function renderForecast(weekly){
  const c=document.getElementById('forecastBars'); if(!weekly){c.innerHTML='';return;}
  const mx=Math.max(...Object.values(weekly),1);
  c.innerHTML=Object.entries(weekly).map(([w,v])=>`<div class="forecast-bar-row"><span class="forecast-week-label">${w}</span><div class="forecast-track"><div class="forecast-fill" style="width:${(v/mx*100).toFixed(0)}%"></div></div><span class="forecast-val">${fmt(v)}</span></div>`).join('');
}
function renderTrendChart(monthly){
  const ctx=document.getElementById('trendChart'); if(!ctx)return;
  if(trendChart)trendChart.destroy();
  const labels=Object.keys(monthly);
  trendChart=new Chart(ctx,{type:'line',data:{labels,datasets:[{label:'Income',data:labels.map(m=>monthly[m].income),borderColor:'#00E5A0',backgroundColor:'rgba(0,229,160,0.1)',tension:0.3},{label:'Expense',data:labels.map(m=>monthly[m].expense),borderColor:'#FF3B5C',backgroundColor:'rgba(255,59,92,0.1)',tension:0.3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#B0C8E0'}}}}});
}

// ═══════════════════════════════════════════════
// OCR
// ═══════════════════════════════════════════════
document.getElementById('incomeImage').addEventListener('change',async function(){
  const file=this.files[0]; if(!file)return;
  const fd=new FormData(); fd.append('image',file);
  try{
    const r=await fetch('/api/ocr_income',{method:'POST',body:fd,credentials:'include'});
    const d=await r.json();
    if(d.transactions&&d.transactions.length>0){showOCRModal(d.transactions);}
    else{toast('No transactions found in image',true);}
  }catch(e){toast('OCR failed: '+e.message,true);}
});

function showOCRModal(items){
  window.ocrItems=items.map(i=>({...i,_del:false}));
  const list=document.getElementById('ocrItemsList');
  list.innerHTML='';
  items.forEach((item,idx)=>{
    const row=document.createElement('div');
    row.className='budget-item';
    row.innerHTML=`<span style="flex:1;font-size:0.82rem;">${esc(item.category)} – ₱${item.amount} – ${esc(item.note||'')}</span><select class="form-select" style="width:100px;" onchange="window.ocrItems[${idx}].type=this.value"><option value="income" ${item.type==='income'?'selected':''}>Income</option><option value="expense" ${item.type==='expense'?'selected':''}>Expense</option></select><button class="btn-del" onclick="this.parentElement.style.display='none';window.ocrItems[${idx}]._del=true;">🗑</button>`;
    list.appendChild(row);
  });
  document.getElementById('ocrSaveBtn').onclick=async()=>{
    for(const item of window.ocrItems){
      if(item._del)continue;
      try{await api('/api/transactions',{method:'POST',body:JSON.stringify({amount:Math.abs(item.amount),category:item.category,tx_type:item.type,is_need:['Food & Dining','Transport','Groceries','Health','Debt repayment','Mortgage'].includes(item.category),note:item.note||''})});}
      catch(e){toast(e.message,true);}
    }
    toast('Transactions saved!'); document.getElementById('ocrModal').style.display='none'; loadDashboard();
  };
  document.getElementById('ocrCancelBtn').onclick=()=>document.getElementById('ocrModal').style.display='none';
  document.getElementById('ocrModal').style.display='flex';
}

// ═══════════════════════════════════════════════
// FUTURE EXPENSES
// ═══════════════════════════════════════════════
async function loadFutureExpenses(){if(!currentUser)return;try{renderFutureList(await api('/api/future_expenses'));}catch(e){toast(e.message,true);}}
function renderFutureList(exps){
  const list=document.getElementById('futureList');
  if(!exps.length){list.innerHTML='<div class="empty-state"><div class="empty-state-icon">📌</div><div class="empty-state-text">No pinned expenses yet.</div></div>';return;}
  list.innerHTML=exps.map(e=>`<div class="future-item"><div class="future-info"><div class="future-desc">${esc(e.description)}</div><div class="future-meta">${esc(e.category)} · ${e.cycle} · ${e.date}</div></div><div class="future-amount">${fmt(e.amount)}</div><button class="btn-del" onclick="deleteFuture(${e.id})">🗑</button></div>`).join('');
}
document.getElementById('pinFutureBtn').addEventListener('click',async()=>{
  const desc=document.getElementById('futureDesc').value;
  const amount=parseFloat(document.getElementById('futureAmt').value);
  const date=document.getElementById('futureDate').value;
  if(!desc||!amount||!date){toast('Please fill all fields',true);return;}
  if(amount<1){toast('Amount must be at least ₱1',true);return;}
  try{await api('/api/future_expenses',{method:'POST',body:JSON.stringify({description:desc,amount,category:document.getElementById('futureCat').value,cycle:document.getElementById('futureCycle').value,date})});toast('Pinned');document.getElementById('futureDesc').value='';document.getElementById('futureAmt').value='';loadFutureExpenses();}catch(e){toast(e.message,true);}
});
async function deleteFuture(id){if(!confirm('Remove?'))return;try{await api(`/api/future_expenses/${id}`,{method:'DELETE'});toast('Removed');loadFutureExpenses();}catch(e){toast(e.message,true);}}
document.getElementById('applyFutureBtn').addEventListener('click',async()=>{try{const r=await api('/api/apply_future_expenses',{method:'POST'});toast(`Processed ${r.count} pending expenses`);loadFutureExpenses();}catch(e){toast(e.message,true);}});

// ═══════════════════════════════════════════════
// INSIGHTS
// ═══════════════════════════════════════════════
async function loadInsights(){
  if(!currentUser)return;
  try{
    const lng=await api(`/api/longevity/${currentUser.id}`);
    const bal=lng.balance, avg=lng.avg_daily_spend;
    const scenarios=[{label:'🏦 Saver',daily:avg*0.8,css:'saver-fill'},{label:'⚖️ Balanced',daily:avg,css:'neutral-fill'},{label:'🛍️ Spender',daily:avg*1.2,css:'spender-fill'}];
    const maxD=Math.max(...scenarios.map(s=>bal/s.daily),1);
    document.getElementById('scenarioBars').innerHTML=scenarios.map(s=>{const d=Math.floor(bal/s.daily);return `<div class="scenario-bar-row"><span class="scenario-label">${s.label}</span><div class="scenario-track"><div class="scenario-fill ${s.css}" style="width:${Math.min((d/maxD)*100,100)}%"></div></div><span class="scenario-val">${d} days</span></div>`;}).join('');
    document.getElementById('longevityDays').textContent=avg>0?Math.floor(bal/avg):'—';
    document.getElementById('longevityBalance').textContent=fmt(bal);
    document.getElementById('longevityDaily').textContent=fmt(avg);
    const predict=await api(`/api/predict/${currentUser.id}`);
    // Forecast insights
    const fi=document.getElementById('forecastBarsInsights'); const fw=predict.predictions?.weekly||{};
    const fmax=Math.max(...Object.values(fw),1);
    fi.innerHTML=Object.entries(fw).map(([w,v])=>`<div class="forecast-bar-row"><span class="forecast-week-label">${w}</span><div class="forecast-track"><div class="forecast-fill" style="width:${(v/fmax*100).toFixed(0)}%"></div></div><span class="forecast-val">${fmt(v)}</span></div>`).join('');
    // Category chart
    const ctx=document.getElementById('catChart');
    if(catChartInst)catChartInst.destroy();
    const cats=predict.predictions?.categories||{};
    catChartInst=new Chart(ctx,{type:'doughnut',data:{labels:Object.keys(cats),datasets:[{data:Object.values(cats),backgroundColor:['#00E5A0','#3B8BFF','#F5A623','#FF3B5C','#9B59F5','#00b87a','#FF8C00']}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{labels:{color:'#B0C8E0'}}}}});
  }catch(e){toast(e.message,true);}
}

// ═══════════════════════════════════════════════
// HISTORY
// ═══════════════════════════════════════════════
async function loadHistory(){if(!currentUser)return;try{const txs=await api('/api/transactions');allTransactions=txs;renderHistory(txs);}catch(e){toast(e.message,true);}}
function renderHistory(txs){
  const list=document.getElementById('historyList');
  const q=document.getElementById('historySearch').value.toLowerCase();
  const badges=document.getElementById('showNeedWantBadges').checked;
  const f=txs.filter(t=>t.category.toLowerCase().includes(q)||(t.note||'').toLowerCase().includes(q));
  if(!f.length){list.innerHTML='<div class="empty-state"><div class="empty-state-icon">🔍</div><div class="empty-state-text">No matching transactions.</div></div>';return;}
  list.innerHTML=f.map(t=>`<div class="tx-item"><div class="tx-cat-icon">${CAT_ICONS[t.category]||'📦'}</div><div class="tx-info"><div class="tx-cat">${esc(t.category)}${badges?(t.is_need?'<span class="tx-badge need">Need</span>':'<span class="tx-badge want">Want</span>'):''}</div><div class="tx-meta">${t.note?esc(t.note)+' · ':''}${fmtDate(t.tx_date)}</div></div><div class="tx-amount ${t.tx_type}">${t.tx_type==='income'?'+':'-'}${fmt(t.amount)}</div><button class="btn-del" onclick="deleteTx(${t.id})">🗑</button></div>`).join('');
}
document.getElementById('historySearch').addEventListener('input',()=>renderHistory(allTransactions));
document.getElementById('showNeedWantBadges').addEventListener('change',()=>renderHistory(allTransactions));
document.getElementById('toggleHistoryViewBtn').addEventListener('click',()=>{
  const c=document.getElementById('historyContent'),btn=document.getElementById('toggleHistoryViewBtn');
  historyVisible=!historyVisible; c.style.display=historyVisible?'block':'none'; btn.textContent=historyVisible?'🙈 Hide':'👁 Show';
});
async function deleteTx(id){if(!confirm('Delete?'))return;try{await api(`/api/transactions/${id}`,{method:'DELETE'});toast('Deleted');allTransactions=allTransactions.filter(t=>t.id!==id);renderHistory(allTransactions);}catch(e){toast(e.message,true);}}

// ═══════════════════════════════════════════════
// PROFILE
// ═══════════════════════════════════════════════
document.getElementById('profileFileInput').addEventListener('change',function(){
  const file=this.files[0]; if(!file)return;
  const r=new FileReader(); r.onload=ev=>document.getElementById('profileAvatar').value=ev.target.result; r.readAsDataURL(file);
});
document.getElementById('saveProfileBtn').addEventListener('click',async()=>{
  const body={name:document.getElementById('profileName').value.trim(),email:document.getElementById('profileEmail').value.trim(),avatar_url:document.getElementById('profileAvatar').value.trim()};
  const pw=document.getElementById('profilePass').value; if(pw)body.password=pw;
  try{
    const u=await api('/api/profile',{method:'PUT',body:JSON.stringify(body)});
    currentUser=u;
    const av=document.getElementById('userAvatar'),init=av.querySelector('.avatar-initial'),img=av.querySelector('img');
    if(u.avatar_url){img.src=u.avatar_url;img.style.display='block';init.style.display='none';}
    else{img.style.display='none';init.style.display='block';init.textContent=u.name?.charAt(0)?.toUpperCase()||'?';}
    toast('Profile updated!');
  }catch(e){toast(e.message,true);}
});

// ═══════════════════════════════════════════════
// ADMIN
// ═══════════════════════════════════════════════
async function loadAdmin(){
  if(!currentUser||currentUser.role!=='admin')return;
  try{
    const stats=await api('/api/admin/stats');
    document.getElementById('adminStats').innerHTML=`<div class="stat-card"><div class="stat-value">${stats.total_users}</div><div class="stat-label">Users</div></div><div class="stat-card"><div class="stat-value">${stats.total_transactions}</div><div class="stat-label">Transactions</div></div><div class="stat-card"><div class="stat-value">${fmt(stats.total_income)}</div><div class="stat-label">Total Income</div></div><div class="stat-card"><div class="stat-value">${fmt(stats.total_expense)}</div><div class="stat-label">Total Expense</div></div><div class="stat-card blue-glow"><div class="stat-value">${stats.avg_health_score}</div><div class="stat-label">Avg Health Score</div></div>`;
    loadAdminUsers(); loadAdminTransactions();
  }catch(e){toast(e.message,true);}
}
document.getElementById('adminSearchUser').addEventListener('input',loadAdminUsers);
document.getElementById('adminUserFilter').addEventListener('change',loadAdminTransactions);
document.querySelectorAll('#adminTabs .tab').forEach(tab=>{tab.addEventListener('click',()=>{document.querySelectorAll('#adminTabs .tab').forEach(t=>t.classList.remove('active'));tab.classList.add('active');document.getElementById('adminUsersPanel').style.display=tab.dataset.tab==='users'?'block':'none';document.getElementById('adminTransactionsPanel').style.display=tab.dataset.tab==='users'?'none':'block';});});
async function loadAdminUsers(){try{const users=await api('/api/admin/users');const q=document.getElementById('adminSearchUser').value.toLowerCase();const f=users.filter(u=>u.name.toLowerCase().includes(q)||u.email.toLowerCase().includes(q));document.getElementById('adminUserTable').innerHTML=f.map(u=>`<tr><td>${u.id}</td><td>${esc(u.name)}</td><td>${esc(u.email)}</td><td>${esc(u.role)}</td><td>${fmtDate(u.created_at)}</td><td><button onclick="adminDelUser(${u.id})" class="btn btn-danger" style="padding:4px 8px;font-size:0.7rem;">Del</button></td></tr>`).join('');}catch(e){toast(e.message,true);}}
async function adminDelUser(id){if(!confirm('Delete user and all their data?'))return;try{await api(`/api/admin/users/${id}`,{method:'DELETE'});toast('User deleted');loadAdminUsers();}catch(e){toast(e.message,true);}}
async function loadAdminTransactions(){try{const uid=document.getElementById('adminUserFilter').value;const txs=await api(uid?`/api/admin/transactions?user_id=${uid}`:'/api/admin/transactions');document.getElementById('adminTxTable').innerHTML=txs.map(t=>`<tr><td>${fmtDate(t.tx_date)}</td><td>User ${t.user_id}</td><td>${esc(t.category)}</td><td>${t.tx_type}</td><td>${fmt(t.amount)}</td><td>${t.is_need?'Need':'Want'}</td></tr>`).join('');const users=await api('/api/admin/users');document.getElementById('adminUserFilter').innerHTML='<option value="">All Users</option>'+users.map(u=>`<option value="${u.id}">${esc(u.name)} (${u.id})</option>`).join('');}catch(e){toast(e.message,true);}}

// ═══════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════
document.getElementById('exportBtn').addEventListener('click',()=>window.open('/api/export/csv','_blank'));

// ═══════════════════════════════════════════════
// CHATBOT
// ═══════════════════════════════════════════════
(function(){
  const cb=document.getElementById('chatbot'),hdr=document.getElementById('chatHeader');
  let ox,oy,drag=false;
  hdr.addEventListener('mousedown',e=>{if(e.target.id==='chatToggleBtn')return;drag=true;ox=e.clientX-cb.getBoundingClientRect().left;oy=e.clientY-cb.getBoundingClientRect().top;cb.style.cursor='grabbing';e.preventDefault();});
  document.addEventListener('mousemove',e=>{if(!drag)return;cb.style.left=Math.max(0,Math.min(e.clientX-ox,window.innerWidth-cb.offsetWidth))+'px';cb.style.top=Math.max(0,Math.min(e.clientY-oy,window.innerHeight-cb.offsetHeight))+'px';cb.style.right='auto';cb.style.bottom='auto';});
  document.addEventListener('mouseup',()=>{if(drag){drag=false;cb.style.cursor='';}});
  document.getElementById('chatToggleBtn').addEventListener('click',()=>{cb.classList.toggle('minimized');document.getElementById('chatToggleBtn').textContent=cb.classList.contains('minimized')?'□':'–';});
})();
document.getElementById('chatSend').addEventListener('click',sendChat);
document.getElementById('chatInp').addEventListener('keypress',e=>{if(e.key==='Enter')sendChat();});
async function sendChat(){
  const inp=document.getElementById('chatInp');const msg=inp.value.trim();if(!msg)return;
  const msgs=document.getElementById('chatMsgs');
  msgs.innerHTML+=`<div class="msg user">${esc(msg)}</div>`;inp.value='';
  try{const r=await api('/api/ai/chat',{method:'POST',body:JSON.stringify({message:msg})});msgs.innerHTML+=`<div class="msg bot">${esc(r.reply)}</div>`;}
  catch(e){msgs.innerHTML+=`<div class="msg bot">Sorry, I'm having trouble right now.</div>`;}
  msgs.scrollTop=msgs.scrollHeight;
}

// ═══════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════
(async()=>{
  try{const user=await api('/api/me');currentUser=user;authOverlay.style.display='none';initApp();}
  catch(e){authOverlay.style.display='flex';}
})();
</script>
</body>
</html>"""


# ----------------------------------------------------------------------
# Route to serve the HTML page
# ----------------------------------------------------------------------
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
