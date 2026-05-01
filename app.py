import json
import csv
import io
import os
import google.generativeai as genai
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps

from flask import Flask, request, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import inspect, text

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://makamandag_db_user:zcDibuXdlpEpcZNGEYLc9nqpgWwuTTfO@dpg-d7od7md7vvec739acfj0-a/makamandag_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Gemini AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ----------------------------------------------------------------------
# Database models (unchanged)
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
            'monthly_budget_limit': self.monthly_budget_limit
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
    tx_date = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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
# Authentication helper
# ----------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    user_id = session.get('user_id')
    return User.query.get(user_id) if user_id else None

# ----------------------------------------------------------------------
# AI-powered allocation endpoint (improved with budget limits)
# ----------------------------------------------------------------------
@app.route('/api/ai_allocate', methods=['POST'])
@login_required
def ai_allocate():
    user = get_current_user()
    data = request.json
    selected_categories = data.get('categories', [])
    monthly_budget = data.get('monthly_budget', user.monthly_budget_limit or 10000)

    # Fetch user's recent spending
    recent_spending = defaultdict(float)
    transactions = Transaction.query.filter_by(user_id=user.id, tx_type='expense').order_by(Transaction.tx_date.desc()).limit(20).all()
    for t in transactions:
        recent_spending[t.category] += t.amount

    # Fetch existing budget limits
    budget_limits = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user.id).all()}

    # Mindset & status
    mindset = user.spending_mindset.lower()
    status = user.social_status.lower()

    prompt = f"""
You are a financial AI. Allocate 100% of a monthly budget of ₱{monthly_budget} among these categories: {selected_categories}.
User profile:
- Social status: {status}
- Spending mindset: {mindset}
- Recent spending: {dict(recent_spending)}
- Existing budget limits (if any): {budget_limits}

Rules:
- If mindset is "saver": give at least 70% to needs (Food, Debt, Mortgage, Transport). Keep wants low.
- If "spender": allow up to 40% to wants.
- If "neutral": balanced.
- Use existing budget limits as a reference but ensure total 100%.
Return ONLY a JSON object with category names as keys and percentage values as floats summing to 100.
Example: {{"Food & dining": 35.0, "Debt repayment": 25.0, ...}}
No extra text.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        allocation = json.loads(response.text.strip())
        total = sum(allocation.values())
        if abs(total - 100) > 0.1:
            factor = 100 / total
            allocation = {k: round(v * factor, 1) for k, v in allocation.items()}
        return jsonify({'allocation': allocation}), 200
    except Exception as e:
        print(f"Gemini allocation error: {e}")
        # Smart fallback: use recent spending ratios if available
        if recent_spending:
            total_spent = sum(recent_spending.values())
            if total_spent > 0:
                allocation = {}
                for cat in selected_categories:
                    # Default to equal split if no spending data
                    pct = (recent_spending.get(cat, 0) / total_spent) * 100
                    allocation[cat] = round(pct, 1)
                # Normalize to 100
                total = sum(allocation.values())
                if total != 100:
                    factor = 100 / total
                    allocation = {k: round(v * factor, 1) for k, v in allocation.items()}
                return jsonify({'allocation': allocation, 'fallback': 'recent_spending'}), 200
        # Last resort: equal split
        fallback_pct = 100 / len(selected_categories) if selected_categories else 0
        fallback = {cat: round(fallback_pct, 1) for cat in selected_categories}
        return jsonify({'allocation': fallback, 'fallback': 'equal'}), 200

# ----------------------------------------------------------------------
# Savings plan endpoint
# ----------------------------------------------------------------------
@app.route('/api/savings_plan', methods=['GET'])
@login_required
def savings_plan():
    user = get_current_user()
    summary = get_monthly_summary(user.id)
    monthly_income = summary['income']
    monthly_expense = summary['expense']
    balance = summary['balance']
    mindset = user.spending_mindset

    prompt = f"""
User monthly income: ₱{monthly_income:.2f}, monthly expenses: ₱{monthly_expense:.2f}, current balance: ₱{balance:.2f}.
Spending mindset: {mindset}.
Recommend how much they should save per day, per week, and per month. Give realistic, actionable amounts.
Return a JSON object: {{"daily": float, "weekly": float, "monthly": float}}.
No extra text.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        plan = json.loads(response.text.strip())
        return jsonify(plan), 200
    except Exception as e:
        print(f"Savings plan error: {e}")
        # Fallback: save 20% of disposable income
        disposable = max(0, monthly_income - monthly_expense)
        monthly_save = disposable * 0.2
        return jsonify({
            "daily": round(monthly_save / 30, 2),
            "weekly": round(monthly_save / 4, 2),
            "monthly": round(monthly_save, 2)
        }), 200

# ----------------------------------------------------------------------
# Apply future expenses (deduction)
# ----------------------------------------------------------------------
@app.route('/api/apply_future_expenses', methods=['POST'])
@login_required
def apply_future_expenses():
    user = get_current_user()
    today = datetime.utcnow().date()
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
            is_need=(exp.category in ['Food & dining', 'Debt repayment', 'Mortgage', 'Transport']),
            priority=1,
            note=f"Auto-deducted future expense: {exp.description}"
        )
        db.session.add(tx)
        applied.append(exp.description)
        db.session.delete(exp)
    db.session.commit()
    return jsonify({'applied': applied, 'count': len(applied)}), 200

# ----------------------------------------------------------------------
# ML / Prediction helpers (full implementations)
# ----------------------------------------------------------------------
def compute_health_score(user_id):
    user = User.query.get(user_id)
    if not user:
        return 50
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
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
    need_expense = sum(e.amount for e in expenses if e.is_need)
    total_exp = want_expense + need_expense
    want_ratio = want_expense / total_exp if total_exp > 0 else 0
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.limit_amount for b in budgets}
    overspend_penalty = 0
    for cat, limit in budget_dict.items():
        cat_spent = sum(e.amount for e in expenses if e.category == cat)
        if cat_spent > limit:
            overspend_penalty += (cat_spent - limit) / limit
    score = 70
    score += int(savings_rate * 20)
    score -= int(want_ratio * 15)
    score -= min(20, int(overspend_penalty * 10))
    return max(0, min(100, score))

def generate_weekly_forecast(user_id):
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    if len(expenses) < 7:
        avg_daily = sum(e.amount for e in expenses) / max(1, len(expenses))
    else:
        daily_totals = defaultdict(float)
        for e in expenses:
            key = e.tx_date.date()
            daily_totals[key] += e.amount
        days_sorted = sorted(daily_totals.keys())
        values = [daily_totals[d] for d in days_sorted]
        n = len(values)
        if n > 1:
            x_mean = (n - 1) / 2
            y_mean = sum(values) / n
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean
            week_sums = [0, 0, 0, 0]
            for day in range(1, 29):
                predicted = max(0, intercept + slope * (n + day))
                week_sums[(day - 1) // 7] += predicted
            return {
                'Week 1': week_sums[0],
                'Week 2': week_sums[1],
                'Week 3': week_sums[2],
                'Week 4': week_sums[3]
            }
    daily_avg = sum(e.amount for e in expenses) / max(1, len(expenses))
    weekly = daily_avg * 7
    return {'Week 1': weekly, 'Week 2': weekly, 'Week 3': weekly, 'Week 4': weekly}

def get_category_forecast(user_id):
    now = datetime.utcnow()
    first_of_month = datetime(now.year, now.month, 1)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= first_of_month
    ).all()
    cat_totals = defaultdict(float)
    for e in expenses:
        cat_totals[e.category] += e.amount
    return dict(cat_totals)

def generate_advice(user_id):
    user = User.query.get(user_id)
    if not user:
        return []
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    current_month_expenses = get_category_forecast(user_id)
    advice = []
    for category, spent in current_month_expenses.items():
        limit = budgets.get(category)
        if limit and spent > limit:
            advice.append({
                'cat': category,
                'msg': f"You've exceeded the {category} budget by ₱{spent - limit:.2f}. Consider reducing non-essential purchases."
            })
    expenses = Transaction.query.filter_by(user_id=user_id, tx_type='expense').all()
    wants = sum(e.amount for e in expenses if not e.is_need)
    needs = sum(e.amount for e in expenses if e.is_need)
    total = wants + needs
    if total > 0 and (wants / total) > 0.4:
        advice.append({
            'cat': 'Wants',
            'msg': f"Your wants spending is {wants/total*100:.0f}% of total expenses. Try to keep wants below 30% for better savings."
        })
    if not advice:
        advice.append({'cat': 'Great job!', 'msg': 'Your spending is well within budgets. Keep it up!'})
    return advice

def get_monthly_summary(user_id):
    user = User.query.get(user_id)
    if not user:
        return {'balance': 0, 'expense': 0, 'income': 0, 'monthly': {}}
    all_trans = Transaction.query.filter_by(user_id=user_id).all()
    balance = 0
    for t in all_trans:
        if t.tx_type == 'income':
            balance += t.amount
        else:
            balance -= t.amount
    now = datetime.utcnow()
    first_of_month = datetime(now.year, now.month, 1)
    this_month_expense = sum(t.amount for t in all_trans if t.tx_type == 'expense' and t.tx_date >= first_of_month)
    this_month_income = sum(t.amount for t in all_trans if t.tx_type == 'income' and t.tx_date >= first_of_month)
    monthly = defaultdict(lambda: {'income': 0, 'expense': 0})
    for t in all_trans:
        key = t.tx_date.strftime('%Y-%m')
        if t.tx_type == 'income':
            monthly[key]['income'] += t.amount
        else:
            monthly[key]['expense'] += t.amount
    sorted_months = sorted(monthly.keys())[-6:]
    monthly_hist = {m: monthly[m] for m in sorted_months}
    return {
        'balance': balance,
        'expense': this_month_expense,
        'income': this_month_income,
        'monthly': monthly_hist
    }

def compute_longevity(user_id):
    summary = get_monthly_summary(user_id)
    balance = summary['balance']
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    expenses = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.tx_type == 'expense',
        Transaction.tx_date >= thirty_days_ago
    ).all()
    total_spent = sum(e.amount for e in expenses)
    avg_daily = total_spent / 30 if len(expenses) > 0 else 0
    days = int(balance / avg_daily) if avg_daily > 0 else 0
    return {
        'balance': balance,
        'avg_daily_spend': avg_daily,
        'days': days
    }

# ----------------------------------------------------------------------
# Routes (all endpoints)
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return HTML_PAGE

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not name or not email or not password:
        return jsonify({'error': 'Missing fields'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400
    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    session['user_id'] = user.id
    return jsonify(user.to_dict()), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
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
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(user.to_dict()), 200

@app.route('/api/transactions', methods=['GET'])
@login_required
def list_transactions():
    user = get_current_user()
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).all()
    return jsonify([t.to_dict() for t in transactions]), 200

@app.route('/api/transactions', methods=['POST'])
@login_required
def create_transaction():
    user = get_current_user()
    data = request.json
    required = ['amount', 'category', 'tx_type']
    if not all(k in data for k in required):
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

    # Auto-apply future expenses if salary income
    if tx.tx_type == 'income' and tx.category.lower() == 'salary':
        apply_future_expenses_logic(user.id)
    return jsonify(tx.to_dict()), 201

def apply_future_expenses_logic(user_id):
    today = datetime.utcnow().date()
    future_expenses = FutureExpense.query.filter(
        FutureExpense.user_id == user_id,
        FutureExpense.expense_date <= today,
        FutureExpense.cycle == 'One-time'
    ).all()
    for exp in future_expenses:
        tx = Transaction(
            user_id=user_id,
            amount=exp.amount,
            category=exp.category,
            tx_type='expense',
            is_need=(exp.category in ['Food & dining', 'Debt repayment', 'Mortgage', 'Transport']),
            priority=1,
            note=f"Auto-deducted future expense: {exp.description}"
        )
        db.session.add(tx)
        db.session.delete(exp)
    db.session.commit()

@app.route('/api/transactions/<int:tx_id>', methods=['PATCH'])
@login_required
def patch_transaction(tx_id):
    user = get_current_user()
    tx = Transaction.query.get(tx_id)
    if not tx or tx.user_id != user.id:
        return jsonify({'error': 'Not found'}), 404
    data = request.json
    if 'is_need' in data:
        tx.is_need = data['is_need']
    if 'priority' in data:
        tx.priority = data['priority']
    if 'note' in data:
        tx.note = data['note']
    db.session.commit()
    return jsonify(tx.to_dict()), 200

@app.route('/api/budgets/<int:user_id>', methods=['GET'])
@login_required
def get_budgets(user_id):
    current = get_current_user()
    if current.id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    budgets = Budget.query.filter_by(user_id=user_id).all()
    return jsonify([{'category': b.category, 'limit': b.limit_amount} for b in budgets]), 200

@app.route('/api/budgets/<int:user_id>', methods=['POST'])
@login_required
def upsert_budget(user_id):
    current = get_current_user()
    if current.id != user_id:
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
    return jsonify({'category': category, 'limit': limit}), 200

@app.route('/api/user/profile', methods=['POST'])
@login_required
def update_profile():
    user = get_current_user()
    data = request.json
    if 'social_status' in data:
        user.social_status = data['social_status']
    if 'spending_mindset' in data:
        user.spending_mindset = data['spending_mindset']
    if 'monthly_budget_limit' in data:
        user.monthly_budget_limit = data['monthly_budget_limit']
    db.session.commit()
    return jsonify(user.to_dict()), 200

@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    current = get_current_user()
    if current.id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(get_monthly_summary(user_id)), 200

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    current = get_current_user()
    if current.id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    expenses = Transaction.query.filter_by(user_id=user_id, tx_type='expense').count()
    has_data = expenses > 0
    if not has_data:
        return jsonify({
            'has_data': False,
            'score': None,
            'predictions': {'weekly': {}, 'categories': {}},
            'advice': []
        }), 200
    score = compute_health_score(user_id)
    weekly = generate_weekly_forecast(user_id)
    categories = get_category_forecast(user_id)
    advice = generate_advice(user_id)
    return jsonify({
        'has_data': True,
        'score': score,
        'predictions': {'weekly': weekly, 'categories': categories},
        'advice': advice
    }), 200

@app.route('/api/longevity/<int:user_id>')
@login_required
def longevity(user_id):
    current = get_current_user()
    if current.id != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(compute_longevity(user_id)), 200

@app.route('/api/export/csv')
@login_required
def export_csv():
    user = get_current_user()
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Category', 'Type', 'Need?', 'Priority', 'Note', 'Amount'])
    for t in transactions:
        writer.writerow([
            t.tx_date.strftime('%Y-%m-%d %H:%M'),
            t.category,
            t.tx_type,
            'Need' if t.is_need else 'Want' if t.tx_type == 'expense' else '',
            t.priority,
            t.note,
            t.amount
        ])
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers.set('Content-Disposition', 'attachment', filename='transactions.csv')
    return response

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    user = get_current_user()
    data = request.json
    user_message = data.get('message', '')

    summary = get_monthly_summary(user.id)
    recent_transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.tx_date.desc()).limit(10).all()
    recent_list = [f"{t.category}: ₱{t.amount:.2f} on {t.tx_date.strftime('%Y-%m-%d')}" for t in recent_transactions]
    health = compute_health_score(user.id)
    advice = generate_advice(user.id)

    prompt = f"""
You are SmartSpend AI, a friendly personal finance assistant for {user.name}.

Current snapshot:
- Balance: ₱{summary['balance']:,.2f}
- Monthly income: ₱{summary['income']:,.2f}
- Monthly expenses: ₱{summary['expense']:,.2f}
- Health score: {health}/100
- Recent advice: {advice}

Recent transactions:
{chr(10).join(recent_list) if recent_list else 'None'}

User asks: "{user_message}"

Provide helpful, actionable, concise advice (max 200 words). Include future insights, saving tips, and suggestions to reduce wants if needed. Never say you are an AI unless asked.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip()
    except Exception as e:
        print(f"Chat error: {e}")
        reply = "Sorry, I'm having trouble connecting. Please try again later."
    return jsonify({'reply': reply}), 200

@app.route('/api/future_expenses', methods=['GET'])
@login_required
def get_future_expenses():
    user = get_current_user()
    expenses = FutureExpense.query.filter_by(user_id=user.id).order_by(FutureExpense.expense_date).all()
    return jsonify([e.to_dict() for e in expenses]), 200

@app.route('/api/future_expenses', methods=['POST'])
@login_required
def create_future_expense():
    user = get_current_user()
    data = request.json
    required = ['description', 'amount', 'category', 'date']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing fields'}), 400
    expense = FutureExpense(
        user_id=user.id,
        description=data['description'],
        amount=data['amount'],
        category=data['category'],
        cycle=data.get('cycle', 'One-time'),
        expense_date=datetime.fromisoformat(data['date']).date()
    )
    db.session.add(expense)
    db.session.commit()
    return jsonify(expense.to_dict()), 201

@app.route('/api/future_expenses/<int:exp_id>', methods=['DELETE'])
@login_required
def delete_future_expense(exp_id):
    user = get_current_user()
    expense = FutureExpense.query.get(exp_id)
    if not expense or expense.user_id != user.id:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(expense)
    db.session.commit()
    return jsonify({'message': 'Deleted'}), 200

@app.route('/api/allocations', methods=['GET'])
@login_required
def get_allocations():
    user = get_current_user()
    allocs = UserAllocation.query.filter_by(user_id=user.id).all()
    return jsonify([{'category_name': a.category_name, 'type': a.type, 'percentage': a.percentage} for a in allocs]), 200

@app.route('/api/allocations', methods=['POST'])
@login_required
def update_allocations():
    user = get_current_user()
    data = request.json
    if not isinstance(data, list):
        return jsonify({'error': 'Expected list of allocations'}), 400
    UserAllocation.query.filter_by(user_id=user.id).delete()
    for item in data:
        alloc = UserAllocation(
            user_id=user.id,
            category_name=item['category_name'],
            type=item['type'],
            percentage=item['percentage']
        )
        db.session.add(alloc)
    db.session.commit()
    return jsonify({'message': 'Allocations saved'}), 200

# ----------------------------------------------------------------------
# Schema migration
# ----------------------------------------------------------------------
def ensure_schema():
    inspector = inspect(db.engine)
    if inspector.has_table('users'):
        existing_columns = [col['name'] for col in inspector.get_columns('users')]
        if 'password' in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users DROP COLUMN password'))
                conn.commit()
            print("✅ Dropped obsolete 'password' column.")
        if 'password_hash' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN password_hash VARCHAR(128) NOT NULL DEFAULT \'\''))
                conn.commit()
            print("✅ Added password_hash column.")
        if 'social_status' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN social_status VARCHAR(20) DEFAULT \'Middle\''))
                conn.commit()
            print("✅ Added social_status column.")
        if 'spending_mindset' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN spending_mindset VARCHAR(20) DEFAULT \'Neutral\''))
                conn.commit()
            print("✅ Added spending_mindset column.")
        if 'monthly_budget_limit' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN monthly_budget_limit FLOAT DEFAULT 0.0'))
                conn.commit()
            print("✅ Added monthly_budget_limit column.")

with app.app_context():
    db.create_all()
    ensure_schema()

# ----------------------------------------------------------------------
# Embedded HTML (full frontend with all features)
# ----------------------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartSpend - AI Finance</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --bg: #0a0f1a; --bg2: #0f1622; --bg3: #151e2d;
            --panel: rgba(21, 30, 45, 0.9); --border: rgba(0, 210, 130, 0.2);
            --border2: rgba(255, 255, 255, 0.05); --green: #00d282; --green2: #00ff9d;
            --green-dim: rgba(0, 210, 130, 0.15); --red: #ff4d6d; --red-dim: rgba(255, 77, 109, 0.15);
            --amber: #f0a500; --blue: #3b82f6; --purple: #a855f7;
            --muted: #6c86a0; --text: #e2eff8;
            --mono: 'JetBrains Mono', monospace; --sans: 'Space Grotesk', sans-serif; --r: 16px;
        }
        body { font-family: var(--sans); background: var(--bg); color: var(--text); }
        .sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 80px; background: rgba(15, 22, 34, 0.98); backdrop-filter: blur(12px); border-right: 1px solid var(--border2); display: flex; flex-direction: column; align-items: center; padding: 24px 0; gap: 10px; z-index: 100; transition: width 0.3s; }
        .sidebar:hover { width: 240px; }
        .sidebar-logo { width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, var(--green), #009e5f); display: flex; align-items: center; justify-content: center; margin-bottom: 28px; cursor: pointer; font-size: 1.5rem; }
        .nav-item { width: 100%; display: flex; align-items: center; gap: 16px; padding: 12px 24px; border-radius: 12px; cursor: pointer; border: none; background: transparent; color: var(--muted); font-size: 0.9rem; font-weight: 500; white-space: nowrap; overflow: hidden; transition: all 0.2s; }
        .nav-item:hover { background: var(--green-dim); color: var(--text); transform: translateX(6px); }
        .nav-item.active { background: var(--green-dim); color: var(--green); border-left: 3px solid var(--green); }
        .nav-label { opacity: 0; transition: opacity 0.2s; }
        .sidebar:hover .nav-label { opacity: 1; }
        .sidebar-bottom { margin-top: auto; width: 100%; }
        .main { margin-left: 80px; padding: 24px 32px; min-height: 100vh; }
        @media (max-width: 768px) { .sidebar { width: 70px; } .main { margin-left: 70px; padding: 20px; } }
        .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 28px; flex-wrap: wrap; gap: 16px; }
        .topbar h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #fff, var(--green)); background-clip: text; -webkit-background-clip: text; color: transparent; }
        .score-pill { display: flex; align-items: center; gap: 10px; padding: 8px 20px; border-radius: 99px; background: var(--green-dim); border: 1px solid var(--border); font-size: 0.85rem; font-weight: 600; }
        .avatar { width: 42px; height: 42px; border-radius: 50%; background: linear-gradient(135deg, var(--blue), var(--purple)); display: flex; align-items: center; justify-content: center; font-weight: 700; cursor: pointer; }
        .signout-btn { background: var(--red-dim); border: 1px solid rgba(255, 77, 109, 0.2); color: var(--red); padding: 8px 18px; border-radius: 10px; cursor: pointer; }
        .screen { display: none; animation: fadeUp 0.35s ease; }
        .screen.active { display: block; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .stat-card { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; cursor: pointer; }
        .stat-value { font-size: 1.8rem; font-weight: 700; font-family: var(--mono); }
        .stat-label { font-size: 0.7rem; text-transform: uppercase; color: var(--muted); margin-top: 8px; }
        .panel { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; margin-bottom: 20px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
        .btn { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); border-radius: 10px; padding: 10px 14px; cursor: pointer; font-weight: 600; }
        .btn-green { background: linear-gradient(135deg, var(--green), #009e5f); color: #000; border: none; }
        .btn-ai { background: linear-gradient(135deg, var(--purple), var(--blue)); color: white; border: none; }
        input, select { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); border-radius: 10px; padding: 10px 14px; outline: none; }
        .cat-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
        .cat-row input[type="number"] { width: 80px; }
        .allocation-box { background: var(--bg3); border-radius: 12px; padding: 16px; margin: 12px 0; }
        .ml-badge { background: linear-gradient(135deg, var(--purple), var(--blue)); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 8px; }
        .forecast-scenarios { display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
        .scenario-card { background: var(--bg3); border-radius: 12px; padding: 12px; flex: 1; text-align: center; cursor: pointer; }
        .scenario-card.active { border: 2px solid var(--green); background: var(--green-dim); }
        .future-expense-item { background: var(--bg3); border-radius: 12px; padding: 12px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        .warning { color: var(--red); font-size: 0.8rem; margin-top: 8px; }
        .chat-container { position: fixed; bottom: 24px; right: 24px; z-index: 10001; cursor: move; }
        .chat-window { width: 380px; height: 480px; background: var(--bg2); backdrop-filter: blur(10px); border: 1px solid var(--green); border-radius: 24px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        .chat-header { padding: 14px 18px; background: var(--bg3); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; cursor: move; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 85%; padding: 10px 14px; border-radius: 18px; font-size: 0.85rem; }
        .user-message { align-self: flex-end; background: var(--green-dim); color: var(--green); border-bottom-right-radius: 4px; }
        .bot-message { align-self: flex-start; background: var(--bg3); color: var(--text); border-bottom-left-radius: 4px; }
        .chat-input { display: flex; padding: 14px; gap: 10px; background: var(--bg3); border-top: 1px solid var(--border); }
        #toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 1px solid var(--green); border-radius: 12px; padding: 12px 24px; opacity: 0; transition: all 0.25s; z-index: 10000; }
        #toast.show { opacity: 1; }
        .auth-overlay { position: fixed; inset: 0; background: rgba(8,13,20,0.98); backdrop-filter: blur(20px); z-index: 9999; display: flex; align-items: center; justify-content: center; }
        .auth-card { background: linear-gradient(145deg, #0d1520, #0a1220); border: 1px solid rgba(0,210,130,0.2); border-radius: 28px; padding: 40px; width: 400px; max-width: 90%; }
        .last-expense-item { background: var(--bg3); border-radius: 10px; padding: 10px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
        .forecast-row { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
        .forecast-week { width: 70px; font-weight: 600; color: var(--green); }
        .forecast-bar-track { flex: 1; height: 8px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
        .forecast-bar-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--green2)); width: 0; border-radius: 99px; transition: width 1s ease; }
    </style>
</head>
<body>
<nav class="sidebar">
    <div class="sidebar-logo">💰</div>
    <button class="nav-item active" data-nav="dashboard"><span class="nav-label">Dashboard</span></button>
    <button class="nav-item" data-nav="add"><span class="nav-label">Add Transaction</span></button>
    <button class="nav-item" data-nav="analytics"><span class="nav-label">Analytics</span></button>
    <button class="nav-item" data-nav="budgets"><span class="nav-label">Budgets</span></button>
    <button class="nav-item" data-nav="transactions"><span class="nav-label">History</span></button>
    <div class="sidebar-bottom"><button class="nav-item" id="exportBtn"><span class="nav-label">Export CSV</span></button></div>
</nav>
<main class="main">
    <div class="topbar">
        <div><h1 id="pageTitle">Dashboard</h1></div>
        <div class="topbar-right">
            <div class="score-pill" id="scorePill"><div class="score-dot"></div><span>Health: <strong id="topScore">—</strong><span class="ml-badge">ML</span></span></div>
            <div class="avatar" id="userAvatar">US</div>
            <button class="signout-btn" id="signoutBtn">Sign Out</button>
        </div>
    </div>
    <div id="toast"></div>

    <!-- DASHBOARD -->
    <div class="screen active" id="screen-dashboard">
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value" id="sBalance">—</div><div class="stat-label">Balance</div></div>
            <div class="stat-card"><div class="stat-value" id="sExpense">—</div><div class="stat-label">Monthly Expenses</div></div>
            <div class="stat-card"><div class="stat-value" id="sIncome">—</div><div class="stat-label">Monthly Income</div></div>
            <div class="stat-card"><div class="stat-value" id="sScore">—</div><div class="stat-label">ML Health Score</div></div>
        </div>
        <div class="panel"><div class="panel-header">Income vs Expense Trend</div><canvas id="monthlyChart" height="200"></canvas></div>
        <div class="panel"><div class="panel-header">ML Spending Forecast (4 weeks)</div><div id="forecastBars"></div></div>
        <div class="panel"><div class="panel-header">AI Advice</div><div id="adviceList"></div></div>
    </div>

    <!-- ADD TRANSACTION SCREEN -->
    <div class="screen" id="screen-add">
        <div class="panel">
            <div class="panel-header">🧠 AI Autonomous Budget Allocation <span class="ml-badge">GEMINI AI</span></div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
                <div style="flex:1;"><label>Monthly Budget (₱)</label><input type="number" id="monthlyBudgetCap" placeholder="Auto from income" step="100"></div>
                <div style="flex:1;"><label>Social Status</label><select id="socialStatusDiagram"><option>Low</option><option selected>Middle</option><option>Upper</option></select></div>
                <div style="flex:1;"><label>Mindset</label><select id="mindsetDiagram"><option>Saver</option><option selected>Neutral</option><option>Spender</option></select></div>
                <div style="flex:1;"><label>Cycle</label><select id="budgetCycle"><option>Daily</option><option>Weekly</option><option selected>Monthly</option><option>Yearly</option></select></div>
            </div>
            <div style="text-align: right; margin-bottom: 10px;">
                <button id="aiAutoAllocateBtn" class="btn btn-ai">🤖 AI Auto-Allocate (Gemini)</button>
                <span class="ml-badge">100% autonomous</span>
            </div>
            <div id="categoryAllocationList"></div>
            <div class="allocation-box" id="allocationDisplay">Needs: -- / Wants: -- / Savings: --</div>
            <div class="pref-checkbox"><label><input type="checkbox" id="autoRemainingToWants" checked> Automatically allocate remaining % to Wants</label></div>
            <div id="allocationWarning" class="warning" style="display: none;">⚠️ Total allocation must be 100%</div>
        </div>

        <div class="panel">
            <div class="panel-header">📌 PIN FUTURE EXPENSE</div>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
                <input type="text" id="futureDesc" placeholder="Description" style="flex:2">
                <input type="number" id="futureAmount" placeholder="Amount (₱)" style="flex:1">
                <select id="futureCategory"><option>Entertainment</option><option>Food & Dining</option><option>Transport</option><option>Groceries</option><option>Health</option><option>Other</option></select>
                <select id="futureCycle"><option>One-time</option><option>Weekly</option><option>Monthly</option></select>
                <input type="date" id="futureDate">
                <button id="pinFutureExpenseBtn" class="btn btn-green">Pin Expense</button>
                <button id="applyFutureBtn" class="btn btn-ai">💰 Process Pending Future Expenses</button>
            </div>
            <div id="futureExpensesList"></div>
        </div>

        <div class="panel">
            <div class="panel-header">📝 LAST 3 EXPENSES (History)</div>
            <div id="lastExpenseList"></div>
        </div>

        <div class="panel">
            <div class="panel-header">📈 ML FORECAST</div>
            <select id="forecastScenarioSelect"><option value="optimistic">Optimistic</option><option value="realistic" selected>Realistic</option><option value="pessimistic">Pessimistic</option></select>
            <button id="linkToAnalyticsBtn" class="btn">View in Analytics →</button>
            <div id="forecastScenariosDisplay" class="forecast-scenarios"></div>
        </div>

        <div class="panel">
            <div class="panel-header">⚙️ REAL‑TIME ML Preferences</div>
            <label><input type="checkbox" id="prefAlertWant"> Alert me before any want expense</label>
            <label><input type="checkbox" id="prefDailyForecast" checked> Show forecast summary</label>
        </div>

        <div class="panel budget-limit-panel">
            <div class="panel-header">💰 Budget Limits (by category) <span class="auto-badge">auto-save</span></div>
            <div id="budgetInputsAdd" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;"></div>
        </div>
    </div>

    <!-- ANALYTICS SCREEN (fully restored) -->
    <div class="screen" id="screen-analytics">
        <div class="panel"><div class="panel-header">⏳ Budget Longevity</div><div id="longevityContainer">Loading...</div></div>
        <div class="panel"><div class="panel-header">Weekly Forecast</div><canvas id="forecastChart" height="200"></canvas></div>
        <div class="panel"><div class="panel-header">Category Breakdown</div><canvas id="catBarChart" height="200"></canvas></div>
    </div>

    <div class="screen" id="screen-budgets">
        <div class="panel"><div class="panel-header">Set Monthly Budget Limits <span class="auto-badge">auto-save</span></div><div id="budgetInputsStandalone" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px;"></div></div>
    </div>

    <div class="screen" id="screen-transactions">
        <div class="table-wrap"><table><thead><th>Date</th><th>Category</th><th>Need?</th><th>Priority</th><th>Note</th><th>Type</th><th>Amount</th></thead><tbody id="txTableBody"></tbody></table></div>
    </div>

    <div id="authOverlay" class="auth-overlay"><div class="auth-card"><h2 id="authTitle">Welcome back</h2><input type="text" id="regName" placeholder="Full Name" style="display:none"><input type="email" id="authEmail" placeholder="Email"><input type="password" id="authPass" placeholder="Password"><input type="password" id="authConfirm" placeholder="Confirm Password" style="display:none"><div id="termsRow" style="display:none;"><label><input type="checkbox" id="termsCheck"> Accept Terms</label></div><div id="authMsg" style="color:#ff4d6d;"></div><button id="authBtn">Sign In</button><div id="toggleAuthLink">Don't have an account? Register</div></div></div>

    <div id="chatContainer" class="chat-container"><div class="chat-window"><div class="chat-header" id="chatHeader">🤖 SmartSpend AI <span class="ml-badge">Gemini</span><button id="closeChatBtn">✕</button></div><div class="chat-messages" id="chatMessages"><div class="message bot-message">💬 I'm SmartSpend AI. Ask me about your spending, savings, future expenses, or how to improve your finances.</div></div><div class="chat-input"><input id="chatInput" placeholder="Ask..."><button id="sendChatBtn">Send</button></div></div></div>
</main>

<script>
let currentUser = null, allTransactions = [], currentSummary = { balance:0, expense:0, income:0 }, currentPrediction = { has_data:false, score:null, predictions:{ weekly:{}, categories:{} }, advice:[] };
let monthlyChart, forecastChart, catBarChart, isLogin = true;
const PRIO_MAP = {0:'Low',1:'Medium',2:'High',3:'Critical'};
function fmt(amt) { return '₱' + Number(amt).toLocaleString('en-PH', { minimumFractionDigits:2 }); }
function toast(msg) { let t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2500); }
async function apiFetch(url, opts={}) { let res = await fetch(url, {...opts, credentials:'include', headers:{'Content-Type':'application/json'}}); if(!res.ok) throw new Error((await res.json()).error); return res.json(); }

// Category config
let categoryConfig = [
    { name: "Food & dining", defaultPct: 0, type: "need" },
    { name: "Debt repayment", defaultPct: 0, type: "need" },
    { name: "Mortgage", defaultPct: 0, type: "need" },
    { name: "Entertainment", defaultPct: 0, type: "want" },
    { name: "Transport", defaultPct: 0, type: "need" },
    { name: "Subscription", defaultPct: 0, type: "want" },
    { name: "Hobbies", defaultPct: 0, type: "want" }
];

function renderCategoryAllocation() {
    let container = document.getElementById('categoryAllocationList');
    let html = '';
    categoryConfig.forEach((cat, idx) => {
        html += `<div class="cat-row">
            <select class="need-want" data-idx="${idx}">
                <option value="need" ${cat.type === 'need' ? 'selected' : ''}>Need</option>
                <option value="want" ${cat.type === 'want' ? 'selected' : ''}>Want</option>
            </select>
            <label><input type="checkbox" class="cat-enabled" data-idx="${idx}" ${cat.pct > 0 ? 'checked' : ''}> ${cat.name}</label>
            <input type="number" class="cat-pct" data-idx="${idx}" value="${cat.pct || 0}" step="1" min="0" max="100" style="width:80px;"> %
        </div>`;
    });
    container.innerHTML = html;
    document.querySelectorAll('.need-want').forEach(sel => sel.addEventListener('change', updateAllocationFromCategories));
    document.querySelectorAll('.cat-enabled').forEach(chk => chk.addEventListener('change', () => { updateAllocationFromCategories(); validateAllocation(); }));
    document.querySelectorAll('.cat-pct').forEach(inp => inp.addEventListener('input', () => { updateAllocationFromCategories(); validateAllocation(); }));
}

function updateAllocationFromCategories() {
    let selected = [];
    for (let i = 0; i < categoryConfig.length; i++) {
        let enabled = document.querySelector(`.cat-enabled[data-idx="${i}"]`)?.checked;
        let type = document.querySelector(`.need-want[data-idx="${i}"]`)?.value;
        let pct = parseFloat(document.querySelector(`.cat-pct[data-idx="${i}"]`)?.value || 0);
        if (enabled && pct > 0) selected.push({ name: categoryConfig[i].name, type, pct });
    }
    let totalNeedPct = selected.filter(c => c.type === 'need').reduce((s,c)=>s+c.pct,0);
    let totalWantPct = selected.filter(c => c.type === 'want').reduce((s,c)=>s+c.pct,0);
    let auto = document.getElementById('autoRemainingToWants').checked;
    if (auto && (totalNeedPct + totalWantPct) < 100) totalWantPct = 100 - totalNeedPct;
    totalNeedPct = Math.min(100, totalNeedPct);
    totalWantPct = Math.min(100 - totalNeedPct, totalWantPct);
    let budget = parseFloat(document.getElementById('monthlyBudgetCap').value) || 10000;
    let needAmount = budget * totalNeedPct / 100;
    let wantAmount = budget * totalWantPct / 100;
    let savings = budget - needAmount - wantAmount;
    document.getElementById('allocationDisplay').innerHTML = `Needs (${totalNeedPct}%): ${fmt(needAmount)} 🔒 | Wants (${totalWantPct}%): ${fmt(wantAmount)} ⚡ | Savings (${(savings/budget*100).toFixed(0)}%): ${fmt(savings)} 🏦`;
}

function validateAllocation() {
    let total = 0;
    for (let i = 0; i < categoryConfig.length; i++) {
        if (document.querySelector(`.cat-enabled[data-idx="${i}"]`)?.checked) {
            total += parseFloat(document.querySelector(`.cat-pct[data-idx="${i}"]`)?.value || 0);
        }
    }
    let warn = document.getElementById('allocationWarning');
    if (Math.abs(total - 100) > 0.01) { warn.style.display = 'block'; warn.innerText = `⚠️ Total is ${total}% – must be 100%.`; }
    else { warn.style.display = 'none'; }
}

async function aiAutoAllocate() {
    if (!currentUser) return;
    let selected = [];
    for (let i = 0; i < categoryConfig.length; i++) {
        if (document.querySelector(`.cat-enabled[data-idx="${i}"]`)?.checked) selected.push(categoryConfig[i].name);
    }
    if (selected.length === 0) { toast("Check at least one category"); return; }
    toast("🤖 AI allocating...");
    let budget = parseFloat(document.getElementById('monthlyBudgetCap').value) || 10000;
    try {
        let res = await apiFetch('/api/ai_allocate', { method: 'POST', body: JSON.stringify({ categories: selected, monthly_budget: budget }) });
        let alloc = res.allocation;
        for (let i = 0; i < categoryConfig.length; i++) {
            let cat = categoryConfig[i].name;
            if (alloc[cat] !== undefined) {
                let inp = document.querySelector(`.cat-pct[data-idx="${i}"]`);
                if (inp) inp.value = alloc[cat].toFixed(1);
                let chk = document.querySelector(`.cat-enabled[data-idx="${i}"]`);
                if (chk && !chk.checked) chk.checked = true;
            }
        }
        updateAllocationFromCategories();
        validateAllocation();
        toast("✅ AI allocation complete! Total 100%");
    } catch(e) { toast("AI failed, using fallback"); console.error(e); }
}

let futureExpenses = [];
function renderFutureExpenses() {
    let container = document.getElementById('futureExpensesList');
    if (!container) return;
    if (futureExpenses.length === 0) { container.innerHTML = '<div class="future-expense-item">No pinned future expenses.</div>'; return; }
    let html = '';
    futureExpenses.forEach((exp, idx) => {
        html += `<div class="future-expense-item"><div><strong>${exp.description}</strong><br>${fmt(exp.amount)} | ${exp.category} | ${exp.cycle} | ${exp.date}</div><button class="btn" data-future-idx="${idx}" style="background:var(--red-dim);">Delete</button></div>`;
    });
    container.innerHTML = html;
    document.querySelectorAll('[data-future-idx]').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            let idx = parseInt(btn.dataset.futureIdx);
            let id = futureExpenses[idx].id;
            try { await apiFetch(`/api/future_expenses/${id}`, { method: 'DELETE' }); toast('Future expense removed'); await loadFutureExpenses(); } catch(e) { toast('Delete failed'); }
        });
    });
}
document.getElementById('pinFutureExpenseBtn')?.addEventListener('click', async () => {
    let desc = document.getElementById('futureDesc').value.trim();
    let amount = parseFloat(document.getElementById('futureAmount').value);
    let category = document.getElementById('futureCategory').value;
    let cycle = document.getElementById('futureCycle').value;
    let date = document.getElementById('futureDate').value;
    if (!desc || isNaN(amount) || amount<=0 || !date) { toast("Fill all fields"); return; }
    try {
        let res = await apiFetch('/api/future_expenses', { method: 'POST', body: JSON.stringify({ description: desc, amount, category, cycle, date }) });
        toast('Future expense pinned');
        await loadFutureExpenses();
        document.getElementById('futureDesc').value = '';
        document.getElementById('futureAmount').value = '';
        document.getElementById('futureDate').value = '';
    } catch(e) { toast('Error pinning'); }
});
document.getElementById('applyFutureBtn')?.addEventListener('click', async () => {
    toast("Processing pending future expenses...");
    try { let res = await apiFetch('/api/apply_future_expenses', { method: 'POST' }); toast(`Applied ${res.count} expenses.`); await loadDashboard(); loadFutureExpenses(); } catch(e) { toast("Error applying"); }
});
async function loadFutureExpenses() { try { futureExpenses = await apiFetch('/api/future_expenses'); renderFutureExpenses(); } catch(e) {} }

async function updateLastExpenses() {
    let expenses = allTransactions.filter(t=>t.tx_type==='expense').sort((a,b)=>new Date(b.tx_date)-new Date(a.tx_date)).slice(0,3);
    let container = document.getElementById('lastExpenseList');
    if (!container) return;
    if (expenses.length === 0) { container.innerHTML = '<div>No expenses yet.</div>'; return; }
    let html = '';
    expenses.forEach((exp, idx) => {
        html += `<div class="last-expense-item"><strong>${exp.category}</strong>: ${fmt(exp.amount)} (${exp.is_need ? 'Need' : 'Want'})<br><small>${new Date(exp.tx_date).toLocaleDateString()}</small>${idx===0 ? `<button class="btn small" id="toggleLastBtn">Mark as ${exp.is_need ? 'Want' : 'Need'}</button>` : ''}</div>`;
    });
    container.innerHTML = html;
    let toggleBtn = document.getElementById('toggleLastBtn');
    if (toggleBtn && expenses[0]) {
        toggleBtn.addEventListener('click', async () => {
            let last = expenses[0];
            await apiFetch(`/api/transactions/${last.id}`, { method:'PATCH', body:JSON.stringify({ is_need: !last.is_need }) });
            toast("Updated! Reloading...");
            await loadDashboard();
            updateLastExpenses();
        });
    }
}

async function updateForecastScenarios() {
    if(!currentUser) return;
    let monthlyBudget = parseFloat(document.getElementById('monthlyBudgetCap').value) || (currentSummary.income || 10000);
    let expenses = allTransactions.filter(t=>t.tx_type==='expense');
    let now = new Date();
    let cycle = document.getElementById('budgetCycle').value;
    let startOfCycle, totalDays;
    if(cycle === 'Daily') { startOfCycle = new Date(now.getFullYear(), now.getMonth(), now.getDate()); totalDays = 1; }
    else if(cycle === 'Weekly') { let day = now.getDay(); startOfCycle = new Date(now); startOfCycle.setDate(now.getDate() - day); totalDays = 7; }
    else if(cycle === 'Monthly') { startOfCycle = new Date(now.getFullYear(), now.getMonth(), 1); totalDays = new Date(now.getFullYear(), now.getMonth()+1, 0).getDate(); }
    else { startOfCycle = new Date(now.getFullYear(), 0, 1); totalDays = 366; }
    let spentThisCycle = expenses.filter(e=> new Date(e.tx_date) >= startOfCycle).reduce((s,e)=>s+e.amount,0);
    let daysElapsed = Math.min(totalDays, Math.floor((now - startOfCycle) / (1000*60*60*24)));
    let remainingDays = Math.max(1, totalDays - daysElapsed);
    let avgDailySpent = spentThisCycle / Math.max(1, daysElapsed);
    let realisticRemaining = avgDailySpent * remainingDays;
    let optimisticRemaining = (avgDailySpent * 0.8) * remainingDays;
    let pessimisticRemaining = (avgDailySpent * 1.2) * remainingDays;
    let remainingBudget = monthlyBudget - spentThisCycle;
    let opt = Math.max(0, remainingBudget - optimisticRemaining);
    let real = Math.max(0, remainingBudget - realisticRemaining);
    let pess = Math.max(0, remainingBudget - pessimisticRemaining);
    document.getElementById('forecastOpt').innerText = fmt(opt);
    document.getElementById('forecastReal').innerText = fmt(real);
    document.getElementById('forecastPess').innerText = fmt(pess);
    let selected = document.getElementById('forecastScenarioSelect').value;
    document.querySelectorAll('.scenario-card').forEach(card => {
        let scenario = card.dataset.scenario;
        if(scenario === selected) card.classList.add('active');
        else card.classList.remove('active');
    });
}
document.getElementById('forecastScenarioSelect')?.addEventListener('change', updateForecastScenarios);
document.getElementById('linkToAnalyticsBtn')?.addEventListener('click', () => navigate('analytics'));

async function updateMLDiagram() {
    if(!currentUser) return;
    let incomeTotal = allTransactions.filter(t=>t.tx_type==='income').reduce((s,t)=>s+t.amount,0);
    let budget = parseFloat(document.getElementById('monthlyBudgetCap').value) || incomeTotal || 10000;
    updateAllocationFromCategories();
    await updateForecastScenarios();
    let status = document.getElementById('socialStatusDiagram').value;
    let mindset = document.getElementById('mindsetDiagram').value;
    await apiFetch('/api/user/profile', { method:'POST', body:JSON.stringify({ social_status:status, spending_mindset:mindset }) });
}
async function autoSaveBudget(cat, val) { if(!currentUser) return; if(val && !isNaN(parseFloat(val)) && parseFloat(val)>0) { await apiFetch(`/api/budgets/${currentUser.id}`, { method:'POST', body:JSON.stringify({ category: cat, limit: parseFloat(val) }) }); toast(`Saved ${cat} limit`); } }
function attachAutoSave(container, cats) { cats.forEach(cat=>{ let inp = document.getElementById(`${container}_${cat.replace(/\\s/g,'')}`); if(inp && !inp.hasAttribute('data-auto')){ inp.setAttribute('data-auto','true'); inp.addEventListener('change',()=>autoSaveBudget(cat,inp.value)); } }); }
async function loadBudgetsForAdd() { let budgets = await apiFetch(`/api/budgets/${currentUser.id}`); let limits = Object.fromEntries(budgets.map(b=>[b.category, b.limit])); let cats = ['Food & Dining','Transport','Groceries','Entertainment','Health','Other']; let html = cats.map(c=>`<div><label>${c}</label><input type="number" id="budgetAdd_${c.replace(/\\s/g,'')}" value="${limits[c]||''}" placeholder="₱ limit"></div>`).join(''); document.getElementById('budgetInputsAdd').innerHTML = html; attachAutoSave('budgetAdd', cats); }
async function loadBudgetsStandalone() { let budgets = await apiFetch(`/api/budgets/${currentUser.id}`); let limits = Object.fromEntries(budgets.map(b=>[b.category, b.limit])); let cats = ['Food & Dining','Transport','Groceries','Entertainment','Health','Other']; let html = cats.map(c=>`<div><label>${c}</label><input type="number" id="budgetStand_${c.replace(/\\s/g,'')}" value="${limits[c]||''}" placeholder="₱ limit"></div>`).join(''); document.getElementById('budgetInputsStandalone').innerHTML = html; attachAutoSave('budgetStand', cats); }

async function loadDashboard() {
    if(!currentUser) return;
    try {
        let summary = await apiFetch(`/api/summary/${currentUser.id}`);
        let pred = await apiFetch(`/api/predict/${currentUser.id}`);
        allTransactions = await apiFetch(`/api/transactions`);
        currentSummary = summary; currentPrediction = pred;
        document.getElementById('sBalance').innerText = fmt(summary.balance);
        document.getElementById('sExpense').innerText = fmt(summary.expense);
        document.getElementById('sIncome').innerText = fmt(summary.income);
        let score = pred.has_data ? pred.score : null;
        document.getElementById('sScore').innerText = score ? score+'/100' : '—';
        document.getElementById('topScore').innerText = score || '—';
        if(pred.has_data) {
            let weeks = pred.predictions.weekly;
            let maxVal = Math.max(...Object.values(weeks),1);
            document.getElementById('forecastBars').innerHTML = Object.entries(weeks).map(([w,v])=>`<div class="forecast-row"><span class="forecast-week">${w}</span><div class="forecast-bar-track"><div class="forecast-bar-fill" style="width:${(v/maxVal*100)}%"></div></div><span>${fmt(v)}</span></div>`).join('');
            document.getElementById('adviceList').innerHTML = pred.advice.map(a=>`<div class="advice-item"><strong>${a.cat}:</strong> ${a.msg}</div>`).join('');
            if(monthlyChart) monthlyChart.destroy();
            let months = Object.keys(summary.monthly).slice(-6);
            monthlyChart = new Chart(document.getElementById('monthlyChart'), { type:'bar', data:{ labels:months, datasets:[{ label:'Income', data:months.map(m=>summary.monthly[m]?.income||0), backgroundColor:'rgba(0,210,130,0.6)' },{ label:'Expense', data:months.map(m=>summary.monthly[m]?.expense||0), backgroundColor:'rgba(255,77,109,0.6)' }] } });
        }
        renderTransactions();
        loadAnalytics();
        updateLastExpenses();
        updateMLDiagram();
        await loadBudgetsForAdd();
        await loadBudgetsStandalone();
    } catch(e) { toast('Error loading data'); console.error(e); }
}
function renderTransactions() { document.getElementById('txTableBody').innerHTML = allTransactions.slice(0,50).map(t=>`<tr><td>${new Date(t.tx_date).toLocaleDateString()}</td><td>${t.category}</td><td>${t.is_need ? 'Need' : 'Want'}</td><td>${PRIO_MAP[t.priority] || ''}</td><td>${t.note||''}</td><td>${t.tx_type}</td><td>${fmt(t.amount)}</td></tr>`).join(''); }
async function loadAnalytics() {
    if(!currentUser) return;
    try {
        let longevity = await apiFetch(`/api/longevity/${currentUser.id}`);
        document.getElementById('longevityContainer').innerHTML = `<div>💰 Balance: ${fmt(longevity.balance)}<br>📉 Avg Daily: ${fmt(longevity.avg_daily_spend)}<br>📅 Days left: ${longevity.days}</div>`;
    } catch(e) {}
    if(currentPrediction.has_data) {
        if(forecastChart) forecastChart.destroy();
        forecastChart = new Chart(document.getElementById('forecastChart'), { type:'line', data:{ labels:Object.keys(currentPrediction.predictions.weekly), datasets:[{ label:'ML Forecast', data:Object.values(currentPrediction.predictions.weekly), borderColor:'#00d282' }] } });
        if(catBarChart) catBarChart.destroy();
        catBarChart = new Chart(document.getElementById('catBarChart'), { type:'bar', data:{ labels:Object.keys(currentPrediction.predictions.categories), datasets:[{ label:'Spent', data:Object.values(currentPrediction.predictions.categories), backgroundColor:'#3b82f6' }] }, options:{ indexAxis:'y' } });
    }
}

function navigate(screenId) {
    document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
    document.getElementById(`screen-${screenId}`).classList.add('active');
    document.getElementById('pageTitle').innerText = screenId.charAt(0).toUpperCase()+screenId.slice(1);
    if(screenId === 'analytics') loadAnalytics();
    if(screenId === 'add') { updateMLDiagram(); loadBudgetsForAdd(); renderCategoryAllocation(); updateAllocationFromCategories(); renderFutureExpenses(); validateAllocation(); updateLastExpenses(); }
    if(screenId === 'budgets') loadBudgetsStandalone();
}
document.querySelectorAll('.nav-item').forEach(btn=>btn.addEventListener('click',()=>{ let scr = btn.dataset.nav; if(scr) navigate(scr); }));
document.getElementById('exportBtn').addEventListener('click', ()=> window.location.href='/api/export/csv');
document.getElementById('signoutBtn').addEventListener('click', async()=>{ await fetch('/api/logout',{method:'POST',credentials:'include'}); location.reload(); });
document.getElementById('socialStatusDiagram')?.addEventListener('change', updateMLDiagram);
document.getElementById('mindsetDiagram')?.addEventListener('change', updateMLDiagram);
document.getElementById('budgetCycle')?.addEventListener('change', updateMLDiagram);
document.getElementById('monthlyBudgetCap')?.addEventListener('input', updateMLDiagram);
document.getElementById('autoRemainingToWants')?.addEventListener('change', updateAllocationFromCategories);
document.getElementById('aiAutoAllocateBtn')?.addEventListener('click', aiAutoAllocate);

let authBtn = document.getElementById('authBtn'), toggleLink = document.getElementById('toggleAuthLink');
toggleLink?.addEventListener('click', ()=>{ isLogin = !isLogin; document.getElementById('authTitle').innerText = isLogin ? 'Welcome back' : 'Create account'; document.getElementById('regName').style.display = isLogin ? 'none' : 'block'; document.getElementById('authConfirm').style.display = isLogin ? 'none' : 'block'; document.getElementById('termsRow').style.display = isLogin ? 'none' : 'flex'; authBtn.innerText = isLogin ? 'Sign In' : 'Register'; });
authBtn?.addEventListener('click', async()=>{ let email = document.getElementById('authEmail').value, pass = document.getElementById('authPass').value, name = document.getElementById('regName').value; if(!isLogin && (!name || !document.getElementById('termsCheck').checked)) { document.getElementById('authMsg').innerText = 'Accept terms & name required'; return; } let endpoint = isLogin ? '/api/login' : '/api/register'; let body = isLogin ? { email, password:pass } : { name, email, password:pass }; try { let res = await fetch(endpoint, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), credentials:'include' }); if(res.ok) { location.reload(); } else { let err = await res.json(); document.getElementById('authMsg').innerText = err.error || 'Auth failed'; } } catch(e){ document.getElementById('authMsg').innerText = 'Error connecting'; } });
document.getElementById('sendChatBtn')?.addEventListener('click', async()=>{ let input = document.getElementById('chatInput'); let msg = input.value.trim(); if(!msg) return; let chatDiv = document.getElementById('chatMessages'); chatDiv.innerHTML += `<div class="message user-message">${escapeHtml(msg)}</div>`; input.value = ''; chatDiv.scrollTop = chatDiv.scrollHeight; try { let res = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ message:msg }), credentials:'include' }); let data = await res.json(); let reply = data.reply || "AI: I'll analyze."; chatDiv.innerHTML += `<div class="message bot-message">${escapeHtml(reply)}</div>`; } catch(e) { chatDiv.innerHTML += `<div class="message bot-message">💬 Error, try again later.</div>`; } chatDiv.scrollTop = chatDiv.scrollHeight; });
function escapeHtml(str) { return str.replace(/[&<>]/g, function(m){ if(m === '&') return '&amp;'; if(m === '<') return '&lt;'; if(m === '>') return '&gt;'; return m;}); }
document.getElementById('closeChatBtn')?.addEventListener('click',()=>{ document.getElementById('chatContainer').style.display = 'none'; });
async function init() { let res = await fetch('/api/me', { credentials:'include' }); if(res.ok) { currentUser = await res.json(); document.getElementById('authOverlay').style.display = 'none'; document.getElementById('userAvatar').innerText = currentUser.name.slice(0,2).toUpperCase(); renderCategoryAllocation(); await loadDashboard(); loadFutureExpenses(); } else { document.getElementById('authOverlay').style.display = 'flex'; } }
init();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True, port=5000)
