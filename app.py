import json
import csv
import io
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
        return bcrypt.check_password_hash(self.password_hash, password)

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
# ML / Prediction helpers (unchanged)
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
# Routes (unchanged)
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
    return jsonify(tx.to_dict()), 201

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
    message = data.get('message', '').lower()
    user_id = user.id
    if 'health' in message or 'score' in message:
        score = compute_health_score(user_id)
        return jsonify({'reply': f"Your financial health score is {score}/100."})
    if 'forecast' in message:
        forecast = generate_weekly_forecast(user_id)
        reply = "📈 Spending forecast for next 4 weeks:\n" + "\n".join([f"{k}: ₱{v:,.2f}" for k, v in forecast.items()])
        return jsonify({'reply': reply})
    if 'investment' in message:
        return jsonify({'reply': "Consider putting 20% of monthly surplus into low-cost index funds."})
    if 'needs vs wants' in message:
        expenses = Transaction.query.filter_by(user_id=user_id, tx_type='expense').all()
        wants = sum(e.amount for e in expenses if not e.is_need)
        needs = sum(e.amount for e in expenses if e.is_need)
        total = wants + needs
        if total > 0:
            reply = f"Needs: ₱{needs:,.2f} ({needs/total*100:.0f}%), Wants: ₱{wants:,.2f} ({wants/total*100:.0f}%)."
        else:
            reply = "No expense data yet."
        return jsonify({'reply': reply})
    return jsonify({'reply': "Ask about: health score, forecast, investment, needs vs wants, priority."})

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
# Embedded HTML (full frontend)
# ----------------------------------------------------------------------
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
... (your full HTML code here - same as provided) ...
</html>'''

# ----------------------------------------------------------------------
# Schema migration: add missing columns if they don't exist
# ----------------------------------------------------------------------
def ensure_schema():
    """Add missing columns to existing tables without dropping data."""
    inspector = inspect(db.engine)
    
    # Check users table
    if inspector.has_table('users'):
        existing_columns = [col['name'] for col in inspector.get_columns('users')]
        if 'password_hash' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN password_hash VARCHAR(128) NOT NULL DEFAULT \'\''))
                conn.commit()
            print("✅ Added password_hash column to users table.")
        if 'social_status' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN social_status VARCHAR(20) DEFAULT \'Middle\''))
                conn.commit()
            print("✅ Added social_status column to users table.")
        if 'spending_mindset' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN spending_mindset VARCHAR(20) DEFAULT \'Neutral\''))
                conn.commit()
            print("✅ Added spending_mindset column to users table.")
        if 'monthly_budget_limit' not in existing_columns:
            with db.engine.connect() as conn:
                conn.execute(text('ALTER TABLE users ADD COLUMN monthly_budget_limit FLOAT DEFAULT 0.0'))
                conn.commit()
            print("✅ Added monthly_budget_limit column to users table.")
    
    # Ensure other tables exist (create_all will handle, but we also add missing columns if needed)
    # For simplicity, we rely on db.create_all() to create missing tables.
    # But for columns added later to existing tables, you'd need more ALTER statements.
    # These are the main ones for now.

# ----------------------------------------------------------------------
# Create tables and run migration
# ----------------------------------------------------------------------
with app.app_context():
    db.create_all()          # creates tables if they don't exist
    ensure_schema()          # adds missing columns to existing tables

if __name__ == '__main__':
    app.run(debug=True, port=5000)
