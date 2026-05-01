import os
import datetime
import hashlib
import secrets
import json
import csv
import io
from functools import wraps
from flask import Flask, request, jsonify, session, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)

# === CONFIGURATION ===
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False

default_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smartspend.db')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{default_db_path}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ========== MODELS ==========
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    tx_type = db.Column(db.String(10), nullable=False)       # 'income' or 'expense'
    is_need = db.Column(db.Boolean, default=False, nullable=False)
    priority = db.Column(db.Integer, default=0, nullable=False)
    note = db.Column(db.String(200))
    tx_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Budget(db.Model):
    __tablename__ = 'budgets'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    limit_amount = db.Column(db.Float, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'category'),)

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    social_status = db.Column(db.String(20), nullable=False, default='Middle')
    spending_mindset = db.Column(db.String(20), nullable=False, default='Neutral')
    wants_needs_json = db.Column(db.Text, default='{}')

# ========== CREATE TABLES & FALLBACK MIGRATIONS ==========
with app.app_context():
    db.create_all()
    # Ensure is_need and priority columns exist (for older dbs)
    for col in ['is_need', 'priority']:
        try:
            db.session.execute(f'ALTER TABLE transactions ADD COLUMN {col} {"BOOLEAN DEFAULT 0" if col=="is_need" else "INTEGER DEFAULT 0"}')
            db.session.commit()
        except Exception:
            pass

    # Create default admin
    if not User.query.filter_by(email='admin@smartspend.com').first():
        hashed = hashlib.sha256('admin123'.encode()).hexdigest()
        admin = User(name='Admin', email='admin@smartspend.com', password=hashed, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin created: admin@smartspend.com / admin123")

# ========== HELPERS ==========
def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

def get_user_profile(user_id):
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        return {'social_status': 'Middle', 'spending_mindset': 'Neutral', 'wants_needs': {}}
    return {
        'social_status': profile.social_status,
        'spending_mindset': profile.spending_mindset,
        'wants_needs': json.loads(profile.wants_needs_json) if profile.wants_needs_json else {}
    }

def _prio_label(p):
    return {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}.get(p, 'Low')

def _priority_penalty(priority):
    return {0: 2.0, 1: 1.5, 2: 0.8, 3: 0.3}.get(priority, 1.5)

def calculate_health_score(transactions, budgets):
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    expense = sum(t['amount'] for t in transactions if t['tx_type'] == 'expense')
    if income == 0:
        return 0
    savings_rate = max(0, min(1, (income - expense) / income))
    score = savings_rate * 100

    cat_spending = {}
    for t in transactions:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']

    budget_penalty = 0
    for cat, limit in budgets.items():
        spent = cat_spending.get(cat, 0)
        if limit > 0 and spent > limit:
            over_ratio = (spent - limit) / limit
            cat_priorities = [t.get('priority', 0) for t in transactions if t['category'] == cat and t['tx_type'] == 'expense']
            worst_priority = min(cat_priorities) if cat_priorities else 0
            budget_penalty += over_ratio * 10 * _priority_penalty(worst_priority)
    if budgets:
        score = max(0, min(100, score - budget_penalty))
    return int(score)

def forecast_spending(transactions):
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    if not expenses:
        return {"Week 1": 0, "Week 2": 0, "Week 3": 0, "Week 4": 0}
    weekly = {}
    for t in expenses:
        date = t['tx_date']
        if isinstance(date, str):
            date = datetime.datetime.fromisoformat(date)
        week_num = date.isocalendar()[1]
        year = date.year
        key = f"{year}-W{week_num}"
        weekly[key] = weekly.get(key, 0) + t['amount']
    weekly_vals = list(weekly.values())
    if len(weekly_vals) < 2:
        avg = sum(weekly_vals) / max(1, len(weekly_vals))
    else:
        window = weekly_vals[-3:] if len(weekly_vals) >= 3 else weekly_vals
        avg = sum(window) / len(window)
    forecast = {}
    for i in range(1, 5):
        forecast[f"Week {i}"] = round(avg * (0.95 + i * 0.02), 2)
    return forecast

def generate_advice(transactions, budgets, user_profile):
    advice = []
    cat_spending = {}
    for t in transactions:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    for cat, limit in budgets.items():
        spent = cat_spending.get(cat, 0)
        if limit > 0:
            if spent > limit:
                cat_prios = [t.get('priority', 0) for t in transactions if t['category'] == cat and t['tx_type'] == 'expense']
                worst_prio = min(cat_prios) if cat_prios else 0
                advice.append({
                    "cat": cat,
                    "msg": f"⚠️ Overspent by ₱{spent-limit:.2f} (Priority: {_prio_label(worst_prio)}). Reduce or adjust budget."
                })
            elif spent < limit * 0.7:
                advice.append({"cat": cat, "msg": f"✅ Great! Underspent by ₱{limit-spent:.2f}. Consider saving."})
            else:
                advice.append({"cat": cat, "msg": f"✔️ On track. Spent ₱{spent:.2f} of ₱{limit:.2f} budget."})
    if not advice:
        advice.append({"cat": "General", "msg": "No budget limits set. Set budgets to get personalized advice."})

    mindset = user_profile.get('spending_mindset', 'Neutral')
    if mindset == 'Saver' and cat_spending:
        total = sum(cat_spending.values())
        if total > 0:
            advice.append({"cat": "Mindset", "msg": f"🧠 Saver mode. Aim for {min(30, int((total/5000)*10))}% of surplus to investments."})
    elif mindset == 'Spender':
        advice.append({"cat": "Mindset", "msg": "💸 Spender alert! Use priority to rank expenses before spending."})
    return advice

def generate_scenarios(user_id, transactions, budgets):
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    total_expense = sum(e['amount'] for e in expenses)
    savings = max(0, income - total_expense)
    profile = get_user_profile(user_id)
    social = profile['social_status']
    mindset = profile['spending_mindset']
    cat_spend = {}
    need_spend = {}
    for e in expenses:
        cat = e['category']
        amt = e['amount']
        cat_spend[cat] = cat_spend.get(cat, 0) + amt
        if e.get('is_need', False):
            need_spend[cat] = need_spend.get(cat, 0) + amt

    default_cats = ['Food & Dining', 'Transport', 'Groceries', 'Entertainment', 'Health']

    def build_scenario(name, savings_mult, invest_mult, essential_mult, disc_mult):
        limits = {}
        for cat in default_cats:
            spent = cat_spend.get(cat, 0)
            if cat in need_spend:
                multiplier = essential_mult
            else:
                multiplier = disc_mult
            if cat in ['Food & Dining', 'Groceries', 'Health']:
                suggested = max(spent * multiplier, 2000) if spent > 0 else 3000
            else:
                suggested = max(spent * multiplier, 1000) if spent > 0 else 2000
            limits[cat] = round(suggested)

        if social == 'Low':
            limits['Entertainment'] = limits.get('Entertainment', 1500) * 0.6
            limits['Transport'] = limits.get('Transport', 2000) * 0.8
        elif social == 'Upper':
            limits['Entertainment'] = limits.get('Entertainment', 3000) * 1.5
            limits['Health'] = limits.get('Health', 3000) * 1.3
        if mindset == 'Saver':
            for cat in ['Entertainment', 'Transport']:
                limits[cat] = limits.get(cat, 2000) * 0.7
        elif mindset == 'Spender':
            for cat in ['Entertainment', 'Food & Dining']:
                limits[cat] = limits.get(cat, 3000) * 1.4

        for cat in limits:
            limits[cat] = max(100, int(limits[cat]))

        suggested_savings = income * savings_mult if income > 0 else savings * savings_mult
        suggested_investment = suggested_savings * invest_mult
        return {
            'name': name,
            'budget_limits': limits,
            'savings_goal': round(suggested_savings),
            'investment_goal': round(suggested_investment),
            'expected_savings_rate': round(savings_mult * 100)
        }

    cons = build_scenario('Conservative (Safe)', 0.25, 0.6, 1.2, 0.7)
    bal = build_scenario('Balanced (Moderate)', 0.20, 0.7, 1.0, 1.0)
    agg = build_scenario('Aggressive (Growth)', 0.15, 0.9, 0.9, 1.3)
    return [cons, bal, agg]

# ========== API ROUTES ==========
@app.route('/')
def index():
    return HTML_PAGE

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not data or not data.get('email') or not data.get('password') or not data.get('name'):
        return jsonify({'error': 'Missing fields'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    hashed = hash_password(data['password'])
    user = User(name=data['name'], email=data['email'], password=hashed, role='user')
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data.get('email')).first()
    if not user or user.password != hash_password(data.get('password', '')):
        return jsonify({'error': 'Invalid credentials'}), 401
    session['user_id'] = user.id
    session['user_name'] = user.name
    return jsonify({'user': {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role}})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'})

@app.route('/api/me')
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 401
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role})

@app.route('/api/user/profile', methods=['GET', 'POST'])
@login_required
def user_profile():
    user_id = session['user_id']
    if request.method == 'GET':
        profile = UserProfile.query.filter_by(user_id=user_id).first()
        if not profile:
            return jsonify({'social_status': 'Middle', 'spending_mindset': 'Neutral', 'wants_needs': {}})
        return jsonify({'social_status': profile.social_status, 'spending_mindset': profile.spending_mindset, 'wants_needs': json.loads(profile.wants_needs_json) if profile.wants_needs_json else {}})
    else:
        data = request.json
        profile = UserProfile.query.filter_by(user_id=user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.session.add(profile)
        profile.social_status = data.get('social_status', 'Middle')
        profile.spending_mindset = data.get('spending_mindset', 'Neutral')
        profile.wants_needs_json = json.dumps(data.get('wants_needs', {}))
        db.session.commit()
        return jsonify({'message': 'Profile updated'})

# Unified helper to save a single budget
def _save_budget(user_id, category, limit):
    budget = Budget.query.filter_by(user_id=user_id, category=category).first()
    if budget:
        budget.limit_amount = limit
    else:
        budget = Budget(user_id=user_id, category=category, limit_amount=limit)
        db.session.add(budget)
    db.session.commit()

@app.route('/api/budgets/<int:user_id>', methods=['GET', 'POST'])
@login_required
def manage_budgets(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    if request.method == 'GET':
        budgets = Budget.query.filter_by(user_id=user_id).all()
        return jsonify([{'category': b.category, 'limit': b.limit_amount} for b in budgets])
    else:
        data = request.json
        category = data.get('category')
        limit = data.get('limit')
        if not category or limit is None:
            return jsonify({'error': 'Category and limit required'}), 400
        _save_budget(user_id, category, limit)
        return jsonify({'message': 'Budget saved'})

@app.route('/api/budgets/bulk/<int:user_id>', methods=['POST'])
@login_required
def bulk_budgets(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.json
    limits = data.get('limits', {})
    for category, limit_amount in limits.items():
        _save_budget(user_id, category, limit_amount)
    return jsonify({'message': 'Budgets applied'})

@app.route('/api/transactions', methods=['GET', 'POST'])
@login_required
def transactions():
    user_id = session['user_id']
    if request.method == 'GET':
        txns = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.tx_date.desc()).all()
        return jsonify([{
            'id': t.id, 'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type,
            'is_need': t.is_need, 'priority': t.priority,
            'note': t.note, 'tx_date': t.tx_date.isoformat()
        } for t in txns])
    else:
        data = request.json
        if not data or 'amount' not in data or 'category' not in data or 'tx_type' not in data:
            return jsonify({'error': 'Invalid transaction data'}), 400
        is_need = data.get('is_need', False) if data['tx_type'] == 'expense' else False
        priority = data.get('priority', 0) if data['tx_type'] == 'expense' else 0
        txn = Transaction(
            user_id=user_id,
            amount=data['amount'],
            category=data['category'],
            tx_type=data['tx_type'],
            is_need=is_need,
            priority=priority,
            note=data.get('note', ''),
            tx_date=datetime.datetime.utcnow()
        )
        db.session.add(txn)
        db.session.commit()
        return jsonify({'message': 'Transaction added', 'id': txn.id})

@app.route('/api/transactions/<int:txn_id>', methods=['PATCH'])
@login_required
def update_transaction(txn_id):
    txn = Transaction.query.get_or_404(txn_id)
    if txn.user_id != session['user_id']:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.json
    if 'is_need' in data:
        txn.is_need = data['is_need']
    if 'priority' in data:
        txn.priority = data['priority']
    db.session.commit()
    return jsonify({'message': 'Transaction updated'})

@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    txns = Transaction.query.filter_by(user_id=user_id).all()
    balance = 0
    expense = 0
    income = 0
    monthly = {}
    for t in txns:
        if t.tx_type == 'income':
            balance += t.amount
            income += t.amount
        else:
            balance -= t.amount
            expense += t.amount
        month_key = t.tx_date.strftime('%Y-%m')
        if month_key not in monthly:
            monthly[month_key] = {'income': 0, 'expense': 0}
        if t.tx_type == 'income':
            monthly[month_key]['income'] += t.amount
        else:
            monthly[month_key]['expense'] += t.amount
    return jsonify({'balance': balance, 'expense': expense, 'income': income, 'monthly': monthly})

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    txns = Transaction.query.filter_by(user_id=user_id).all()
    if not txns:
        return jsonify({'has_data': False, 'score': None, 'predictions': {}, 'advice': []})
    tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type,
                'is_need': t.is_need, 'priority': t.priority, 'tx_date': t.tx_date} for t in txns]
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.limit_amount for b in budgets}
    profile = get_user_profile(user_id)
    health_score = calculate_health_score(tx_list, budget_dict)
    weekly_forecast = forecast_spending(tx_list)
    cat_spending = {}
    for t in tx_list:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    advice = generate_advice(tx_list, budget_dict, profile)
    return jsonify({
        'has_data': True,
        'score': health_score,
        'predictions': {'weekly': weekly_forecast, 'categories': cat_spending},
        'advice': advice
    })

@app.route('/api/budget/scenarios/<int:user_id>')
@login_required
def budget_scenarios(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    txns = Transaction.query.filter_by(user_id=user_id).all()
    if not txns:
        return jsonify({'scenarios': []})
    tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type,
                'is_need': t.is_need, 'priority': t.priority, 'tx_date': t.tx_date} for t in txns]
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.limit_amount for b in budgets}
    scenarios = generate_scenarios(user_id, tx_list, budget_dict)
    return jsonify({'scenarios': scenarios})

@app.route('/api/longevity/<int:user_id>')
@login_required
def budget_longevity(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    txns = Transaction.query.filter_by(user_id=user_id).all()
    balance = 0
    total_expense = 0
    for t in txns:
        if t.tx_type == 'income':
            balance += t.amount
        else:
            balance -= t.amount
            total_expense += t.amount
    if total_expense == 0:
        return jsonify({'error': 'No expense data yet'}), 400
    now = datetime.datetime.utcnow()
    thirty_days_ago = now - datetime.timedelta(days=30)
    recent_expenses = [t for t in txns if t.tx_type == 'expense' and t.tx_date >= thirty_days_ago]
    if recent_expenses:
        avg_daily = sum(e.amount for e in recent_expenses) / 30.0
    else:
        avg_daily = total_expense / 30.0
    if avg_daily <= 0:
        return jsonify({'error': 'No spending to project'}), 400
    days_left = balance / avg_daily if avg_daily > 0 else 0
    return jsonify({
        'balance': balance,
        'avg_daily_spend': round(avg_daily, 2),
        'days': round(max(0, days_left), 1)
    })

@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.json
    user_msg = data.get('message', '').lower()
    user_id = session['user_id']

    txns = Transaction.query.filter_by(user_id=user_id).all()
    if not txns:
        return jsonify({'response': "You don't have any transactions yet. Add some to get personalized advice!"})

    income = sum(t.amount for t in txns if t.tx_type == 'income')
    expense = sum(t.amount for t in txns if t.tx_type == 'expense')
    savings = income - expense
    savings_rate = (savings / income * 100) if income > 0 else 0

    if 'health score' in user_msg or 'score' in user_msg:
        budgets = Budget.query.filter_by(user_id=user_id).all()
        budget_dict = {b.category: b.limit_amount for b in budgets}
        tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'is_need': t.is_need, 'priority': t.priority} for t in txns]
        score = calculate_health_score(tx_list, budget_dict)
        return jsonify({'response': f"ML Health Score: {score}/100. { 'Great job!' if score >= 70 else 'Work on reducing low-priority expenses.' }"})

    elif 'forecast' in user_msg or 'future' in user_msg:
        tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'tx_date': t.tx_date} for t in txns]
        forecast = forecast_spending(tx_list)
        weeks = ', '.join([f"{k}: ₱{v:,.2f}" for k,v in forecast.items()])
        return jsonify({'response': f"ML forecast: {weeks}. Prioritize high-priority needs this month."})

    elif 'invest' in user_msg or 'investment' in user_msg:
        return jsonify({'response': f"With {savings_rate:.1f}% savings rate, consider ₱{max(500, int(savings*0.2)):,.0f} in index funds. High priority spending should be covered first."})

    elif 'needs' in user_msg or 'wants' in user_msg:
        needs = sum(t.amount for t in txns if t.tx_type == 'expense' and t.is_need)
        wants = expense - needs
        return jsonify({'response': f"Needs: ₱{needs:,.2f}, Wants: ₱{wants:,.2f}. Set priorities to auto‑classify needs."})

    elif 'priority' in user_msg or 'ranking' in user_msg:
        prios = {}
        for t in txns:
            if t.tx_type == 'expense':
                p = t.priority
                prios.setdefault(p, 0)
                prios[p] += t.amount
        msg = "Spending by priority: " + ", ".join([f"{_prio_label(p)}: ₱{amt:,.2f}" for p, amt in sorted(prios.items())])
        return jsonify({'response': msg + ". Higher priority = more essential. Cut low priority first."})

    else:
        return jsonify({'response': f"I can help with: 'health score', 'forecast', 'investment advice', 'needs vs wants', 'priority / ranking'. Your savings rate: {savings_rate:.1f}%."})

@app.route('/api/export/csv')
@login_required
def export_csv():
    user_id = session['user_id']
    txns = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.tx_date.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Category', 'Type', 'Amount', 'Need/Want', 'Priority', 'Note'])
    for t in txns:
        writer.writerow([t.tx_date.strftime('%Y-%m-%d'), t.category, t.tx_type, t.amount,
                         'Need' if t.is_need else 'Want', _prio_label(t.priority), t.note])
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=smartspend_export.csv'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# ========== FRONTEND (Embedded, fully auto‑save, no manual buttons) ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>SmartSpend · AI-Powered Finance</title>
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
        body { font-family: var(--sans); background: var(--bg); color: var(--text); min-height: 100vh; }
        .sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 80px; background: rgba(15, 22, 34, 0.98); backdrop-filter: blur(12px); border-right: 1px solid var(--border2); display: flex; flex-direction: column; align-items: center; padding: 24px 0; gap: 10px; z-index: 100; transition: width 0.3s; }
        .sidebar:hover { width: 240px; }
        .sidebar-logo { width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, var(--green), #009e5f); display: flex; align-items: center; justify-content: center; margin-bottom: 28px; cursor: pointer; font-size: 1.5rem; }
        .nav-item { width: 100%; display: flex; align-items: center; gap: 16px; padding: 12px 24px; border-radius: 12px; cursor: pointer; border: none; background: transparent; color: var(--muted); font-size: 0.9rem; font-weight: 500; white-space: nowrap; overflow: hidden; transition: all 0.2s; }
        .nav-item:hover { background: var(--green-dim); color: var(--text); transform: translateX(6px); }
        .nav-item.active { background: var(--green-dim); color: var(--green); border-left: 3px solid var(--green); }
        .nav-label { opacity: 0; transition: opacity 0.2s; }
        .sidebar:hover .nav-label { opacity: 1; }
        .sidebar-bottom { margin-top: auto; width: 100%; }
        .main { margin-left: 80px; padding: 24px 32px; position: relative; min-height: 100vh; }
        @media (max-width: 768px) { .sidebar { width: 70px; } .main { margin-left: 70px; padding: 20px; } }
        @media (max-width: 600px) { .sidebar { display: none; } .main { margin-left: 0; } }
        .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; flex-wrap: wrap; gap: 16px; }
        .topbar h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #fff, var(--green)); background-clip: text; -webkit-background-clip: text; color: transparent; }
        .topbar-right { display: flex; align-items: center; gap: 16px; }
        .score-pill { display: flex; align-items: center; gap: 10px; padding: 8px 20px; border-radius: 99px; background: var(--green-dim); border: 1px solid var(--border); font-size: 0.85rem; font-weight: 600; transition: all 0.3s; }
        .score-pill.green { border-color: var(--green); color: var(--green); }
        .score-pill.red { border-color: var(--red); color: var(--red); }
        .score-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
        .score-pill.red .score-dot { background: var(--red); }
        @keyframes pulse { 0%,100%{ opacity:1; } 50%{ opacity:0.4; } }
        .avatar { width: 42px; height: 42px; border-radius: 50%; background: linear-gradient(135deg, var(--blue), var(--purple)); display: flex; align-items: center; justify-content: center; font-weight: 700; cursor: pointer; }
        .signout-btn { background: var(--red-dim); border: 1px solid rgba(255, 77, 109, 0.2); color: var(--red); padding: 8px 18px; border-radius: 10px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
        .screen { display: none; animation: fadeUp 0.35s ease; }
        .screen.active { display: block; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .stat-card { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; transition: all 0.25s; cursor: pointer; }
        .stat-value { font-size: 1.8rem; font-weight: 700; font-family: var(--mono); }
        .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 8px; }
        .panel { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; margin-bottom: 20px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
        .panel-title { font-size: 1rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
        input, select, .btn { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); border-radius: 10px; padding: 10px 14px; font-size: 0.85rem; outline: none; transition: all 0.2s; }
        input:focus, select:focus { border-color: var(--green); box-shadow: 0 0 0 2px var(--green-dim); }
        .btn { cursor: pointer; font-weight: 600; }
        .btn-green { background: linear-gradient(135deg, var(--green), #009e5f); color: #000; border: none; }
        .btn-green:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 210, 130, 0.3); }
        .flex-form { display: flex; flex-direction: column; gap: 16px; }
        .row-2cols { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        @media (max-width: 700px) { .row-2cols { grid-template-columns: 1fr; } }
        .allocation-box { background: var(--bg3); border-radius: 12px; padding: 16px; margin: 12px 0; }
        .forecast-scenarios { display: flex; justify-content: space-between; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
        .scenario-card { background: var(--bg3); border-radius: 12px; padding: 12px; flex: 1; text-align: center; }
        .pref-checkbox { margin: 8px 0; display: flex; align-items: center; gap: 10px; }
        .ml-badge { background: linear-gradient(135deg, var(--purple), var(--blue)); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 8px; }
        .table-wrap { overflow-x: auto; border-radius: var(--r); background: var(--panel); border: 1px solid var(--border2); }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 14px 16px; background: var(--bg3); color: var(--muted); font-size: 0.7rem; text-transform: uppercase; }
        td { padding: 12px 16px; border-bottom: 1px solid var(--border2); font-size: 0.85rem; }
        .advice-item { padding: 14px; border-radius: 12px; margin-bottom: 10px; background: var(--bg3); transition: all 0.2s; }
        .forecast-row { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
        .forecast-week { width: 70px; font-weight: 600; color: var(--green); }
        .forecast-bar-track { flex: 1; height: 8px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
        .forecast-bar-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--green2)); width: 0; border-radius: 99px; transition: width 1s ease; }
        #toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 1px solid var(--green); border-radius: 12px; padding: 12px 24px; opacity: 0; transform: translateY(10px); transition: all 0.25s; z-index: 10000; backdrop-filter: blur(12px); }
        #toast.show { opacity: 1; transform: translateY(0); }
        .auth-overlay { position: fixed; inset: 0; background: rgba(8,13,20,0.98); backdrop-filter: blur(20px); z-index: 9999; display: flex; align-items: center; justify-content: center; }
        .auth-card { background: linear-gradient(145deg, #0d1520, #0a1220); border: 1px solid rgba(0,210,130,0.2); border-radius: 28px; padding: 40px; width: 400px; max-width: 90%; }
        .chat-container { position: fixed; bottom: 24px; right: 24px; z-index: 10001; cursor: move; }
        .chat-window { width: 380px; height: 480px; background: var(--bg2); backdrop-filter: blur(10px); border: 1px solid var(--green); border-radius: 24px; display: flex; flex-direction: column; overflow: hidden; cursor: default; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
        .chat-header { padding: 14px 18px; background: var(--bg3); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; cursor: move; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 85%; padding: 10px 14px; border-radius: 18px; font-size: 0.85rem; animation: messagePop 0.3s ease; }
        .user-message { align-self: flex-end; background: var(--green-dim); color: var(--green); border-bottom-right-radius: 4px; }
        .bot-message { align-self: flex-start; background: var(--bg3); color: var(--text); border-bottom-left-radius: 4px; }
        .chat-input { display: flex; padding: 14px; gap: 10px; background: var(--bg3); border-top: 1px solid var(--border); }
        .budget-limit-panel { margin-top: 24px; border-top: 1px solid var(--border); padding-top: 20px; }
        .auth-input { width: 100%; margin-bottom: 16px; }
        .auth-btn { width: 100%; background: var(--green); color: #000; font-weight: bold; padding: 12px; border: none; border-radius: 20px; cursor: pointer; }
        .auth-link { text-align: center; color: var(--muted); font-size: 0.8rem; }
        .auto-badge { font-size: 0.6rem; background: var(--green-dim); border-radius: 12px; padding: 2px 6px; margin-left: 8px; color: var(--green);}
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

    <!-- ADD TRANSACTION SCREEN (auto-save budgets, no Save button) -->
    <div class="screen" id="screen-add">
        <div class="row-2cols">
            <div class="panel">
                <div class="panel-header">➕ Record Income</div>
                <form id="incomeForm" class="flex-form">
                    <input type="number" id="incomeAmount" placeholder="Amount (₱)" step="0.01" required>
                    <input type="text" id="incomeNote" placeholder="Note (optional)">
                    <button type="submit" class="btn-green">Add Income</button>
                </form>
                <div style="margin-top: 24px;"><div class="panel-header">📝 Record Expense</div>
                <form id="expenseForm" class="flex-form">
                    <input type="number" id="expenseAmount" placeholder="Amount (₱)" step="0.01" required>
                    <select id="expenseCategory">
                        <option>Food & Dining</option><option>Transport</option><option>Groceries</option>
                        <option>Entertainment</option><option>Health</option><option>Other</option>
                    </select>
                    <div style="display:flex; gap:10px; align-items:center;">
                        <label><input type="checkbox" id="isNeed"> ✅ NEED</label>
                        <label>Priority:
                            <select id="expensePriority">
                                <option value="0">Low</option><option value="1" selected>Medium</option>
                                <option value="2">High</option><option value="3">Critical</option>
                            </select>
                        </label>
                    </div>
                    <input type="text" id="expenseNote" placeholder="Note">
                    <button type="submit" class="btn-green">Record Expense</button>
                </form>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">🧠 ML Controlled Budget Allocation <span class="ml-badge">LIVE</span></div>
                <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px;">
                    <div style="flex:1;"><label>Monthly Budget (₱)</label><input type="number" id="monthlyBudgetCap" placeholder="Auto from income" step="500"></div>
                    <div style="flex:1;"><label>Social Status</label><select id="socialStatusDiagram"><option>Low</option><option selected>Middle</option><option>Upper</option></select></div>
                    <div style="flex:1;"><label>Mindset</label><select id="mindsetDiagram"><option>Saver</option><option selected>Neutral</option><option>Spender</option></select></div>
                    <div style="flex:1;"><label>Cycle</label><select id="budgetCycle"><option>Daily</option><option>Weekly</option><option selected>Monthly</option><option>Yearly</option></select></div>
                </div>
                <div class="allocation-box" id="allocationDisplay">
                    Needs: -- / Wants: -- / Savings: --
                </div>
                <div style="margin: 16px 0; padding: 12px; background: var(--bg3); border-radius: 12px;" id="lastExpenseBox">
                    <strong>📝 LAST EXPENSE</strong><br>
                    <span id="lastExpenseDesc">No expenses yet</span>
                    <div id="lastExpenseActions" style="margin-top: 8px;"></div>
                </div>
                <div>
                    <div class="panel-title">📈 FORECAST (next 7 days)</div>
                    <div class="forecast-scenarios" id="forecastScenarios">
                        <div class="scenario-card"><strong>🟢 Optimistic</strong><br><span id="forecastOpt">--</span></div>
                        <div class="scenario-card"><strong>🔵 Realistic</strong><br><span id="forecastReal">--</span></div>
                        <div class="scenario-card"><strong>🔴 Pessimistic</strong><br><span id="forecastPess">--</span></div>
                    </div>
                </div>
                <div style="margin-top: 16px;">
                    <div class="panel-title">⚙️ ML Preferences</div>
                    <label class="pref-checkbox"><input type="checkbox" id="prefAutoReallocate"> Allow ML to auto‑reallocate wants to needs</label>
                    <label class="pref-checkbox"><input type="checkbox" id="prefAlertWant"> Alert me before any want expense</label>
                    <label class="pref-checkbox"><input type="checkbox" id="prefDailyForecast" checked> Show daily forecast summary</label>
                </div>
            </div>
        </div>

        <!-- BUDGET LIMITS PANEL: fully automatic save on change -->
        <div class="panel budget-limit-panel">
            <div class="panel-header">💰 Budget Limits (by category) <span class="auto-badge">⚡ auto‑saves on change</span></div>
            <div id="budgetInputsAdd" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px;"></div>
        </div>
    </div>

    <!-- ANALYTICS -->
    <div class="screen" id="screen-analytics">
        <div class="panel"><div class="panel-header">⏳ Budget Longevity</div><div id="longevityContainer">Loading...</div></div>
        <div class="panel"><div class="panel-header">Weekly Forecast</div><canvas id="forecastChart" height="200"></canvas></div>
        <div class="panel"><div class="panel-header">Category Breakdown</div><canvas id="catBarChart" height="200"></canvas></div>
    </div>

    <!-- BUDGETS SCREEN: also auto‑save, no manual Save button -->
    <div class="screen" id="screen-budgets">
        <div class="panel">
            <div class="panel-header">Set Monthly Budget Limits <span class="auto-badge">✏️ auto‑save on edit</span></div>
            <div id="budgetInputsStandalone" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px;"></div>
        </div>
    </div>

    <!-- TRANSACTIONS -->
    <div class="screen" id="screen-transactions">
        <div class="table-wrap"><table><thead><tr><th>Date</th><th>Category</th><th>Need?</th><th>Priority</th><th>Note</th><th>Type</th><th>Amount</th></tr></thead><tbody id="txTableBody"></tbody></table></div>
    </div>

    <!-- AUTH OVERLAY -->
    <div id="authOverlay" class="auth-overlay">
        <div class="auth-card">
            <h2 id="authTitle">Welcome back</h2>
            <input type="text" id="regName" class="auth-input" placeholder="Full Name" style="display:none">
            <input type="email" id="authEmail" class="auth-input" placeholder="Email">
            <input type="password" id="authPass" class="auth-input" placeholder="Password">
            <input type="password" id="authConfirm" class="auth-input" placeholder="Confirm Password" style="display:none">
            <div id="termsRow" style="display:none;"><label><input type="checkbox" id="termsCheck"> Accept Terms</label></div>
            <div id="authMsg" style="color:#ff4d6d;"></div>
            <button id="authBtn" class="auth-btn">Sign In</button>
            <div id="toggleAuthLink" class="auth-link" style="cursor:pointer; margin-top:12px;">Don't have an account? Register</div>
        </div>
    </div>

    <!-- CHATBOT -->
    <div id="chatContainer" class="chat-container">
        <div class="chat-window">
            <div class="chat-header" id="chatHeader">🤖 SmartSpend AI <span class="ml-badge">ML</span><button id="closeChatBtn">✕</button></div>
            <div class="chat-messages" id="chatMessages"><div class="message bot-message">✨ Ask: 'health score', 'forecast', 'investment', 'needs vs wants', 'priority'.</div></div>
            <div class="chat-input"><input id="chatInput" placeholder="Ask..."><button id="sendChatBtn">Send</button></div>
        </div>
    </div>
</main>

<script>
let currentUser = null, allTransactions = [], currentSummary = { balance:0, expense:0, income:0 }, currentPrediction = { has_data:false, score:null, predictions:{ weekly:{}, categories:{} }, advice:[] };
let monthlyChart, forecastChart, catBarChart, isLogin = true;
const PRIO_MAP = {'0':'Low','1':'Medium','2':'High','3':'Critical'};
function fmt(amt) { return '₱' + Number(amt).toLocaleString('en-PH', { minimumFractionDigits:2 }); }
function toast(msg) { let t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2500); }
async function apiFetch(url, opts={}) {
    let res = await fetch(url, {...opts, credentials:'include', headers:{'Content-Type':'application/json'}});
    if(!res.ok) throw new Error((await res.json()).error || 'Request failed');
    return res.json();
}

// --- Allocation & ML diagram (unchanged) ---
function getAllocation(budget, status, mindset) {
    let needsPct, wantsPct, savingsPct;
    if(status === 'Low') { needsPct=70; wantsPct=20; savingsPct=10; }
    else if(status === 'Middle') { needsPct=50; wantsPct=30; savingsPct=20; }
    else { needsPct=30; wantsPct=30; savingsPct=40; }
    if(mindset === 'Spender') { wantsPct += 10; needsPct -= 5; savingsPct -= 5; }
    else if(mindset === 'Saver') { savingsPct += 10; wantsPct -= 5; needsPct -= 5; }
    needsPct = Math.min(100, Math.max(0, needsPct));
    wantsPct = Math.min(100, Math.max(0, wantsPct));
    savingsPct = Math.min(100, Math.max(0, 100 - needsPct - wantsPct));
    let needsAmt = budget * needsPct / 100;
    let wantsAmt = budget * wantsPct / 100;
    let savingsAmt = budget * savingsPct / 100;
    return { needsPct, wantsPct, savingsPct, needsAmt, wantsAmt, savingsAmt };
}
function computeScenarios(budget, spentSoFar, daysElapsed, totalDays) {
    let remainingDays = Math.max(1, totalDays - daysElapsed);
    let avgDailySpent = spentSoFar / Math.max(1, daysElapsed);
    let realisticRemaining = avgDailySpent * remainingDays;
    let optimisticRemaining = (avgDailySpent * 0.8) * remainingDays;
    let pessimisticRemaining = (avgDailySpent * 1.2) * remainingDays;
    let remainingBudget = budget - spentSoFar;
    return { optimistic: Math.max(0, remainingBudget - optimisticRemaining), realistic: Math.max(0, remainingBudget - realisticRemaining), pessimistic: Math.max(0, remainingBudget - pessimisticRemaining) };
}
async function updateMLDiagram() {
    if(!currentUser) return;
    let txns = allTransactions;
    let incomeTotal = txns.filter(t=>t.tx_type==='income').reduce((s,t)=>s+t.amount,0);
    let monthlyBudget = parseFloat(document.getElementById('monthlyBudgetCap').value) || incomeTotal || 10000;
    let status = document.getElementById('socialStatusDiagram').value;
    let mindset = document.getElementById('mindsetDiagram').value;
    let cycle = document.getElementById('budgetCycle').value;
    let alloc = getAllocation(monthlyBudget, status, mindset);
    document.getElementById('allocationDisplay').innerHTML = `Needs (${alloc.needsPct}%): ${fmt(alloc.needsAmt)} 🔒 &nbsp;|&nbsp; Wants (${alloc.wantsPct}%): ${fmt(alloc.wantsAmt)} ⚡ &nbsp;|&nbsp; Savings (${alloc.savingsPct}%): ${fmt(alloc.savingsAmt)} 🏦`;
    let expenses = txns.filter(t=>t.tx_type==='expense').sort((a,b)=>new Date(b.tx_date)-new Date(a.tx_date));
    if(expenses.length) {
        let last = expenses[0];
        document.getElementById('lastExpenseDesc').innerHTML = `${last.category}: ${fmt(last.amount)} (${last.is_need ? '✅ Need' : '⚪ Want'})<br><small>${new Date(last.tx_date).toLocaleDateString()}</small>`;
        document.getElementById('lastExpenseActions').innerHTML = `<button class="btn" id="toggleNeedBtn" style="background:var(--green-dim);">Mark as ${last.is_need ? 'Want' : 'Need'}</button> <span class="ml-badge">ML suggests: ${last.is_need ? '✅ Keep as Need' : 'Consider moving to Need if essential'}</span>`;
        document.getElementById('toggleNeedBtn')?.addEventListener('click', async()=>{
            await apiFetch(`/api/transactions/${last.id}`, { method:'PATCH', body:JSON.stringify({ is_need: !last.is_need }) });
            toast('Updated! Reloading...');
            await loadDashboard();
            updateMLDiagram();
        });
    } else {
        document.getElementById('lastExpenseDesc').innerHTML = 'No expenses yet';
        document.getElementById('lastExpenseActions').innerHTML = '';
    }
    let now = new Date();
    let startOfCycle, totalDays;
    if(cycle === 'Daily') { startOfCycle = new Date(now.getFullYear(), now.getMonth(), now.getDate()); totalDays = 1; }
    else if(cycle === 'Weekly') { let day = now.getDay(); startOfCycle = new Date(now); startOfCycle.setDate(now.getDate() - day); totalDays = 7; }
    else if(cycle === 'Monthly') { startOfCycle = new Date(now.getFullYear(), now.getMonth(), 1); totalDays = new Date(now.getFullYear(), now.getMonth()+1, 0).getDate(); }
    else { startOfCycle = new Date(now.getFullYear(), 0, 1); totalDays = 366; }
    let spentThisCycle = txns.filter(t=>t.tx_type==='expense' && new Date(t.tx_date) >= startOfCycle).reduce((s,t)=>s+t.amount,0);
    let daysElapsed = Math.min(totalDays, Math.floor((now - startOfCycle) / (1000*60*60*24)));
    let scenarios = computeScenarios(monthlyBudget, spentThisCycle, daysElapsed, totalDays);
    document.getElementById('forecastOpt').innerHTML = fmt(scenarios.optimistic);
    document.getElementById('forecastReal').innerHTML = fmt(scenarios.realistic);
    document.getElementById('forecastPess').innerHTML = fmt(scenarios.pessimistic);
    await apiFetch('/api/user/profile', { method:'POST', body:JSON.stringify({ social_status:status, spending_mindset:mindset }) });
}

// --- AUTO-SAVE BUDGET (core) ---
let saveTimeout = null;
function autoSaveBudget(category, limitValue) {
    if(!currentUser) return;
    if(limitValue && !isNaN(parseFloat(limitValue)) && parseFloat(limitValue) > 0) {
        if(saveTimeout) clearTimeout(saveTimeout);
        saveTimeout = setTimeout(async () => {
            try {
                await apiFetch(`/api/budgets/${currentUser.id}`, { method:'POST', body:JSON.stringify({ category, limit: parseFloat(limitValue) }) });
                toast(`💾 Saved: ${category} → ${fmt(parseFloat(limitValue))}`);
            } catch(e) { console.warn("Auto-save failed", e); }
        }, 500);
    }
}
function attachAutoSaveToInputs(containerId, categories) {
    categories.forEach(cat => {
        let rawId = cat.replace(/\\s/g, '');
        let inputEl = document.getElementById(`${containerId}_${rawId}`);
        if(inputEl && !inputEl.hasAttribute('data-auto-save')) {
            inputEl.setAttribute('data-auto-save', 'true');
            inputEl.addEventListener('change', (e) => autoSaveBudget(cat, e.target.value));
            inputEl.addEventListener('blur', (e) => autoSaveBudget(cat, e.target.value));
        }
    });
}
async function loadBudgetInputs(containerId, targetDivId) {
    if(!currentUser) return;
    let budgets = await apiFetch(`/api/budgets/${currentUser.id}`);
    let limits = Object.fromEntries(budgets.map(b=>[b.category, b.limit]));
    let categories = ['Food & Dining','Transport','Groceries','Entertainment','Health','Other'];
    let html = categories.map(cat=>`<div><label>${cat}</label><input type="number" id="${containerId}_${cat.replace(/\\s/g,'')}" value="${limits[cat]||''}" placeholder="Auto-save limit"></div>`).join('');
    document.getElementById(targetDivId).innerHTML = html;
    attachAutoSaveToInputs(containerId, categories);
}

// --- Dashboard & data loading ---
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
        updateMLDiagram();
        loadBudgetInputs('budgetAdd', 'budgetInputsAdd');
        loadBudgetInputs('budgetStand', 'budgetInputsStandalone');
    } catch(e) { toast('Error loading data'); }
}
function renderTransactions() { document.getElementById('txTableBody').innerHTML = allTransactions.slice(0,100).map(t=>`<tr><td>${new Date(t.tx_date).toLocaleDateString()}</td><td>${t.category}</td><td>${t.is_need ? 'Need' : 'Want'}</td><td>${PRIO_MAP[t.priority]||''}</td><td>${t.note||''}</td><td>${t.tx_type}</td><td>${fmt(t.amount)}</td></tr>`).join(''); }
async function loadAnalytics() {
    if(!currentUser) return;
    try { let longevity = await apiFetch(`/api/longevity/${currentUser.id}`); document.getElementById('longevityContainer').innerHTML = `<div>💰 Balance: ${fmt(longevity.balance)}<br>📉 Avg Daily: ${fmt(longevity.avg_daily_spend)}<br>📅 Days left: ${longevity.days}</div>`; } catch(e) { document.getElementById('longevityContainer').innerHTML = 'Not enough data'; }
    if(currentPrediction.has_data) {
        if(forecastChart) forecastChart.destroy();
        forecastChart = new Chart(document.getElementById('forecastChart'), { type:'line', data:{ labels:Object.keys(currentPrediction.predictions.weekly), datasets:[{ label:'ML Forecast', data:Object.values(currentPrediction.predictions.weekly), borderColor:'#00d282' }] } });
        if(catBarChart) catBarChart.destroy();
        catBarChart = new Chart(document.getElementById('catBarChart'), { type:'bar', data:{ labels:Object.keys(currentPrediction.predictions.categories), datasets:[{ label:'Spent', data:Object.values(currentPrediction.predictions.categories), backgroundColor:'#3b82f6' }] }, options:{ indexAxis:'y' } });
    }
}

// --- Navigation ---
function navigate(screenId) {
    document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
    document.getElementById(`screen-${screenId}`).classList.add('active');
    document.getElementById('pageTitle').innerText = screenId.charAt(0).toUpperCase()+screenId.slice(1);
    if(screenId === 'analytics') loadAnalytics();
    if(screenId === 'add') { updateMLDiagram(); loadBudgetInputs('budgetAdd', 'budgetInputsAdd'); }
    if(screenId === 'budgets') loadBudgetInputs('budgetStand', 'budgetInputsStandalone');
}
document.querySelectorAll('.nav-item').forEach(btn=>btn.addEventListener('click',()=>{ let scr = btn.dataset.nav; if(scr) navigate(scr); }));
document.getElementById('exportBtn').addEventListener('click', ()=> window.location.href='/api/export/csv');
document.getElementById('signoutBtn').addEventListener('click', async()=>{ await fetch('/api/logout',{method:'POST',credentials:'include'}); location.reload(); });
document.getElementById('incomeForm').addEventListener('submit', async(e)=>{ e.preventDefault(); await apiFetch('/api/transactions', { method:'POST', body:JSON.stringify({ amount: parseFloat(document.getElementById('incomeAmount').value), category:'Income', tx_type:'income', note: document.getElementById('incomeNote').value }) }); toast('Income recorded'); document.getElementById('incomeForm').reset(); await loadDashboard(); });
document.getElementById('expenseForm').addEventListener('submit', async(e)=>{ e.preventDefault(); let amount = parseFloat(document.getElementById('expenseAmount').value); let category = document.getElementById('expenseCategory').value; let isNeed = document.getElementById('isNeed').checked; let priority = parseInt(document.getElementById('expensePriority').value); let note = document.getElementById('expenseNote').value; if(document.getElementById('prefAlertWant').checked && !isNeed && category !== 'Income') { if(!confirm('This is a WANT expense. Continue?')) return; } await apiFetch('/api/transactions', { method:'POST', body:JSON.stringify({ amount, category, tx_type:'expense', is_need:isNeed, priority, note }) }); toast('Expense recorded'); document.getElementById('expenseForm').reset(); document.getElementById('isNeed').checked = false; await loadDashboard(); });
document.getElementById('socialStatusDiagram').addEventListener('change', updateMLDiagram);
document.getElementById('mindsetDiagram').addEventListener('change', updateMLDiagram);
document.getElementById('budgetCycle').addEventListener('change', updateMLDiagram);
document.getElementById('monthlyBudgetCap').addEventListener('input', updateMLDiagram);

// --- Auth & init (same as before) ---
const authOverlay = document.getElementById('authOverlay');
const authTitle = document.getElementById('authTitle');
const regName = document.getElementById('regName');
const authConfirm = document.getElementById('authConfirm');
const termsRow = document.getElementById('termsRow');
const authBtn = document.getElementById('authBtn');
const toggleAuthLink = document.getElementById('toggleAuthLink');
toggleAuthLink.addEventListener('click', () => {
    isLogin = !isLogin;
    if(isLogin) {
        authTitle.innerText = 'Welcome back';
        regName.style.display = 'none';
        authConfirm.style.display = 'none';
        termsRow.style.display = 'none';
        authBtn.innerText = 'Sign In';
        toggleAuthLink.innerText = "Don't have an account? Register";
    } else {
        authTitle.innerText = 'Create Account';
        regName.style.display = 'block';
        authConfirm.style.display = 'block';
        termsRow.style.display = 'block';
        authBtn.innerText = 'Register';
        toggleAuthLink.innerText = 'Already have an account? Sign In';
    }
});
authBtn.addEventListener('click', async () => {
    const email = document.getElementById('authEmail').value;
    const password = document.getElementById('authPass').value;
    if(isLogin) {
        try {
            let res = await fetch('/api/login', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email, password}), credentials:'include' });
            if(res.ok) { location.reload(); } else { let err = await res.json(); document.getElementById('authMsg').innerText = err.error || 'Login failed'; }
        } catch(e) { document.getElementById('authMsg').innerText = 'Error'; }
    } else {
        const name = document.getElementById('regName').value;
        const confirm = document.getElementById('authConfirm').value;
        const terms = document.getElementById('termsCheck').checked;
        if(!name || !email || !password || !confirm) { document.getElementById('authMsg').innerText = 'All fields required'; return; }
        if(password !== confirm) { document.getElementById('authMsg').innerText = 'Passwords do not match'; return; }
        if(!terms) { document.getElementById('authMsg').innerText = 'Accept terms'; return; }
        try {
            let res = await fetch('/api/register', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name, email, password}), credentials:'include' });
            if(res.ok) { document.getElementById('authMsg').innerText = 'Registered! Please sign in.'; isLogin = true; toggleAuthLink.click(); }
            else { let err = await res.json(); document.getElementById('authMsg').innerText = err.error || 'Registration failed'; }
        } catch(e) { document.getElementById('authMsg').innerText = 'Error'; }
    }
});
document.getElementById('sendChatBtn').addEventListener('click', async () => {
    let input = document.getElementById('chatInput');
    let msg = input.value.trim();
    if(!msg) return;
    let chatDiv = document.getElementById('chatMessages');
    chatDiv.innerHTML += `<div class="message user-message">${escapeHtml(msg)}</div>`;
    input.value = '';
    try {
        let res = await apiFetch('/api/chatbot', { method:'POST', body:JSON.stringify({ message: msg }) });
        chatDiv.innerHTML += `<div class="message bot-message">${escapeHtml(res.response)}</div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
    } catch(e) { chatDiv.innerHTML += `<div class="message bot-message">Error contacting AI</div>`; }
});
function escapeHtml(str) { return str.replace(/[&<>]/g, function(m){ if(m==='&') return '&amp;'; if(m==='<') return '&lt;'; if(m==='>') return '&gt;'; return m;}); }
document.getElementById('closeChatBtn').addEventListener('click', ()=> document.getElementById('chatContainer').style.display = 'none');
let drag = false, offsetX, offsetY;
const chatContainer = document.getElementById('chatContainer');
const chatHeader = document.getElementById('chatHeader');
chatHeader.addEventListener('mousedown', (e) => { drag = true; offsetX = e.clientX - chatContainer.offsetLeft; offsetY = e.clientY - chatContainer.offsetTop; });
window.addEventListener('mousemove', (e) => { if(drag) { chatContainer.style.left = (e.clientX - offsetX) + 'px'; chatContainer.style.top = (e.clientY - offsetY) + 'px'; chatContainer.style.right = 'auto'; chatContainer.style.bottom = 'auto'; } });
window.addEventListener('mouseup', () => drag = false);
async function init() {
    let res = await fetch('/api/me', { credentials:'include' });
    if(res.ok) { currentUser = await res.json(); document.getElementById('authOverlay').style.display = 'none'; document.getElementById('userAvatar').innerText = currentUser.name.slice(0,2).toUpperCase(); await loadDashboard(); }
    else { document.getElementById('authOverlay').style.display = 'flex'; }
}
init();
</script>
</body>
</html>
"""
