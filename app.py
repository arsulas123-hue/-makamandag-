import os
import datetime
import hashlib
import secrets
import json
import csv
import io
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)

# === DATABASE CONFIGURATION ===
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
    is_need = db.Column(db.Boolean, default=False)           # manually set need/want
    priority = db.Column(db.Integer, default=0)             # 0 = Low, 1 = Medium, 2 = High, 3 = Critical
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
    wants_needs_json = db.Column(db.Text, default='{}')   # optional custom mapping

# ========== CREATE TABLES & MIGRATIONS ==========
with app.app_context():
    db.create_all()
    # Add is_need column if missing
    try:
        db.session.execute('ALTER TABLE transactions ADD COLUMN is_need BOOLEAN DEFAULT 0')
        db.session.commit()
    except Exception:
        pass
    # Add priority column if missing
    try:
        db.session.execute('ALTER TABLE transactions ADD COLUMN priority INTEGER DEFAULT 0')
        db.session.commit()
    except Exception:
        pass

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

# ---------- ML Functions (enhanced with priority) ----------
def _priority_penalty(priority):
    """Lower priority => higher penalty multiplier for overspending."""
    # priority 0 (low) -> 2.0, 1 (medium) -> 1.5, 2 (high) -> 0.8, 3 (critical) -> 0.3
    penalties = {0: 2.0, 1: 1.5, 2: 0.8, 3: 0.3}
    return penalties.get(priority, 1.5)

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
            # Find the lowest priority (worst case) spent in that category to apply penalty
            cat_priorities = [t.get('priority', 0) for t in transactions if t['category'] == cat and t['tx_type'] == 'expense']
            worst_priority = min(cat_priorities) if cat_priorities else 0
            multiplier = _priority_penalty(worst_priority)
            budget_penalty += over_ratio * 10 * multiplier
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
                # Show priority of the biggest overspent category
                cat_prios = [t.get('priority', 0) for t in transactions if t['category'] == cat and t['tx_type'] == 'expense']
                worst_prio = min(cat_prios) if cat_prios else 0
                prio_label = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}.get(worst_prio, 'Low')
                advice.append({
                    "cat": cat,
                    "msg": f"⚠️ Overspent by ₱{spent-limit:.2f} (Priority: {prio_label}). Reduce or adjust budget."
                })
            elif spent < limit * 0.7:
                advice.append({"cat": cat, "msg": f"✅ Great! Underspent by ₱{limit-spent:.2f}. Consider saving."})
            else:
                advice.append({"cat": cat, "msg": f"✔️ On track. Spent ₱{spent:.2f} of ₱{limit:.2f} budget."})
    if not advice:
        advice.append({"cat": "General", "msg": "No budget limits set. Set budgets to get personalized advice."})

    # Additional insight based on mindset
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
        budget = Budget.query.filter_by(user_id=user_id, category=category).first()
        if budget:
            budget.limit_amount = limit
        else:
            budget = Budget(user_id=user_id, category=category, limit_amount=limit)
            db.session.add(budget)
        db.session.commit()
        return jsonify({'message': 'Budget saved'})

@app.route('/api/budgets/bulk/<int:user_id>', methods=['POST'])
@login_required
def bulk_budgets(user_id):
    if session['user_id'] != user_id:
        return jsonify({'error': 'Forbidden'}), 403
    data = request.json
    limits = data.get('limits', {})
    for category, limit_amount in limits.items():
        budget = Budget.query.filter_by(user_id=user_id, category=category).first()
        if budget:
            budget.limit_amount = limit_amount
        else:
            budget = Budget(user_id=user_id, category=category, limit_amount=limit_amount)
            db.session.add(budget)
    db.session.commit()
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
        if data['tx_type'] == 'expense' and 'priority' not in data:
            data['priority'] = 0  # default low priority if not supplied
        txn = Transaction(
            user_id=user_id,
            amount=data['amount'],
            category=data['category'],
            tx_type=data['tx_type'],
            is_need=data.get('is_need', False),
            priority=data.get('priority', 0),
            note=data.get('note', ''),
            tx_date=datetime.datetime.utcnow()
        )
        db.session.add(txn)
        db.session.commit()
        return jsonify({'message': 'Transaction added', 'id': txn.id})

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
    hours_left = days_left * 24
    weeks_left = days_left / 7
    months_left = days_left / 30.44
    years_left = days_left / 365.25
    return jsonify({
        'balance': balance,
        'avg_daily_spend': round(avg_daily, 2),
        'hours': round(max(0, hours_left), 1),
        'days': round(max(0, days_left), 1),
        'weeks': round(max(0, weeks_left), 1),
        'months': round(max(0, months_left), 1),
        'years': round(max(0, years_left), 2)
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

    def _prio_label(p):
        return {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}.get(p, 'Low')

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
        # Show expenses by priority
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

# ========== FRONTEND (Single HTML file - updated with priority) ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartSpend · AI-Powered Finance</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root { --bg: #0a0f1a; --bg2: #0f1622; --bg3: #151e2d; --panel: rgba(21, 30, 45, 0.9); --border: rgba(0, 210, 130, 0.2); --border2: rgba(255, 255, 255, 0.05); --green: #00d282; --green2: #00ff9d; --green-dim: rgba(0, 210, 130, 0.15); --red: #ff4d6d; --amber: #f0a500; --blue: #3b82f6; --purple: #a855f7; --muted: #6c86a0; --text: #e2eff8; --mono: 'JetBrains Mono', monospace; --sans: 'Space Grotesk', sans-serif; --r: 16px; }
        body { font-family: var(--sans); background: var(--bg); color: var(--text); min-height: 100vh; }
        .sidebar { position: fixed; left: 0; top: 0; bottom: 0; width: 80px; background: rgba(15, 22, 34, 0.98); backdrop-filter: blur(12px); border-right: 1px solid var(--border2); display: flex; flex-direction: column; align-items: center; padding: 24px 0; gap: 10px; z-index: 100; transition: width 0.3s; }
        .sidebar:hover { width: 240px; }
        .sidebar-logo { width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, var(--green), #009e5f); display: flex; align-items: center; justify-content: center; margin-bottom: 28px; cursor: pointer; }
        .nav-item { width: 100%; display: flex; align-items: center; gap: 16px; padding: 12px 24px; border-radius: 12px; cursor: pointer; border: none; background: transparent; color: var(--muted); font-size: 0.9rem; font-weight: 500; white-space: nowrap; overflow: hidden; transition: all 0.2s; }
        .nav-item:hover { background: var(--green-dim); color: var(--text); transform: translateX(6px); }
        .nav-item.active { background: var(--green-dim); color: var(--green); border-left: 3px solid var(--green); }
        .nav-label { opacity: 0; transition: opacity 0.2s; }
        .sidebar:hover .nav-label { opacity: 1; }
        .sidebar-bottom { margin-top: auto; width: 100%; }
        .main { margin-left: 80px; padding: 24px 32px; position: relative; }
        @media (max-width: 768px) { .sidebar { width: 70px; } .main { margin-left: 70px; padding: 20px; } }
        @media (max-width: 600px) { .sidebar { display: none; } .main { margin-left: 0; } }
        .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px; flex-wrap: wrap; gap: 16px; }
        .topbar h1 { font-size: 1.75rem; font-weight: 700; background: linear-gradient(135deg, #fff, var(--green)); background-clip: text; -webkit-background-clip: text; color: transparent; }
        .score-pill { display: flex; align-items: center; gap: 10px; padding: 8px 20px; border-radius: 99px; background: var(--green-dim); border: 1px solid var(--border); font-size: 0.85rem; font-weight: 600; }
        .score-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100%{ opacity:1; } 50%{ opacity:0.4; } }
        .avatar { width: 42px; height: 42px; border-radius: 50%; background: linear-gradient(135deg, var(--blue), var(--purple)); display: flex; align-items: center; justify-content: center; font-weight: 700; cursor: pointer; }
        .signout-btn { background: rgba(255,77,109,0.2); border: 1px solid rgba(255,77,109,0.2); color: var(--red); padding: 8px 18px; border-radius: 10px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
        .screen { display: none; animation: fadeUp 0.35s ease; }
        .screen.active { display: block; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .stat-card { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; transition: all 0.25s; }
        .stat-value { font-size: 1.8rem; font-weight: 700; font-family: var(--mono); }
        .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 8px; }
        .panel { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; margin-bottom: 20px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
        .btn-green { background: linear-gradient(135deg, var(--green), #009e5f); color: #000; border: none; padding: 10px 18px; border-radius: 10px; cursor: pointer; font-weight: 600; }
        input, select, button { font-family: inherit; }
        input, select { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); border-radius: 10px; padding: 10px 14px; outline: none; }
        .flex-form { display: flex; flex-direction: column; gap: 16px; }
        .forecast-row { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
        .forecast-week { width: 70px; font-weight: 600; color: var(--green); }
        .forecast-bar-track { flex: 1; height: 8px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
        .forecast-bar-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--green2)); width: 0; border-radius: 99px; transition: width 1s ease; }
        .advice-item { padding: 14px; border-radius: 12px; margin-bottom: 10px; background: var(--bg3); transition: all 0.2s; }
        .waste-alert { background: rgba(255,77,109,0.2); border-left: 3px solid var(--red); }
        #toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 1px solid var(--green); border-radius: 12px; padding: 12px 24px; opacity: 0; transition: 0.25s; z-index: 10000; }
        #toast.show { opacity: 1; }
        .auth-overlay { position: fixed; inset: 0; background: rgba(8,13,20,0.98); backdrop-filter: blur(20px); z-index: 9999; display: flex; align-items: center; justify-content: center; }
        .auth-card { background: linear-gradient(145deg, #0d1520, #0a1220); border: 1px solid rgba(0,210,130,0.2); border-radius: 28px; padding: 40px; width: 400px; max-width: 90%; }
        .chat-container { position: fixed; bottom: 24px; right: 24px; z-index: 10001; }
        .chat-window { width: 380px; height: 480px; background: var(--bg2); backdrop-filter: blur(10px); border: 1px solid var(--green); border-radius: 24px; display: flex; flex-direction: column; overflow: hidden; }
        .chat-header { padding: 14px 18px; background: var(--bg3); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; cursor: move; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 85%; padding: 10px 14px; border-radius: 18px; font-size: 0.85rem; }
        .user-message { align-self: flex-end; background: var(--green-dim); color: var(--green); border-bottom-right-radius: 4px; }
        .bot-message { align-self: flex-start; background: var(--bg3); color: var(--text); border-bottom-left-radius: 4px; }
        .chat-input { display: flex; padding: 14px; gap: 10px; background: var(--bg3); border-top: 1px solid var(--border); }
        .ml-badge { background: linear-gradient(135deg, var(--purple), var(--blue)); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 8px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border2); }
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
    <div class="topbar"><h1 id="pageTitle">Dashboard</h1><div class="topbar-right" style="display:flex; gap:16px; align-items:center;"><div class="score-pill"><div class="score-dot"></div><span>Health: <strong id="topScore">—</strong><span class="ml-badge">ML</span></span></div><div class="avatar" id="userAvatar">US</div><button class="signout-btn" id="signoutBtn">Sign Out</button></div></div>
    <div id="toast" class="toast"></div>

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

    <!-- ADD TRANSACTION (with priority) -->
    <div class="screen" id="screen-add">
        <div class="stats-grid">
            <div class="panel"><div class="panel-header">➕ Add Income</div><form id="incomeForm" class="flex-form"><input type="number" id="incomeAmount" placeholder="Amount (₱)" step="0.01" required><input type="text" id="incomeNote" placeholder="Note"><button type="submit" class="btn-green">Record Income</button></form></div>
            <div class="panel"><div class="panel-header">📝 Record Expense (with Needs/Wants & Priority)</div>
                <form id="expenseForm" class="flex-form">
                    <input type="number" id="expenseAmount" placeholder="Amount (₱)" step="0.01" required>
                    <select id="expenseCategory">
                        <option>Food & Dining</option><option>Transport</option><option>Groceries</option>
                        <option>Entertainment</option><option>Health</option><option>Other</option>
                    </select>
                    <div style="display:flex; gap:10px; align-items:center;">
                        <label><input type="checkbox" id="isNeed"> ✅ NEED</label>
                        <label style="margin-left:12px;">Priority:
                            <select id="expensePriority">
                                <option value="0">Low</option>
                                <option value="1" selected>Medium</option>
                                <option value="2">High</option>
                                <option value="3">Critical</option>
                            </select>
                        </label>
                    </div>
                    <input type="text" id="expenseNote" placeholder="Note">
                    <button type="submit" class="btn-green">Record Expense</button>
                </form>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">🧠 AI Budget Engine</div>
            <div style="display:flex; gap:16px;">
                <select id="socialStatus"><option>Low</option><option selected>Middle</option><option>Upper</option></select>
                <select id="spendingMindset"><option>Saver</option><option selected>Neutral</option><option>Spender</option></select>
                <button id="saveProfileBtn" class="btn-green">Save & Generate Scenarios</button>
            </div>
            <div id="scenariosContainer" style="margin-top:20px;"></div>
        </div>
    </div>

    <!-- ANALYTICS -->
    <div class="screen" id="screen-analytics">
        <div class="panel"><div class="panel-header">⏳ Budget Longevity</div><div id="longevityContainer">Loading...</div></div>
        <div class="panel"><div class="panel-header">Weekly Forecast</div><canvas id="forecastChart" height="200"></canvas></div>
        <div class="panel"><div class="panel-header">Category Breakdown</div><canvas id="catBarChart" height="200"></canvas></div>
    </div>

    <!-- BUDGETS -->
    <div class="screen" id="screen-budgets">
        <div class="panel"><div class="panel-header">Monthly Budget Limits <button id="saveBudgetsBtn" class="btn-green">Save All</button></div>
            <div id="budgetInputs" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px;"></div>
        </div>
    </div>

    <!-- HISTORY (shows priority) -->
    <div class="screen" id="screen-transactions">
        <div class="table-wrap"><table><thead><tr><th>Date</th><th>Category</th><th>Need?</th><th>Priority</th><th>Note</th><th>Type</th><th>Amount</th></tr></thead><tbody id="txTableBody"></tbody></table></div>
    </div>

    <!-- AUTH -->
    <div id="authOverlay" class="auth-overlay"><div class="auth-card"><h2 id="authTitle">Welcome back</h2><input type="text" id="regName" class="auth-input" placeholder="Full Name" style="display:none"><input type="email" id="authEmail" class="auth-input" placeholder="Email"><input type="password" id="authPass" class="auth-input" placeholder="Password"><input type="password" id="authConfirm" class="auth-input" placeholder="Confirm Password" style="display:none"><div id="termsRow" style="display:none;"><label><input type="checkbox" id="termsCheck"> Accept Terms</label></div><div id="authMsg" style="color:#ff4d6d;"></div><button id="authBtn" class="auth-btn">Sign In</button><div id="toggleAuthLink" class="auth-link" style="cursor:pointer; margin-top:12px;">Don't have an account? Register</div></div></div>

    <!-- CHATBOT -->
    <div id="chatContainer" class="chat-container"><div class="chat-window"><div class="chat-header" id="chatHeader">🤖 SmartSpend AI <span class="ml-badge">ML</span><button id="closeChatBtn">✕</button></div><div class="chat-messages" id="chatMessages"><div class="message bot-message">✨ Ask: 'health score', 'forecast', 'investment', 'needs vs wants', 'priority / ranking'.</div></div><div class="chat-input"><input id="chatInput" placeholder="Ask..."><button id="sendChatBtn">Send</button></div></div></div>
</main>

<script>
let currentUser = null, allTransactions = [], currentSummary = { balance:0, expense:0, income:0 }, currentPrediction = { has_data:false, score:null, predictions:{ weekly:{}, categories:{} }, advice:[] };
let monthlyChart, forecastChart, catBarChart, isLogin = true;

function fmt(amt) { return '₱' + Number(amt).toLocaleString('en-PH', { minimumFractionDigits:2 }); }
function toast(msg) { let t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2500); }
async function apiFetch(url, opts={}) { let res = await fetch(url, {...opts, credentials:'include', headers:{'Content-Type':'application/json'}}); if(!res.ok) throw new Error((await res.json()).error); return res.json(); }

const PRIO_MAP = {'0':'Low','1':'Medium','2':'High','3':'Critical'};

async function loadDashboard() {
    if(!currentUser) return;
    try {
        let summary = await apiFetch(`/api/summary/${currentUser.id}`);
        let pred = await apiFetch(`/api/predict/${currentUser.id}`);
        allTransactions = await apiFetch(`/api/transactions?user_id=${currentUser.id}`);
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
    } catch(e) { toast('Error loading data'); }
}

function renderTransactions() {
    document.getElementById('txTableBody').innerHTML = allTransactions.slice(0,50).map(t=>`<tr><td>${new Date(t.tx_date).toLocaleDateString()}</td><td>${t.category}</td><td>${t.is_need ? '✅Need' : '⚪Want'}</td><td>${PRIO_MAP[t.priority]||''}</td><td>${t.note||''}</td><td>${t.tx_type}</td><td>${fmt(t.amount)}</td></tr>`).join('');
}

async function loadAnalytics() {
    if(!currentUser) return;
    try {
        let longevity = await apiFetch(`/api/longevity/${currentUser.id}`);
        document.getElementById('longevityContainer').innerHTML = `<div style="background:var(--bg3);padding:16px;border-radius:12px;">💰 Balance: ${fmt(longevity.balance)}<br>📉 Avg Daily: ${fmt(longevity.avg_daily_spend)}<br>📅 Days left: ${longevity.days} | Weeks: ${longevity.weeks}</div>`;
    } catch(e) { document.getElementById('longevityContainer').innerHTML = 'Add expenses first.'; }
    if(currentPrediction.has_data) {
        if(forecastChart) forecastChart.destroy();
        forecastChart = new Chart(document.getElementById('forecastChart'), { type:'line', data:{ labels:Object.keys(currentPrediction.predictions.weekly), datasets:[{ label:'ML Forecast', data:Object.values(currentPrediction.predictions.weekly), borderColor:'#00d282' }] } });
        if(catBarChart) catBarChart.destroy();
        catBarChart = new Chart(document.getElementById('catBarChart'), { type:'bar', data:{ labels:Object.keys(currentPrediction.predictions.categories), datasets:[{ label:'Spent', data:Object.values(currentPrediction.predictions.categories), backgroundColor:'#3b82f6' }] }, options:{ indexAxis:'y' } });
    }
}

async function saveProfileAndGenerate() {
    let social = document.getElementById('socialStatus').value;
    let mindset = document.getElementById('spendingMindset').value;
    await apiFetch('/api/user/profile', { method:'POST', body:JSON.stringify({ social_status:social, spending_mindset:mindset }) });
    toast('Profile saved. Generating scenarios...');
    let res = await apiFetch(`/api/budget/scenarios/${currentUser.id}`);
    let container = document.getElementById('scenariosContainer');
    if(!res.scenarios.length) { container.innerHTML = '<div>No data yet.</div>'; return; }
    container.innerHTML = '<h4>ML Scenarios</h4>';
    for(let s of res.scenarios) {
        let div = document.createElement('div');
        div.className = 'advice-item';
        div.innerHTML = `<strong>${s.name}</strong><br>Savings Goal: ${fmt(s.savings_goal)} | Investment: ${fmt(s.investment_goal)}<br><button class="btn-green" onclick='applyBudgets(${JSON.stringify(s.budget_limits)})'>Apply Budgets</button>`;
        container.appendChild(div);
    }
}
window.applyBudgets = async (limits) => { await apiFetch(`/api/budgets/bulk/${currentUser.id}`, { method:'POST', body:JSON.stringify({ limits }) }); toast('Budgets applied!'); await loadDashboard(); };

async function loadBudgets() {
    let budgets = await apiFetch(`/api/budgets/${currentUser.id}`);
    let limits = Object.fromEntries(budgets.map(b=>[b.category, b.limit]));
    let categories = ['Food & Dining','Transport','Groceries','Entertainment','Health'];
    let html = categories.map(cat=>`<div><label>${cat}</label><input type="number" id="budget_${cat.replace(/\\s/g,'')}" value="${limits[cat]||''}" placeholder="Limit"></div>`).join('');
    document.getElementById('budgetInputs').innerHTML = html;
    document.getElementById('saveBudgetsBtn').onclick = async ()=>{ for(let cat of categories) { let val = parseFloat(document.getElementById(`budget_${cat.replace(/\\s/g,'')}`).value); if(val && val>0) await apiFetch(`/api/budgets/${currentUser.id}`, { method:'POST', body:JSON.stringify({ category:cat, limit:val }) }); } toast('Budgets saved'); await loadDashboard(); };
}

document.getElementById('incomeForm').addEventListener('submit', async(e)=>{ e.preventDefault(); await apiFetch('/api/transactions', { method:'POST', body:JSON.stringify({ amount: parseFloat(document.getElementById('incomeAmount').value), category:'Income', tx_type:'income', note: document.getElementById('incomeNote').value }) }); toast('Income recorded'); document.getElementById('incomeForm').reset(); await loadDashboard(); });
document.getElementById('expenseForm').addEventListener('submit', async(e)=>{
    e.preventDefault();
    let amount = parseFloat(document.getElementById('expenseAmount').value);
    let category = document.getElementById('expenseCategory').value;
    let isNeed = document.getElementById('isNeed').checked;
    let priority = parseInt(document.getElementById('expensePriority').value);
    let note = document.getElementById('expenseNote').value;
    await apiFetch('/api/transactions', { method:'POST', body:JSON.stringify({ amount, category, tx_type:'expense', is_need:isNeed, priority, note }) });
    toast('Expense recorded');
    document.getElementById('expenseForm').reset(); document.getElementById('isNeed').checked = false;
    document.getElementById('expensePriority').value = '1';
    await loadDashboard();
});

document.querySelectorAll('.nav-item').forEach(btn=>btn.addEventListener('click',()=>{ let scr = btn.dataset.nav; if(scr) navigate(scr); }));
function navigate(screenId) { document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active')); document.getElementById(`screen-${screenId}`).classList.add('active'); document.getElementById('pageTitle').innerText = screenId.charAt(0).toUpperCase()+screenId.slice(1); if(screenId === 'analytics') loadAnalytics(); if(screenId === 'budgets') loadBudgets(); }

document.getElementById('saveProfileBtn').addEventListener('click', saveProfileAndGenerate);
document.getElementById('exportBtn').addEventListener('click', ()=>{ window.location.href = '/api/export/csv'; });
document.getElementById('signoutBtn').addEventListener('click', async()=>{ await fetch('/api/logout',{method:'POST',credentials:'include'}); location.reload(); });
document.getElementById('toggleAuthLink').addEventListener('click',()=>{ isLogin=!isLogin; document.getElementById('authTitle').innerText = isLogin ? 'Welcome back' : 'Create Account'; document.getElementById('regName').style.display = isLogin ? 'none' : 'block'; document.getElementById('authConfirm').style.display = isLogin ? 'none' : 'block'; document.getElementById('termsRow').style.display = isLogin ? 'none' : 'block'; document.getElementById('authBtn').innerText = isLogin ? 'Sign In' : 'Create Account'; document.getElementById('toggleAuthLink').innerHTML = isLogin ? "Don't have an account? Register" : "Already have an account? Sign In"; });
document.getElementById('authBtn').addEventListener('click', async()=>{ let email = document.getElementById('authEmail').value; let pass = document.getElementById('authPass').value; if(!isLogin) { let name = document.getElementById('regName').value; let confirm = document.getElementById('authConfirm').value; let terms = document.getElementById('termsCheck').checked; if(pass !== confirm) { document.getElementById('authMsg').innerText='Passwords mismatch'; return; } if(!terms) { document.getElementById('authMsg').innerText='Accept terms'; return; } await fetch('/api/register', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ name, email, password:pass, accepted_terms:true }), credentials:'include' }); document.getElementById('authMsg').innerText='Account created! Please login.'; setTimeout(()=>document.getElementById('toggleAuthLink').click(),1500); return; } let res = await fetch('/api/login', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ email, password:pass }), credentials:'include' }); if(res.ok) { let data = await res.json(); currentUser = data.user; document.getElementById('authOverlay').style.display = 'none'; document.getElementById('userAvatar').innerText = currentUser.name.slice(0,2).toUpperCase(); await loadDashboard(); navigate('dashboard'); } else { document.getElementById('authMsg').innerText='Invalid credentials'; } });
document.getElementById('sendChatBtn').addEventListener('click', async()=>{ let input = document.getElementById('chatInput'); let msg = input.value.trim(); if(!msg) return; input.value = ''; let container = document.getElementById('chatMessages'); container.innerHTML += `<div class="message user-message">${msg}</div>`; container.scrollTop = container.scrollHeight; let res = await fetch('/api/chatbot', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ message:msg }), credentials:'include' }); let data = await res.json(); container.innerHTML += `<div class="message bot-message">${data.response}</div>`; container.scrollTop = container.scrollHeight; });
document.getElementById('closeChatBtn').addEventListener('click', ()=> document.getElementById('chatContainer').style.display = 'none');
async function init() { let res = await fetch('/api/me', { credentials:'include' }); if(res.ok) { currentUser = await res.json(); document.getElementById('authOverlay').style.display = 'none'; document.getElementById('userAvatar').innerText = currentUser.name.slice(0,2).toUpperCase(); await loadDashboard(); } else { document.getElementById('authOverlay').style.display = 'flex'; } }
init();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
