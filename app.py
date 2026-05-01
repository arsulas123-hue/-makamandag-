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
    is_need = db.Column(db.Boolean, default=False)
    priority = db.Column(db.Integer, default=0)
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

# ========== CREATE TABLES ==========
with app.app_context():
    db.create_all()
    try:
        db.session.execute('ALTER TABLE transactions ADD COLUMN is_need BOOLEAN DEFAULT 0')
        db.session.commit()
    except Exception:
        pass
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

def _priority_penalty(priority):
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
            data['priority'] = 0
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
