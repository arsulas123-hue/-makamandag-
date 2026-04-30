import os
import datetime
import hashlib
import secrets
import json
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)

# === DATABASE CONFIGURATION (FIXED) ===
# Use SQLite as a reliable default, falling back to provided DATABASE_URL if available
# This ensures the app works without external PostgreSQL issues
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False

# Database: Prefer SQLite for local development, allows DATABASE_URL override
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
    tx_type = db.Column(db.String(10), nullable=False)
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

# ---------- ML Functions ----------
def calculate_health_score(transactions, budgets):
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    expense = sum(t['amount'] for t in transactions if t['tx_type'] == 'expense')
    if income == 0:
        return 0
    savings_rate = (income - expense) / income
    score = max(0, min(100, savings_rate * 100))

    cat_spending = {}
    for t in transactions:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    budget_penalty = 0
    total_budget = 0
    for cat, limit in budgets.items():
        spent = cat_spending.get(cat, 0)
        if limit > 0:
            total_budget += limit
            if spent > limit:
                over = (spent - limit) / limit
                budget_penalty += over * 10
    if total_budget > 0:
        score = max(0, min(100, score - budget_penalty))
    return int(score)

def forecast_spending(transactions):
    """Forecast next 4 weeks from expense patterns."""
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

def generate_advice(transactions, budgets):
    advice = []
    cat_spending = {}
    for t in transactions:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    for cat, limit in budgets.items():
        spent = cat_spending.get(cat, 0)
        if limit > 0:
            if spent > limit:
                advice.append({"cat": cat, "msg": f"⚠️ Overspent by ₱{spent-limit:.2f}. Reduce or adjust budget."})
            elif spent < limit * 0.7:
                advice.append({"cat": cat, "msg": f"✅ Great! You underspent by ₱{limit-spent:.2f}. Consider saving the difference."})
            else:
                advice.append({"cat": cat, "msg": f"✔️ On track. Spent ₱{spent:.2f} of ₱{limit:.2f} budget."})
    if not advice:
        advice.append({"cat": "General", "msg": "No budget limits set. Set budgets to get personalized advice."})
    return advice

def generate_scenarios(user_id, transactions, budgets):
    """Generate three financial scenarios based on user profile and spending."""
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    total_expense = sum(e['amount'] for e in expenses)
    savings = max(0, income - total_expense)
    profile = get_user_profile(user_id)
    social = profile['social_status']
    mindset = profile['spending_mindset']
    cat_spend = {}
    for e in expenses:
        cat_spend[e['category']] = cat_spend.get(e['category'], 0) + e['amount']
    default_cats = ['Food & Dining', 'Transport', 'Groceries', 'Entertainment', 'Health']

    def build_scenario(name, savings_mult, invest_mult, essential_mult, disc_mult):
        limits = {}
        for cat in default_cats:
            spent = cat_spend.get(cat, 0)
            if cat in ['Food & Dining', 'Groceries', 'Health']:
                suggested = max(spent * essential_mult, 2000) if spent > 0 else 3000
            else:
                suggested = max(spent * disc_mult, 1000) if spent > 0 else 2000
            limits[cat] = round(suggested)
        if social == 'Low':
            limits['Entertainment'] = limits.get('Entertainment', 1500) * 0.6
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
    cons = build_scenario('Conservative (Safe)', 0.25, 0.6, 1.0, 0.7)
    bal = build_scenario('Balanced (Moderate)', 0.20, 0.7, 1.0, 1.0)
    agg = build_scenario('Aggressive (Growth)', 0.15, 0.9, 0.9, 1.3)
    return [cons, bal, agg]

# ========== API ROUTES ==========
@app.route('/')
def index():
    """Serve the main dashboard HTML (chatbot removed)."""
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
            'note': t.note, 'tx_date': t.tx_date.isoformat()
        } for t in txns])
    else:
        data = request.json
        if not data or 'amount' not in data or 'category' not in data or 'tx_type' not in data:
            return jsonify({'error': 'Invalid transaction data'}), 400
        txn = Transaction(
            user_id=user_id,
            amount=data['amount'],
            category=data['category'],
            tx_type=data['tx_type'],
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
    tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'tx_date': t.tx_date} for t in txns]
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.limit_amount for b in budgets}
    health_score = calculate_health_score(tx_list, budget_dict)
    weekly_forecast = forecast_spending(tx_list)
    cat_spending = {}
    for t in tx_list:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    advice = generate_advice(tx_list, budget_dict)
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
    tx_list = [{'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'tx_date': t.tx_date} for t in txns]
    budgets = Budget.query.filter_by(user_id=user_id).all()
    budget_dict = {b.category: b.limit_amount for b in budgets}
    scenarios = generate_scenarios(user_id, tx_list, budget_dict)
    return jsonify({'scenarios': scenarios})

# ========== MAIN ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# ========== HTML FRONTEND (Chatbot Removed) ==========
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
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
        .sidebar-logo { width: 48px; height: 48px; border-radius: 14px; background: linear-gradient(135deg, var(--green), #009e5f); display: flex; align-items: center; justify-content: center; margin-bottom: 28px; cursor: pointer; }
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
        .signout-btn:hover { background: rgba(255, 77, 109, 0.2); transform: translateY(-2px); }
        .screen { display: none; animation: fadeUp 0.35s ease; }
        .screen.active { display: block; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 28px; }
        .stat-card { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; transition: all 0.25s; cursor: pointer; }
        .stat-card:hover { transform: translateY(-4px); border-color: var(--border); box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
        .stat-value { font-size: 1.8rem; font-weight: 700; font-family: var(--mono); }
        .stat-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 8px; }
        .stat-card.green .stat-value { color: var(--green); }
        .stat-card.red .stat-value { color: var(--red); }
        .stat-card.blue .stat-value { color: var(--blue); }
        .stat-card.amber .stat-value { color: var(--amber); }
        .panel { background: var(--panel); backdrop-filter: blur(10px); border: 1px solid var(--border2); border-radius: var(--r); padding: 22px; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
        .panel-title { font-size: 1rem; font-weight: 600; }
        .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; margin-bottom: 28px; }
        canvas { max-height: 260px; width: 100% !important; }
        input, select, .btn { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); border-radius: 10px; padding: 10px 14px; font-size: 0.85rem; outline: none; transition: all 0.2s; }
        input:focus, select:focus { border-color: var(--green); box-shadow: 0 0 0 2px var(--green-dim); }
        .btn { cursor: pointer; font-weight: 600; }
        .btn-green { background: linear-gradient(135deg, var(--green), #009e5f); color: #000; border: none; }
        .btn-green:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 210, 130, 0.3); }
        .table-wrap { overflow-x: auto; border-radius: var(--r); background: var(--panel); border: 1px solid var(--border2); }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 14px 16px; background: var(--bg3); color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }
        td { padding: 12px 16px; border-bottom: 1px solid var(--border2); font-size: 0.85rem; }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }
        .advice-item { padding: 14px; border-radius: 12px; margin-bottom: 10px; background: var(--bg3); transition: all 0.2s; }
        .advice-item:hover { transform: translateX(6px); background: var(--bg2); }
        .advice-item strong { color: var(--green); }
        .waste-alert { background: rgba(255, 77, 109, 0.2); border-left: 3px solid var(--red); }
        .forecast-row { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; padding: 8px 0; }
        .forecast-week { width: 70px; font-weight: 600; color: var(--green); }
        .forecast-bar-track { flex: 1; height: 8px; background: var(--bg3); border-radius: 99px; overflow: hidden; }
        .forecast-bar-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--green2)); width: 0; border-radius: 99px; transition: width 1s ease; }
        #toast { position: fixed; bottom: 24px; right: 24px; background: var(--bg2); border: 1px solid var(--green); border-radius: 12px; padding: 12px 24px; opacity: 0; transform: translateY(10px); transition: all 0.25s; z-index: 10000; backdrop-filter: blur(12px); }
        #toast.show { opacity: 1; transform: translateY(0); }
        .auth-overlay { position: fixed; inset: 0; background: rgba(8,13,20,0.98); backdrop-filter: blur(20px); z-index: 9999; display: flex; align-items: center; justify-content: center; }
        .auth-card { background: linear-gradient(145deg, #0d1520, #0a1220); border: 1px solid rgba(0,210,130,0.2); border-radius: 28px; padding: 40px; width: 400px; max-width: 90%; }
        .auth-card h2 { text-align: center; margin-bottom: 24px; background: linear-gradient(135deg, #fff, #00d282); background-clip: text; -webkit-background-clip: text; color: transparent; }
        .auth-input { width: 100%; padding: 12px 16px; margin-bottom: 14px; background: #0a1525; border: 1px solid #2a3a4a; border-radius: 12px; color: white; }
        .auth-btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #00d282, #009e5f); border: none; border-radius: 12px; color: #000; font-weight: 700; cursor: pointer; }
        .auth-link { cursor: pointer; text-align: center; margin-top: 16px; color: #6c86a0; }
        .auth-link:hover { color: #00d282; }
        .empty-state { text-align: center; padding: 40px 20px; color: var(--muted); }
        .flex-form { display: flex; flex-direction: column; gap: 16px; }
    </style>
</head>
<body>
<nav class="sidebar">
    <div class="sidebar-logo"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg></div>
    <button class="nav-item active" data-nav="dashboard"><span class="nav-label">Dashboard</span></button>
    <button class="nav-item" data-nav="add"><span class="nav-label">Add Transaction</span></button>
    <button class="nav-item" data-nav="analytics"><span class="nav-label">Analytics</span></button>
    <button class="nav-item" data-nav="budgets"><span class="nav-label">Budgets</span></button>
    <button class="nav-item" data-nav="transactions"><span class="nav-label">History</span></button>
    <div class="sidebar-bottom"><button class="nav-item" id="exportBtn"><span class="nav-label">Export PDF</span></button></div>
</nav>
<main class="main">
    <div class="topbar">
        <div><h1 id="pageTitle">Dashboard</h1></div>
        <div class="topbar-right">
            <div class="score-pill" id="scorePill"><div class="score-dot"></div><span>Health: <strong id="topScore">—</strong><span class="ml-badge" style="background: linear-gradient(135deg, #a855f7, #3b82f6); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; margin-left: 8px;">ML</span></span></div>
            <div class="avatar" id="userAvatar">US</div>
            <button class="signout-btn" id="signoutBtn">Sign Out</button>
        </div>
    </div>
    <div id="toast"></div>

    <!-- DASHBOARD SCREEN -->
    <div class="screen active" id="screen-dashboard">
        <div class="stats-grid">
            <div class="stat-card green"><div class="stat-value" id="sBalance">—</div><div class="stat-label">Balance</div></div>
            <div class="stat-card red"><div class="stat-value" id="sExpense">—</div><div class="stat-label">Monthly Expenses</div></div>
            <div class="stat-card blue"><div class="stat-value" id="sIncome">—</div><div class="stat-label">Monthly Income</div></div>
            <div class="stat-card" id="healthCard"><div class="stat-value" id="sScore">—</div><div class="stat-label">ML Health Score</div></div>
        </div>
        <div class="chart-grid">
            <div class="panel"><div class="panel-header"><div class="panel-title">Income vs Expenses Trend</div></div><canvas id="monthlyChart" height="200"></canvas></div>
            <div class="panel"><div class="panel-header"><div class="panel-title">Spending by Category</div></div><canvas id="donutChart" height="200"></canvas></div>
        </div>
        <div class="chart-grid">
            <div class="panel"><div class="panel-header"><div class="panel-title">ML Spending Forecast (4 weeks) <span class="ml-badge" style="background: linear-gradient(135deg, #a855f7, #3b82f6);">LIVE ML MODEL</span></div><span style="background:var(--green-dim);color:var(--green);padding:4px 12px;border-radius:99px;">AI Powered</span></div><div id="forecastBars"></div></div>
            <div class="panel"><div class="panel-header"><div class="panel-title">Budget Intelligence - AI Advice</div></div><div id="adviceList"></div></div>
        </div>
        <div class="panel" style="margin-top: 20px;"><div class="panel-header"><div class="panel-title">Recent Transactions</div></div><div id="recentList"></div></div>
    </div>

    <!-- ADD TRANSACTION SCREEN -->
    <div class="screen" id="screen-add">
        <div class="stats-grid">
            <div class="panel">
                <div class="panel-header"><div class="panel-title">➕ New Transaction</div></div>
                <form id="txForm" class="flex-form">
                    <input type="number" id="txAmount" placeholder="Amount (₱)" step="0.01" required>
                    <select id="txCategory" required>
                        <option value="Food & Dining">Food & Dining</option>
                        <option value="Transport">Transport</option>
                        <option value="Groceries">Groceries</option>
                        <option value="Entertainment">Entertainment</option>
                        <option value="Health">Health</option>
                        <option value="Other">Other</option>
                    </select>
                    <div style="display:flex; gap:10px;">
                        <button type="button" id="btnInc" style="flex:1;">Income</button>
                        <button type="button" id="btnExp" style="flex:1;">Expense</button>
                    </div>
                    <input type="hidden" id="txType" value="expense">
                    <input type="text" id="txNote" placeholder="Note (optional)">
                    <button type="submit" class="btn-green">Save Transaction</button>
                </form>
            </div>
            <div class="panel">
                <div class="panel-header"><div class="panel-title">🧠 AI Budget Engine (Flowchart)</div><span class="ml-badge" style="background: linear-gradient(135deg, #a855f7, #3b82f6);">ML v2.0</span></div>
                <div id="budgetEngineStatus">⚙️ Set your profile to generate smart scenarios</div>
                <div style="margin-top: 16px;">
                    <label>Social Status</label>
                    <select id="socialStatus" class="budget-input">
                        <option value="Low">Low Class</option>
                        <option value="Middle" selected>Middle Class</option>
                        <option value="Upper">Upper Class</option>
                    </select>
                </div>
                <div style="margin-top: 12px;">
                    <label>Spending Mindset</label>
                    <select id="spendingMindset" class="budget-input">
                        <option value="Saver">Saver 💰</option>
                        <option value="Neutral" selected>Neutral ⚖️</option>
                        <option value="Spender">Spender 💸</option>
                    </select>
                </div>
                <button id="saveProfileBtn" class="btn-green" style="margin: 16px 0;">Save Profile & Generate Scenarios</button>
                <div id="scenariosContainer" style="margin-top: 20px;"></div>
            </div>
        </div>
    </div>

    <!-- ANALYTICS SCREEN -->
    <div class="screen" id="screen-analytics">
        <div class="chart-grid">
            <div class="panel"><div class="panel-header"><div class="panel-title">Weekly Spending Forecast <span class="ml-badge" style="background: linear-gradient(135deg, #a855f7, #3b82f6);">ML PREDICTION</span></div></div><canvas id="forecastChart" height="200"></canvas></div>
            <div class="panel"><div class="panel-header"><div class="panel-title">Category Spending Breakdown</div></div><canvas id="catBarChart" height="200"></canvas></div>
        </div>
        <div class="panel"><div class="panel-header"><div class="panel-title">ML Intelligence Report</div></div><div id="fullAdvice"></div></div>
    </div>

    <!-- BUDGETS SCREEN -->
    <div class="screen" id="screen-budgets">
        <div class="panel">
            <div class="panel-header"><div class="panel-title">Monthly Budget Limits</div><button class="btn-green" id="saveBudgetsBtn">Save All</button></div>
            <div id="budgetInputs" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;"></div>
        </div>
    </div>

    <!-- TRANSACTIONS HISTORY SCREEN -->
    <div class="screen" id="screen-transactions">
        <div class="table-wrap">
            <table>
                <thead><tr><th>Date</th><th>Category</th><th>Note</th><th>Type</th><th>Amount</th></tr></thead>
                <tbody id="txTableBody"></tbody>
            </table>
        </div>
    </div>

    <!-- AUTH OVERLAY -->
    <div id="authOverlay" class="auth-overlay">
        <div class="auth-card">
            <h2 id="authTitle">Welcome back</h2>
            <input type="text" id="regName" class="auth-input" placeholder="Full Name" style="display:none">
            <input type="email" id="authEmail" class="auth-input" placeholder="Email">
            <input type="password" id="authPass" class="auth-input" placeholder="Password">
            <input type="password" id="authConfirm" class="auth-input" placeholder="Confirm Password" style="display:none">
            <div id="termsRow" style="display:none; margin-bottom:14px;"><label><input type="checkbox" id="termsCheck"> Accept Terms</label></div>
            <div id="authMsg" style="color:#ff4d6d; font-size:0.8rem; margin-bottom:12px;"></div>
            <button id="authBtn" class="auth-btn">Sign In</button>
            <div id="toggleAuthLink" class="auth-link">Don't have an account? Register</div>
        </div>
    </div>
</main>

<script>
let currentUser = null;
let allTransactions = [];
let currentSummary = { balance: 0, expense: 0, income: 0 };
let currentPrediction = { has_data: false, score: null, predictions: { weekly: null, categories: {} }, advice: [] };
let monthlyChart, donutChart, forecastChart, catBarChart;
let isLogin = true;

function fmt(amt) { return '₱' + Number(amt).toLocaleString('en-PH', { minimumFractionDigits: 2 }); }
function toast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 2500); }
async function apiFetch(url, opts = {}) {
    const res = await fetch(url, { ...opts, credentials: 'include', headers: { 'Content-Type': 'application/json' } });
    if (!res.ok) { const err = await res.json(); throw new Error(err.error || 'Request failed'); }
    return res.json();
}
function updateHealthScoreDisplay(score) {
    const scoreEl = document.getElementById('sScore');
    const topEl = document.getElementById('topScore');
    const pill = document.getElementById('scorePill');
    if (score === null || score === undefined) {
        if(scoreEl) scoreEl.innerText = '—';
        if(topEl) topEl.innerText = '—';
        if(scoreEl) scoreEl.style.color = 'var(--muted)';
        if(pill) pill.classList.remove('green','red');
        return;
    }
    scoreEl.innerText = score + '/100';
    topEl.innerText = score;
    if (score >= 70) {
        scoreEl.style.color = '#00d282';
        pill.classList.add('green');
        pill.classList.remove('red');
    } else {
        scoreEl.style.color = '#ff4d6d';
        pill.classList.add('red');
        pill.classList.remove('green');
    }
}
function addWasteAlert() {
    if (currentPrediction.has_data && currentPrediction.score !== null && currentPrediction.score < 70) {
        const adviceDiv = document.getElementById('adviceList');
        if (adviceDiv && !document.getElementById('wasteAlert')) {
            const alertDiv = document.createElement('div');
            alertDiv.id = 'wasteAlert';
            alertDiv.className = 'advice-item waste-alert';
            alertDiv.innerHTML = `<strong>⚠️ WASTE ALERT</strong><br>Your health score (${currentPrediction.score}/100) is low. Review discretionary spending.`;
            adviceDiv.insertBefore(alertDiv, adviceDiv.firstChild);
        }
    } else {
        const existing = document.getElementById('wasteAlert');
        if (existing) existing.remove();
    }
}

async function saveProfileAndGenerate() {
    const social = document.getElementById('socialStatus').value;
    const mindset = document.getElementById('spendingMindset').value;
    try {
        await apiFetch('/api/user/profile', {
            method: 'POST',
            body: JSON.stringify({ social_status: social, spending_mindset: mindset })
        });
        toast('Profile saved. Generating scenarios...');
        await displayScenarios();
    } catch(e) { toast('Error saving profile'); }
}

async function displayScenarios() {
    if (!currentUser) return;
    const res = await apiFetch(`/api/budget/scenarios/${currentUser.id}`);
    const scenarios = res.scenarios;
    const container = document.getElementById('scenariosContainer');
    if (!scenarios || scenarios.length === 0) {
        container.innerHTML = '<div class="advice-item">No data yet. Add transactions first.</div>';
        return;
    }
    container.innerHTML = '<h4 style="margin-bottom: 12px;">🎯 ML‑Generated Scenarios (Step 3)</h4>';
    for (let s of scenarios) {
        const card = document.createElement('div');
        card.className = 'advice-item';
        card.style.cursor = 'pointer';
        card.innerHTML = `
            <strong>${s.name}</strong><br>
            💾 Savings Goal: ₱${s.savings_goal.toLocaleString()} | 📈 Investment: ₱${s.investment_goal.toLocaleString()}<br>
            <span style="font-size: 0.75rem;">Expected savings rate: ${s.expected_savings_rate}%</span>
            <div style="margin-top: 8px;">
                <button class="btn-green preview-scenario" data-limits='${JSON.stringify(s.budget_limits)}'>Preview Budget Limits</button>
                <button class="btn approve-scenario" data-limits='${JSON.stringify(s.budget_limits)}' style="background: var(--blue); margin-left: 8px;">✅ Approve & Apply</button>
            </div>
        `;
        container.appendChild(card);
    }
    document.querySelectorAll('.preview-scenario').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const limits = JSON.parse(btn.dataset.limits);
            showBudgetPreview(limits);
        });
    });
    document.querySelectorAll('.approve-scenario').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const limits = JSON.parse(btn.dataset.limits);
            await apiFetch(`/api/budgets/bulk/${currentUser.id}`, {
                method: 'POST',
                body: JSON.stringify({ limits })
            });
            toast('Budgets applied! Analytics & forecast updated.');
            await loadDashboard();
            navigate('dashboard');
        });
    });
}

function showBudgetPreview(limits) {
    let msg = 'Proposed Monthly Budgets:\\n';
    for (let [cat, limit] of Object.entries(limits)) {
        msg += `${cat}: ₱${limit.toLocaleString()}\\n`;
    }
    alert(msg);
}

async function loadUserProfile() {
    if (!currentUser) return;
    try {
        const profile = await apiFetch('/api/user/profile');
        document.getElementById('socialStatus').value = profile.social_status;
        document.getElementById('spendingMindset').value = profile.spending_mindset;
    } catch(e) {}
}

async function loadBudgets() {
    if(!currentUser) return;
    const budgets = await apiFetch(`/api/budgets/${currentUser.id}`);
    const limits = Object.fromEntries(budgets.map(b => [b.category, b.limit]));
    const categories = ['Food & Dining', 'Transport', 'Groceries', 'Entertainment', 'Health'];
    const container = document.getElementById('budgetInputs');
    if(!container) return;
    container.innerHTML = categories.map(cat => `<div><label>${cat}</label><input type="number" id="budget_${cat.replace(/\\s/g,'')}" value="${limits[cat]||''}" placeholder="Limit" style="width:100%"></div>`).join('');
    document.getElementById('saveBudgetsBtn').onclick = async () => {
        for(const cat of categories) {
            const val = parseFloat(document.getElementById(`budget_${cat.replace(/\\s/g,'')}`).value);
            if(val && val>0) await apiFetch(`/api/budgets/${currentUser.id}`, { method: 'POST', body: JSON.stringify({ category: cat, limit: val }) });
        }
        toast('Budgets saved!');
        await loadDashboard();
    };
}

document.getElementById('saveProfileBtn').addEventListener('click', saveProfileAndGenerate);

async function loadDashboard() {
    if (!currentUser) return;
    try {
        const summary = await apiFetch(`/api/summary/${currentUser.id}`);
        const pred = await apiFetch(`/api/predict/${currentUser.id}`);
        allTransactions = await apiFetch(`/api/transactions?user_id=${currentUser.id}`);
        currentSummary = summary;
        currentPrediction = pred;

        document.getElementById('sBalance').innerText = fmt(summary.balance);
        document.getElementById('sExpense').innerText = fmt(summary.expense);
        document.getElementById('sIncome').innerText = fmt(summary.income);

        if (!pred.has_data) {
            updateHealthScoreDisplay(null);
            document.getElementById('forecastBars').innerHTML = '<div class="empty-state">📊 No transactions yet. Add your first income or expense to see ML spending forecast.</div>';
            document.getElementById('adviceList').innerHTML = '<div class="advice-item">💡 Add transactions to receive AI‑powered financial advice and health score.</div>';
            if (monthlyChart) monthlyChart.destroy();
            if (donutChart) donutChart.destroy();
            const monthlyCtx = document.getElementById('monthlyChart').getContext('2d');
            monthlyCtx.clearRect(0,0, document.getElementById('monthlyChart').width, document.getElementById('monthlyChart').height);
            const donutCtx = document.getElementById('donutChart').getContext('2d');
            donutCtx.clearRect(0,0, document.getElementById('donutChart').width, document.getElementById('donutChart').height);
            document.getElementById('recentList').innerHTML = '<div class="empty-state">No recent transactions</div>';
            return;
        }

        updateHealthScoreDisplay(pred.score);
        if (monthlyChart) monthlyChart.destroy();
        const months = Object.keys(summary.monthly).slice(-6);
        monthlyChart = new Chart(document.getElementById('monthlyChart'), {
            type: 'bar',
            data: { labels: months, datasets: [
                { label: 'Income', data: months.map(m => summary.monthly[m]?.income || 0), backgroundColor: 'rgba(0,210,130,0.6)', borderRadius: 8 },
                { label: 'Expense', data: months.map(m => summary.monthly[m]?.expense || 0), backgroundColor: 'rgba(255,77,109,0.6)', borderRadius: 8 }
            ]}
        });
        if (donutChart) donutChart.destroy();
        const catData = pred.predictions.categories;
        donutChart = new Chart(document.getElementById('donutChart'), { type: 'doughnut', data: { labels: Object.keys(catData), datasets: [{ data: Object.values(catData), backgroundColor: ['#00d282','#3b82f6','#f0a500','#a855f7','#ff4d6d'] }] }, options: { cutout: '65%' } });

        const weeks = pred.predictions.weekly || {};
        const maxVal = Object.values(weeks).length ? Math.max(...Object.values(weeks), 1) : 1;
        document.getElementById('forecastBars').innerHTML = Object.entries(weeks).map(([w,v]) => `<div class="forecast-row"><span class="forecast-week">${w}</span><div class="forecast-bar-track"><div class="forecast-bar-fill" style="width:${(v/maxVal*100)}%"></div></div><span>${fmt(v)}</span></div>`).join('');
        document.getElementById('adviceList').innerHTML = pred.advice.map(a => `<div class="advice-item"><strong>${a.cat}</strong>: ${a.msg}</div>`).join('');
        addWasteAlert();
        document.getElementById('recentList').innerHTML = allTransactions.slice(0,5).map(t => `<div style="padding:10px 0; border-bottom:1px solid var(--border2);"><strong>${t.category}</strong><br>${fmt(t.amount)}<br><small>${new Date(t.tx_date).toLocaleDateString()}</small></div>`).join('');
        renderTransactions();
        loadAnalytics();
    } catch(e) { console.error(e); toast('Error loading data'); }
}
function renderTransactions() {
    if(!currentUser) return;
    document.getElementById('txTableBody').innerHTML = allTransactions.slice(0,50).map(t => `<tr><td>${new Date(t.tx_date).toLocaleDateString()}</td><td>${t.category}</td><td>${t.note||'—'}</td><td style="color:${t.tx_type==='income'?'#00d282':'#ff4d6d'}">${t.tx_type}</td><td>${fmt(t.amount)}</td></tr>`).join('');
}
async function loadAnalytics() {
    if(!currentUser) return;
    const pred = currentPrediction;
    if(!pred.has_data) {
        if(forecastChart) forecastChart.destroy();
        if(catBarChart) catBarChart.destroy();
        const fcCtx = document.getElementById('forecastChart').getContext('2d');
        fcCtx.clearRect(0,0, document.getElementById('forecastChart').width, document.getElementById('forecastChart').height);
        const catCtx = document.getElementById('catBarChart').getContext('2d');
        catCtx.clearRect(0,0, document.getElementById('catBarChart').width, document.getElementById('catBarChart').height);
        document.getElementById('fullAdvice').innerHTML = '<div class="advice-item">📊 Add transactions to see ML predictions and category breakdown.</div>';
        return;
    }
    if(forecastChart) forecastChart.destroy();
    forecastChart = new Chart(document.getElementById('forecastChart'), { type: 'line', data: { labels: Object.keys(pred.predictions.weekly), datasets: [{ label: 'ML Predicted Spend', data: Object.values(pred.predictions.weekly), borderColor: '#00d282', tension: 0.3, fill: true, backgroundColor: 'rgba(0,210,130,0.1)' }] }, options: { responsive: true, plugins: { tooltip: { callbacks: { label: (ctx) => '₱' + ctx.raw.toLocaleString('en-PH') } } } } });
    if(catBarChart) catBarChart.destroy();
    catBarChart = new Chart(document.getElementById('catBarChart'), { type: 'bar', data: { labels: Object.keys(pred.predictions.categories), datasets: [{ label: 'Total Spent', data: Object.values(pred.predictions.categories), backgroundColor: '#3b82f6', borderRadius: 8 }] }, options: { indexAxis: 'y' } });
    document.getElementById('fullAdvice').innerHTML = pred.advice.map(a => `<div class="advice-item"><strong>${a.cat}</strong><br>${a.msg}</div>`).join('');
}

// Navigation
document.querySelectorAll('.nav-item').forEach(btn => btn.addEventListener('click', () => { const screen = btn.getAttribute('data-nav'); if(screen) navigate(screen); }));
function navigate(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(`screen-${screenId}`).classList.add('active');
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
    const activeBtn = Array.from(document.querySelectorAll('.nav-item')).find(btn => btn.getAttribute('data-nav') === screenId);
    if(activeBtn) activeBtn.classList.add('active');
    const titles = { dashboard:'Dashboard', add:'Add Transaction', analytics:'Analytics', budgets:'Budgets', transactions:'History' };
    document.getElementById('pageTitle').innerText = titles[screenId]||'SmartSpend';
    if(screenId === 'analytics') loadAnalytics();
    if(screenId === 'budgets') loadBudgets();
    if(screenId === 'transactions') renderTransactions();
}

// Transaction form buttons
document.getElementById('btnExp').addEventListener('click', () => { document.getElementById('txType').value = 'expense'; document.getElementById('btnExp').style.opacity='1'; document.getElementById('btnInc').style.opacity='0.5'; });
document.getElementById('btnInc').addEventListener('click', () => { document.getElementById('txType').value = 'income'; document.getElementById('btnInc').style.opacity='1'; document.getElementById('btnExp').style.opacity='0.5'; });
document.getElementById('txForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = { amount: parseFloat(document.getElementById('txAmount').value), category: document.getElementById('txCategory').value, tx_type: document.getElementById('txType').value, note: document.getElementById('txNote').value };
    try { await apiFetch('/api/transactions', { method:'POST', body: JSON.stringify(data) }); toast('Transaction saved!'); document.getElementById('txForm').reset(); await loadDashboard(); } catch(e) { toast('Error'); }
});

// Auth
document.getElementById('signoutBtn').addEventListener('click', logout);
document.getElementById('toggleAuthLink').addEventListener('click', toggleAuthMode);
document.getElementById('authBtn').addEventListener('click', handleAuth);
function toggleAuthMode() {
    isLogin = !isLogin;
    document.getElementById('authTitle').innerText = isLogin ? 'Welcome back' : 'Create Account';
    document.getElementById('regName').style.display = isLogin ? 'none' : 'block';
    document.getElementById('authConfirm').style.display = isLogin ? 'none' : 'block';
    document.getElementById('termsRow').style.display = isLogin ? 'none' : 'flex';
    document.getElementById('authBtn').innerText = isLogin ? 'Sign In' : 'Create Account';
    document.getElementById('toggleAuthLink').innerHTML = isLogin ? "Don't have an account? Register" : "Already have an account? Sign In";
    document.getElementById('authMsg').innerHTML = '';
}
async function handleAuth() {
    const email = document.getElementById('authEmail').value;
    const pass = document.getElementById('authPass').value;
    const msgDiv = document.getElementById('authMsg');
    if(!email || !pass) { msgDiv.innerText = 'Email and password required'; return; }
    if(!isLogin) {
        const name = document.getElementById('regName').value;
        const confirm = document.getElementById('authConfirm').value;
        const terms = document.getElementById('termsCheck').checked;
        if(!name) { msgDiv.innerText = 'Full name required'; return; }
        if(pass !== confirm) { msgDiv.innerText = 'Passwords do not match'; return; }
        if(!terms) { msgDiv.innerText = 'Accept terms'; return; }
        if(pass.length < 8) { msgDiv.innerText = 'Password must be at least 8 characters'; return; }
        try {
            const res = await fetch('/api/register', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name, email, password: pass, accepted_terms: true }), credentials:'include' });
            const data = await res.json();
            if(!res.ok) throw new Error(data.error);
            msgDiv.style.color = '#00d282';
            msgDiv.innerText = 'Account created! Please login.';
            setTimeout(() => toggleAuthMode(), 1500);
        } catch(e) { msgDiv.innerText = e.message; }
        return;
    }
    try {
        const res = await fetch('/api/login', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ email, password: pass }), credentials:'include' });
        const data = await res.json();
        if(!res.ok) throw new Error(data.error);
        currentUser = data.user;
        document.getElementById('authOverlay').style.display = 'none';
        document.getElementById('userAvatar').innerText = currentUser.name.slice(0,2).toUpperCase();
        await loadUserProfile();
        await loadDashboard();
        navigate('dashboard');
    } catch(e) { msgDiv.innerText = e.message; }
}
async function logout() {
    await fetch('/api/logout', { method:'POST', credentials:'include' });
    currentUser = null;
    document.getElementById('authOverlay').style.display = 'flex';
    document.getElementById('userAvatar').innerText = 'US';
    location.reload();
}
async function init() {
    try {
        const res = await fetch('/api/me', { credentials:'include' });
        if(res.ok) {
            const user = await res.json();
            currentUser = user;
            document.getElementById('authOverlay').style.display = 'none';
            document.getElementById('userAvatar').innerText = user.name.slice(0,2).toUpperCase();
            await loadUserProfile();
            await loadDashboard();
        } else {
            document.getElementById('authOverlay').style.display = 'flex';
        }
    } catch(e) { console.log('Not authenticated'); document.getElementById('authOverlay').style.display = 'flex'; }
}
init();
</script>
</body>
</html>
"""
