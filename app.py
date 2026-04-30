import os, datetime, hashlib, secrets, json
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False

DATABASE_URL = os.environ.get('DATABASE_URL', 
    "postgresql://makamandag_db_user:zcDibuXdlpEpcZNGEYLc9nqpgWwuTTfO@dpg-d7od7md7vvec739acfj0-a/makamandag_db")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
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

# ---------- ML Functions (FIXED) ----------
def calculate_health_score(transactions, budgets):
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    expense = sum(t['amount'] for t in transactions if t['tx_type'] == 'expense')
    if income == 0:
        return 0   # no income → cannot save
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
    """Forecast next 4 weeks – works with both datetime and ISO string dates."""
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

def generate_chatbot_response_with_profile(message, transactions, budgets, profile):
    msg_lower = message.lower()
    if "forecast" in msg_lower:
        forecast = forecast_spending(transactions)
        return f"Based on your spending patterns, the next 4 weeks forecast: {forecast}"
    elif "health score" in msg_lower:
        score = calculate_health_score(transactions, budgets)
        return f"Your current ML health score is {score}/100. {'Keep it up!' if score >= 70 else 'Try to increase savings and control overspending.'}"
    elif "investment" in msg_lower:
        social = profile['social_status']
        if social == 'Upper':
            return "Given your upper-class profile, consider diversifying into stocks and real estate. Aim to invest 30% of income."
        else:
            return "Start with low-cost index funds or micro-investing apps. Aim to invest at least 10% of savings."
    else:
        return "I can help with spending forecasts, health scores, or investment advice. Try asking: 'ML forecast', 'Health score', or 'Investment advice'."

def generate_scenarios(user_id, transactions, budgets):
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

# ========== API ROUTES (all present) ==========
# ... (keep all your existing routes: register, login, logout, me, user/profile, budgets, transactions, summary, predict, chatbot, static serving)
# I'm not repeating them because they are correct in your original code.
# Only the ML helpers above have been fixed.

# Make sure to include the routes you already wrote – they are unchanged.
# The following is a placeholder; your actual routes should remain exactly as you had them.

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
