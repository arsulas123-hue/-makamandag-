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

# ---------- ML Functions ----------
def calculate_health_score(transactions, budgets):
    """Compute health score (0-100) based on savings rate and budget adherence."""
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    expense = sum(t['amount'] for t in transactions if t['tx_type'] == 'expense')
    if income == 0:
        return 0
    savings_rate = (income - expense) / income
    score = max(0, min(100, savings_rate * 100))  # base on savings rate
    
    # Budget adherence factor
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
                budget_penalty += over * 10  # max penalty per category
    if total_budget > 0:
        score = max(0, min(100, score - budget_penalty))
    return int(score)

def forecast_spending(transactions):
    """Return weekly forecast for next 4 weeks using simple moving average."""
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    if not expenses:
        return {"Week 1": 0, "Week 2": 0, "Week 3": 0, "Week 4": 0}
    # group by week
    weekly = {}
    for t in expenses:
        date = t['tx_date'] if isinstance(t['tx_date'], datetime.datetime) else datetime.datetime.fromisoformat(t['tx_date'])
        week_num = date.isocalendar()[1]
        year = date.year
        key = f"{year}-W{week_num}"
        weekly[key] = weekly.get(key, 0) + t['amount']
    weekly_vals = list(weekly.values())
    if len(weekly_vals) < 2:
        avg = sum(weekly_vals) / max(1, len(weekly_vals))
    else:
        # simple moving average of last 3 weeks
        window = weekly_vals[-3:] if len(weekly_vals) >= 3 else weekly_vals
        avg = sum(window) / len(window)
    # Generate next 4 weeks
    forecast = {}
    for i in range(1, 5):
        forecast[f"Week {i}"] = round(avg * (0.95 + i * 0.02), 2)  # slight trend
    return forecast

def generate_advice(transactions, budgets):
    """Generate AI advice per category based on budget vs actual."""
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
    """Simple rule-based chatbot with profile context."""
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

# ---------- API ROUTES ----------
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        accepted_terms = data.get('accepted_terms', False)
        if not name or not email or not password:
            return jsonify({'error': 'Missing fields'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        if not accepted_terms:
            return jsonify({'error': 'Must accept terms'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        hashed = hash_password(password)
        new_user = User(name=name, email=email, password=hashed, role='user')
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User created'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        user = User.query.filter_by(email=email).first()
        if not user or user.password != hash_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        if not user.is_active:
            return jsonify({'error': 'Account disabled'}), 401
        session['user_id'] = user.id
        return jsonify({'user': {'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'})

@app.route('/api/me')
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return jsonify({'error': 'User not found'}), 401
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email, 'role': user.role})

@app.route('/api/user/profile', methods=['GET'])
@login_required
def get_profile():
    user_id = session['user_id']
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        return jsonify({'social_status': 'Middle', 'spending_mindset': 'Neutral', 'wants_needs': {}})
    return jsonify({
        'social_status': profile.social_status,
        'spending_mindset': profile.spending_mindset,
        'wants_needs': json.loads(profile.wants_needs_json) if profile.wants_needs_json else {}
    })

@app.route('/api/user/profile', methods=['POST'])
@login_required
def update_profile():
    user_id = session['user_id']
    data = request.json
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = UserProfile(user_id=user_id)
        db.session.add(profile)
    profile.social_status = data.get('social_status', 'Middle')
    profile.spending_mindset = data.get('spending_mindset', 'Neutral')
    profile.wants_needs_json = json.dumps(data.get('wants_needs', {}))
    db.session.commit()
    return jsonify({'message': 'Profile saved'})

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

@app.route('/api/budget/scenarios/<int:user_id>', methods=['GET'])
@login_required
def get_scenarios(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).all()
    txs_data = [{'amount': t.amount, 'tx_type': t.tx_type, 'category': t.category, 'tx_date': t.tx_date.isoformat()} for t in txs]
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    scenarios = generate_scenarios(user_id, txs_data, budgets)
    return jsonify({'scenarios': scenarios})

@app.route('/api/budgets/bulk/<int:user_id>', methods=['POST'])
@login_required
def bulk_save_budgets(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    data = request.json
    limits = data.get('limits', {})
    for category, limit_amount in limits.items():
        budget = Budget.query.filter_by(user_id=user_id, category=category).first()
        if budget:
            budget.limit_amount = float(limit_amount)
        else:
            budget = Budget(user_id=user_id, category=category, limit_amount=float(limit_amount))
            db.session.add(budget)
    db.session.commit()
    return jsonify({'message': 'Budgets updated from scenario'})

@app.route('/api/budgets/<int:user_id>', methods=['GET'])
@login_required
def get_budgets(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    budgets = Budget.query.filter_by(user_id=user_id).all()
    return jsonify([{'category': b.category, 'limit': b.limit_amount} for b in budgets])

@app.route('/api/budgets/<int:user_id>', methods=['POST'])
@login_required
def set_budget(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    data = request.json
    category = data.get('category')
    limit = data.get('limit')
    if not category or limit is None:
        return jsonify({'error': 'Missing category or limit'}), 400
    budget = Budget.query.filter_by(user_id=user_id, category=category).first()
    if budget:
        budget.limit_amount = float(limit)
    else:
        budget = Budget(user_id=user_id, category=category, limit_amount=float(limit))
        db.session.add(budget)
    db.session.commit()
    return jsonify({'message': 'Budget saved'})

@app.route('/api/transactions', methods=['GET'])
@login_required
def get_transactions():
    user_id = request.args.get('user_id', type=int)
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=session['user_id']).order_by(Transaction.tx_date.desc()).all()
    return jsonify([{'tx_id': t.id, 'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'note': t.note or '', 'tx_date': t.tx_date.isoformat()} for t in txs])

@app.route('/api/transactions', methods=['POST'])
@login_required
def add_transaction():
    try:
        data = request.json
        tx = Transaction(user_id=session['user_id'], amount=float(data['amount']), category=data['category'], tx_type=data['tx_type'], note=data.get('note', ''))
        db.session.add(tx)
        db.session.commit()
        return jsonify({'message': 'Transaction saved', 'id': tx.id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).all()
    txs_data = [{'amount': t.amount, 'tx_type': t.tx_type, 'tx_date': t.tx_date, 'category': t.category} for t in txs]
    total_income = sum(t['amount'] for t in txs_data if t['tx_type'] == 'income')
    total_expense = sum(t['amount'] for t in txs_data if t['tx_type'] == 'expense')
    monthly = {}
    for t in txs_data:
        key = t['tx_date'].strftime('%Y-%m')
        if key not in monthly:
            monthly[key] = {'income': 0, 'expense': 0}
        if t['tx_type'] == 'income':
            monthly[key]['income'] += t['amount']
        else:
            monthly[key]['expense'] += t['amount']
    return jsonify({'balance': total_income - total_expense, 'income': total_income, 'expense': total_expense, 'monthly': monthly})

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).all()
    if not txs:
        return jsonify({'has_data': False, 'score': None, 'predictions': {'weekly': {}, 'categories': {}}, 'advice': []})
    txs_data = [{'amount': t.amount, 'tx_type': t.tx_type, 'category': t.category, 'tx_date': t.tx_date} for t in txs]
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    score = calculate_health_score(txs_data, budgets)
    weekly_forecast = forecast_spending(txs_data)
    cat_spending = {}
    for t in txs_data:
        if t['tx_type'] == 'expense':
            cat_spending[t['category']] = cat_spending.get(t['category'], 0) + t['amount']
    advice = generate_advice(txs_data, budgets)
    return jsonify({
        'has_data': True,
        'score': score,
        'predictions': {'weekly': weekly_forecast, 'categories': cat_spending},
        'advice': advice
    })

@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.json
    user_message = data.get('message', '')
    user_id = session['user_id']
    txs = Transaction.query.filter_by(user_id=user_id).all()
    txs_data = [{'amount': t.amount, 'tx_type': t.tx_type, 'category': t.category, 'tx_date': t.tx_date.isoformat()} for t in txs]
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    profile = get_user_profile(user_id)
    response = generate_chatbot_response_with_profile(user_message, txs_data, budgets, profile)
    return jsonify({'response': response})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
