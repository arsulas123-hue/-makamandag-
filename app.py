import os, datetime, hashlib, secrets
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.linear_model import LinearRegression
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True

DATABASE_URL = "postgresql://makamandag_db_user:zcDibuXdlpEpcZNGEYLc9nqpgWwuTTfO@dpg-d7od7md7vvec739acfj0-a/makamandag_db"
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- Models ----------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    login_count = db.Column(db.Integer, default=0)
    last_login = db.Column(db.DateTime)
    last_ip = db.Column(db.String(50))
    accepted_terms = db.Column(db.Boolean, default=False)
    terms_version = db.Column(db.String(10))
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
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'category'),)

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(50), nullable=False)
    ip = db.Column(db.String(50))
    detail = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# ---------- NEW TABLES FOR LEARNING ----------
class ForecastLog(db.Model):
    __tablename__ = 'forecast_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    week_start = db.Column(db.Date, nullable=False)
    predicted_amount = db.Column(db.Float, nullable=False)
    actual_amount = db.Column(db.Float, nullable=True)
    error = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class AdviceFeedback(db.Model):
    __tablename__ = 'advice_feedback'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category = db.Column(db.String(50))
    advice_text = db.Column(db.String(500))
    helpful = db.Column(db.Boolean)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()
    if User.query.filter_by(role='admin').first() is None:
        hashed = hashlib.sha256('admin123'.encode()).hexdigest()
        admin = User(name='Admin', email='admin@smartspend.com', password=hashed, role='admin', accepted_terms=True)
        db.session.add(admin)
        db.session.commit()

# ---------- Helper functions ----------
def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def log_audit(user_id, action, ip, detail=''):
    log = AuditLog(user_id=user_id, action=action, ip=ip, detail=detail)
    db.session.add(log)
    db.session.commit()

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

# ---------- ADAPTIVE ML FUNCTIONS ----------
def forecast_spending(user_id, transactions, weeks=4):
    # Get past average error
    past_forecasts = ForecastLog.query.filter_by(user_id=user_id, actual_amount!=None).all()
    avg_error = np.mean([f.error for f in past_forecasts]) if past_forecasts else 0

    expenses = [t for t in transactions if t.tx_type == 'expense']
    if len(expenses) < 3:
        avg = np.mean([t.amount for t in expenses]) if expenses else 2000
        base_pred = {f'Week {i+1}': round(avg * (0.9 + 0.2 * np.random.random())) for i in range(weeks)}
    else:
        expenses_sorted = sorted(expenses, key=lambda x: x.tx_date)
        weekly_totals = []
        current_week_num = expenses_sorted[0].tx_date.isocalendar()[1]
        current_total = 0
        for tx in expenses_sorted:
            week_num = tx.tx_date.isocalendar()[1]
            if week_num != current_week_num:
                weekly_totals.append(current_total)
                current_total = tx.amount
                current_week_num = week_num
            else:
                current_total += tx.amount
        if current_total > 0:
            weekly_totals.append(current_total)

        if len(weekly_totals) >= 2:
            X = np.array(range(len(weekly_totals))).reshape(-1,1)
            y = np.array(weekly_totals)
            model = LinearRegression()
            model.fit(X, y)
            future = np.array(range(len(weekly_totals), len(weekly_totals)+weeks)).reshape(-1,1)
            predictions = model.predict(future)
            predictions = np.maximum(predictions, 0)
            base_pred = {f'Week {i+1}': round(predictions[i]) for i in range(weeks)}
        else:
            avg = np.mean(weekly_totals) if weekly_totals else 2000
            base_pred = {f'Week {i+1}': round(avg) for i in range(weeks)}

    # Adjust by past error
    adjusted_pred = {week: max(0, round(val + avg_error)) for week, val in base_pred.items()}

    # Store forecast for future comparison
    today = datetime.date.today()
    for i, (week, amount) in enumerate(adjusted_pred.items()):
        week_start = today + datetime.timedelta(days=7*i)
        # Avoid duplicates
        existing = ForecastLog.query.filter_by(user_id=user_id, week_start=week_start).first()
        if not existing:
            log_entry = ForecastLog(user_id=user_id, week_start=week_start, predicted_amount=amount)
            db.session.add(log_entry)
    db.session.commit()
    return adjusted_pred

def calculate_health_score(user_id, transactions, budgets):
    expenses = [t for t in transactions if t.tx_type == 'expense']
    income = sum(t.amount for t in transactions if t.tx_type == 'income')
    total_expense = sum(e.amount for e in expenses)
    score = 70
    if income > 0:
        savings_rate = (income - total_expense) / income
        score += min(20, max(0, savings_rate * 40))
    compliance_score = 0
    budget_count = 0
    for cat, limit in budgets.items():
        spent = sum(e.amount for e in expenses if e.category == cat)
        if limit > 0:
            budget_count += 1
            if spent <= limit:
                compliance_score += 1
            elif spent <= limit * 1.15:
                compliance_score += 0.5
    if budget_count > 0:
        score += (compliance_score / budget_count) * 10
    if len(expenses) > 3:
        amounts = [e.amount for e in expenses[-12:]]
        if len(amounts) > 1:
            volatility = np.std(amounts) / (np.mean(amounts) + 0.01)
            score -= min(10, volatility * 2)
    return max(0, min(100, round(score)))

def generate_advice(user_id, transactions, budgets):
    expenses = [t for t in transactions if t.tx_type == 'expense']
    cat_spending = {}
    for exp in expenses:
        cat_spending[exp.category] = cat_spending.get(exp.category, 0) + exp.amount
    advice = []
    for cat, spent in cat_spending.items():
        limit = budgets.get(cat, 0)
        if limit > 0:
            pct = (spent / limit) * 100
            if pct > 100:
                advice.append({'cat': cat, 'spent': spent, 'limit': limit, 'pct': round(pct), 'status': 'over', 'msg': f'exceeded by {round(pct-100)}%'})
            elif pct > 85:
                advice.append({'cat': cat, 'spent': spent, 'limit': limit, 'pct': round(pct), 'status': 'warning', 'msg': f'approaching limit ({round(pct)}%)'})
            else:
                advice.append({'cat': cat, 'spent': spent, 'limit': limit, 'pct': round(pct), 'status': 'ok', 'msg': 'on track'})
        else:
            if spent > 5000:
                advice.append({'cat': cat, 'spent': spent, 'limit': None, 'pct': 0, 'status': 'warning', 'msg': f'high spending (₱{spent:,.2f}) - consider budget'})
    # Prioritize advice that user found helpful in the past
    helpful_feedback = AdviceFeedback.query.filter_by(user_id=user_id, helpful=True).all()
    helpful_cats = set(fb.category for fb in helpful_feedback if fb.category)
    advice.sort(key=lambda x: (x['cat'] not in helpful_cats, x['status'] != 'over'), reverse=False)
    return advice

# ---------- API ROUTES (keep all your existing ones) ----------
# ... (your existing /api/register, /api/login, /api/logout, /api/me, /api/transactions, /api/summary, /api/predict, /api/budgets, /api/admin/*)
# Only changes: /api/predict now uses forecast_spending above; /api/advice_feedback is new.

@app.route('/api/advice_feedback', methods=['POST'])
@login_required
def advice_feedback():
    data = request.json
    helpful = data.get('helpful')
    category = data.get('category')
    advice_text = data.get('advice_text')
    feedback = AdviceFeedback(
        user_id=session['user_id'],
        category=category,
        advice_text=advice_text,
        helpful=helpful
    )
    db.session.add(feedback)
    db.session.commit()
    log_audit(session['user_id'], 'advice_feedback', request.remote_addr, f'{category} helpful={helpful}')
    return jsonify({'message': 'Feedback recorded'})

# ---------- WEEKLY FORECAST CORRECTION JOB ----------
def update_forecast_errors():
    with app.app_context():
        one_week_ago = datetime.date.today() - datetime.timedelta(days=7)
        forecasts = ForecastLog.query.filter(
            ForecastLog.week_start <= one_week_ago,
            ForecastLog.actual_amount == None
        ).all()
        for f in forecasts:
            week_end = f.week_start + datetime.timedelta(days=7)
            actual = db.session.query(db.func.sum(Transaction.amount)).filter(
                Transaction.user_id == f.user_id,
                Transaction.tx_type == 'expense',
                Transaction.tx_date >= f.week_start,
                Transaction.tx_date < week_end
            ).scalar() or 0
            f.actual_amount = actual
            f.error = actual - f.predicted_amount
        db.session.commit()
        print(f"Updated {len(forecasts)} forecast records")

# Start scheduler (only once)
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_forecast_errors, trigger="interval", days=7)
scheduler.start()

# ---------- Serve Frontend ----------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    if path and os.path.exists(os.path.join(app.static_folder, path)) and not os.path.isdir(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
