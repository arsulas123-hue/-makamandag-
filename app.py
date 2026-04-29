import os, datetime, hashlib, secrets, traceback, sys
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True

DATABASE_URL = "postgresql://makamandag_db_user:zcDibuXdlpEpcZNGEYLc9nqpgWwuTTfO@dpg-d7od7md7vvec739acfj0-a/makamandag_db"
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
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

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

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

# ML Functions
def calculate_health_score(user_id, transactions, budgets):
    try:
        expenses = [t for t in transactions if t.tx_type == 'expense']
        income = sum(t.amount for t in transactions if t.tx_type == 'income')
        total_expense = sum(e.amount for e in expenses)
        score = 70
        if income > 0:
            savings_rate = (income - total_expense) / income
            score += min(20, max(0, savings_rate * 40))
        compliance = 0
        budget_count = 0
        for cat, limit in budgets.items():
            spent = sum(e.amount for e in expenses if e.category == cat)
            if limit > 0:
                budget_count += 1
                if spent <= limit:
                    compliance += 1
                elif spent <= limit * 1.15:
                    compliance += 0.5
        if budget_count > 0:
            score += (compliance / budget_count) * 10
        return max(0, min(100, round(score)))
    except:
        return 70

def forecast_spending(user_id, transactions, weeks=4):
    try:
        return {f'Week {i+1}': 2000 + (i * 100) for i in range(weeks)}
    except:
        return {f'Week {i+1}': 2000 for i in range(weeks)}

def generate_advice(user_id, transactions, budgets):
    expenses = [t for t in transactions if t.tx_type == 'expense']
    cat_spending = {}
    for e in expenses:
        cat_spending[e.category] = cat_spending.get(e.category, 0) + e.amount
    advice = []
    for cat, spent in cat_spending.items():
        limit = budgets.get(cat, 0)
        if limit > 0:
            pct = (spent / limit) * 100
            if pct > 100:
                advice.append({'cat': cat, 'msg': f'exceeded by {round(pct-100)}%'})
            elif pct > 85:
                advice.append({'cat': cat, 'msg': f'approaching limit ({round(pct)}%)'})
            else:
                advice.append({'cat': cat, 'msg': 'on track'})
        else:
            if spent > 5000:
                advice.append({'cat': cat, 'msg': f'high spending - consider budget'})
    return advice

# API Routes
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
        user_count = User.query.count()
        role = 'admin' if user_count == 0 else 'user'
        new_user = User(name=name, email=email, password=hashed, role=role, accepted_terms=True)
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
        user.login_count += 1
        user.last_login = datetime.datetime.utcnow()
        user.last_ip = request.remote_addr
        db.session.commit()
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

@app.route('/api/transactions', methods=['GET'])
@login_required
def get_transactions():
    user_id = request.args.get('user_id', type=int)
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.tx_date.desc()).all()
    return jsonify([{'tx_id': t.id, 'amount': t.amount, 'category': t.category, 'tx_type': t.tx_type, 'note': t.note, 'tx_date': t.tx_date.isoformat()} for t in txs])

@app.route('/api/transactions', methods=['POST'])
@login_required
def add_transaction():
    data = request.json
    tx = Transaction(user_id=session['user_id'], amount=data['amount'], category=data['category'], tx_type=data['tx_type'], note=data.get('note', ''))
    db.session.add(tx)
    db.session.commit()
    return jsonify({'message': 'Saved'}), 201

@app.route('/api/transactions/<int:tx_id>', methods=['DELETE'])
@login_required
def delete_transaction(tx_id):
    tx = Transaction.query.get(tx_id)
    if tx and tx.user_id == session['user_id']:
        db.session.delete(tx)
        db.session.commit()
    return jsonify({'message': 'Deleted'})

@app.route('/api/summary/<int:user_id>')
@login_required
def summary(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).all()
    total_income = sum(t.amount for t in txs if t.tx_type == 'income')
    total_expense = sum(t.amount for t in txs if t.tx_type == 'expense')
    monthly = {}
    for t in txs:
        key = t.tx_date.strftime('%Y-%m')
        if key not in monthly:
            monthly[key] = {'income': 0, 'expense': 0}
        if t.tx_type == 'income':
            monthly[key]['income'] += t.amount
        else:
            monthly[key]['expense'] += t.amount
    return jsonify({'balance': total_income - total_expense, 'income': total_income, 'expense': total_expense, 'monthly': monthly})

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    txs = Transaction.query.filter_by(user_id=user_id).all()
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    score = calculate_health_score(user_id, txs, budgets)
    forecast = forecast_spending(user_id, txs)
    categories = {}
    for t in txs:
        if t.tx_type == 'expense':
            categories[t.category] = categories.get(t.category, 0) + t.amount
    advice = generate_advice(user_id, txs, budgets)
    return jsonify({'score': score, 'predictions': {'weekly': forecast, 'categories': categories}, 'advice': advice})

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
    budget = Budget.query.filter_by(user_id=user_id, category=data['category']).first()
    if budget:
        budget.limit_amount = data['limit']
    else:
        budget = Budget(user_id=user_id, category=data['category'], limit_amount=data['limit'])
        db.session.add(budget)
    db.session.commit()
    return jsonify({'message': 'Budget saved'})

@app.route('/api/admin/stats')
@admin_required
def admin_stats():
    return jsonify({'total_users': User.query.count(), 'total_transactions': Transaction.query.count()})

@app.route('/api/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'name': u.name, 'email': u.email, 'role': u.role} for u in users])

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
