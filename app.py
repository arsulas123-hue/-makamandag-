import os, datetime, hashlib, secrets, traceback, sys
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

# Try to import ML libraries, but gracefully degrade if missing
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("WARNING: sklearn/numpy not installed. ML features disabled.", file=sys.stderr)

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Use environment variable for database URL (fallback to provided)
DATABASE_URL = os.environ.get('DATABASE_URL', "postgresql://makamandag_db_user:zcDibuXdlpEpcZNGEYLc9nqpgWwuTTfO@dpg-d7od7md7vvec739acfj0-a/makamandag_db")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------- Models (all) ----------
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

# ---------- Create tables (with error handling) ----------
def init_db():
    try:
        db.create_all()
        print("Database tables created/verified.", file=sys.stderr)
        # Create default admin if none exists
        if User.query.filter_by(role='admin').first() is None:
            hashed = hashlib.sha256('admin123'.encode()).hexdigest()
            admin = User(name='Admin', email='admin@smartspend.com', password=hashed, role='admin', accepted_terms=True)
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created.", file=sys.stderr)
    except Exception as e:
        print(f"Database init error: {e}", file=sys.stderr)

with app.app_context():
    init_db()

# ---------- Helpers ----------
def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def log_audit(user_id, action, ip, detail=''):
    try:
        log = AuditLog(user_id=user_id, action=action, ip=ip, detail=detail)
        db.session.add(log)
        db.session.commit()
    except:
        pass

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

# ---------- ML Functions (safe fallback if ML not available) ----------
def forecast_spending(user_id, transactions, weeks=4):
    if not ML_AVAILABLE:
        return {f'Week {i+1}': 2000 for i in range(weeks)}
    try:
        # Get past errors
        past = ForecastLog.query.filter_by(user_id=user_id).filter(ForecastLog.actual_amount != None).all()
        avg_error = np.mean([f.error for f in past]) if past else 0
        expenses = [t for t in transactions if t.tx_type == 'expense']
        if len(expenses) < 3:
            avg = np.mean([t.amount for t in expenses]) if expenses else 2000
            base = {f'Week {i+1}': round(avg * (0.9 + 0.2 * np.random.random())) for i in range(weeks)}
        else:
            # ... (existing logic, omitted for brevity but keep your full version)
            # For brevity, returning dummy; replace with your full function.
            base = {f'Week {i+1}': 2000 for i in range(weeks)}
        adjusted = {w: max(0, round(v + avg_error)) for w, v in base.items()}
        # Store forecast
        today = datetime.date.today()
        for i, (w, amt) in enumerate(adjusted.items()):
            week_start = today + datetime.timedelta(days=7*i)
            if not ForecastLog.query.filter_by(user_id=user_id, week_start=week_start).first():
                db.session.add(ForecastLog(user_id=user_id, week_start=week_start, predicted_amount=amt))
        db.session.commit()
        return adjusted
    except Exception as e:
        print(f"Forecast error: {e}", file=sys.stderr)
        return {f'Week {i+1}': 2000 for i in range(weeks)}

def calculate_health_score(user_id, transactions, budgets):
    if not ML_AVAILABLE:
        return 70
    try:
        # Your existing logic
        return 75
    except:
        return 70

def generate_advice(user_id, transactions, budgets):
    advice = []
    try:
        # Your existing logic
        pass
    except:
        pass
    return advice

# ---------- API ROUTES (with full error handling) ----------
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'ml_available': ML_AVAILABLE})

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
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
        new_user = User(name=name, email=email, password=hashed, role=role, accepted_terms=True, terms_version='1.0')
        db.session.add(new_user)
        db.session.commit()
        log_audit(new_user.id, 'register', request.remote_addr, f'User {email} registered')
        return jsonify({'message': 'User created'}), 201
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        user = User.query.filter_by(email=email).first()
        if not user or user.password != hash_password(password):
            log_audit(None, 'failed_login', request.remote_addr, f'Failed login for {email}')
            return jsonify({'error': 'Invalid credentials'}), 401
        if not user.is_active:
            return jsonify({'error': 'Account disabled'}), 401

        session['user_id'] = user.id
        user.login_count += 1
        user.last_login = datetime.datetime.utcnow()
        user.last_ip = request.remote_addr
        db.session.commit()
        log_audit(user.id, 'login', request.remote_addr, 'Successful login')
        return jsonify({
            'user': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'role': user.role,
                'is_active': user.is_active
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    if 'user_id' in session:
        log_audit(session['user_id'], 'logout', request.remote_addr, 'User logged out')
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
    return jsonify({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role,
        'is_active': user.is_active
    })

# Add other routes similarly (transactions, summary, predict, budgets, etc.)
# For brevity, I'll include the minimal set to get registration/login working.
# You can copy your full working routes from your previous version.

# IMPORTANT: Add a catch-all for missing routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
