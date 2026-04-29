import os
import datetime
import hashlib
import secrets
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

# PostgreSQL database - Using the connection string from your code
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

# ========== CREATE TABLES ==========
with app.app_context():
    db.create_all()
    # Create default admin user if not exists
    if User.query.filter_by(email='admin@smartspend.com').first() is None:
        hashed = hashlib.sha256('admin123'.encode()).hexdigest()
        admin = User(name='Admin', email='admin@smartspend.com', password=hashed, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created: admin@smartspend.com / admin123")

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

# ========== ML FUNCTIONS ==========
def calculate_health_score(transactions, budgets):
    """Calculate financial health score based on savings rate and budget compliance"""
    try:
        expenses = [t for t in transactions if t['tx_type'] == 'expense']
        income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
        total_expense = sum(e['amount'] for e in expenses)
        score = 70
        if income > 0:
            savings_rate = (income - total_expense) / income
            score += min(20, max(0, savings_rate * 40))
        
        budget_compliance = 0
        budget_count = 0
        for cat, limit in budgets.items():
            spent = sum(e['amount'] for e in expenses if e['category'] == cat)
            if limit > 0:
                budget_count += 1
                if spent <= limit:
                    budget_compliance += 1
                elif spent <= limit * 1.15:
                    budget_compliance += 0.5
        if budget_count > 0:
            score += (budget_compliance / budget_count) * 10
        
        return max(0, min(100, round(score)))
    except Exception as e:
        print(f"Error calculating health score: {e}")
        return 70

def forecast_spending(transactions):
    """AI-powered spending forecast based on user's historical data"""
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    weekly = {'Week 1': 2500, 'Week 2': 2400, 'Week 3': 2600, 'Week 4': 2300}
    
    if len(expenses) > 3:
        # Calculate weighted average with trend
        recent = expenses[-min(len(expenses), 8):]
        avg_recent = sum(e['amount'] for e in recent) / len(recent)
        # Add slight upward/downward trend based on recent changes
        if len(recent) >= 4:
            first_half = sum(e['amount'] for e in recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(e['amount'] for e in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            trend_factor = second_half / first_half if first_half > 0 else 1.0
        else:
            trend_factor = 1.0
        
        weekly = {
            f'Week {i+1}': round(avg_recent * (0.85 + 0.08 * i) * trend_factor)
            for i in range(4)
        }
    return weekly

def generate_advice(transactions, budgets):
    """Generate personalized financial advice including investment recommendations"""
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    income = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    total_expense = sum(e['amount'] for e in expenses)
    savings = income - total_expense
    
    cat_spending = {}
    for e in expenses:
        cat_spending[e['category']] = cat_spending.get(e['category'], 0) + e['amount']
    
    advice = []
    
    # Budget-related advice
    for cat, spent in cat_spending.items():
        limit = budgets.get(cat, 0)
        if limit > 0:
            pct = (spent / limit) * 100
            if pct > 100:
                advice.append({'cat': cat, 'msg': f'exceeded limit by {round(pct-100)}%! Consider reducing spending.'})
            elif pct > 85:
                advice.append({'cat': cat, 'msg': f'nearing limit ({round(pct)}%). Monitor your spending.'})
            else:
                advice.append({'cat': cat, 'msg': f'on track ({round(pct)}% of budget). Good job!'})
        else:
            if spent > 5000:
                advice.append({'cat': cat, 'msg': f'high spending (₱{spent:,.2f}). Consider setting a budget.'})
    
    # Investment advice based on savings
    if savings > 0:
        advice.append({
            'cat': 'Investment',
            'msg': f'You have ₱{savings:,.2f} surplus. Consider putting 30% into low-cost index funds, 20% into high-yield savings, and 50% into skill development.'
        })
    else:
        advice.append({
            'cat': 'Warning',
            'msg': f'Your expenses ({total_expense:,.2f}) exceed income ({income:,.2f}). Review discretionary spending.'
        })
    
    # Stock market intelligence
    advice.append({
        'cat': 'Market Insight',
        'msg': 'Current market conditions suggest dollar-cost averaging into diversified ETFs for long-term growth.'
    })
    
    return advice

# ========== API ROUTES ==========
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
        return jsonify({
            'user': {
                'id': user.id, 
                'name': user.name, 
                'email': user.email, 
                'role': user.role
            }
        })
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
    return jsonify({
        'id': user.id, 
        'name': user.name, 
        'email': user.email, 
        'role': user.role
    })

@app.route('/api/transactions', methods=['GET'])
@login_required
def get_transactions():
    user_id = request.args.get('user_id', type=int)
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
    
    # If user_id not provided, use session user_id
    if not user_id:
        user_id = session['user_id']
        
    txs = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.tx_date.desc()).all()
    return jsonify([{
        'tx_id': t.id, 
        'amount': t.amount, 
        'category': t.category, 
        'tx_type': t.tx_type, 
        'note': t.note or '', 
        'tx_date': t.tx_date.isoformat()
    } for t in txs])

@app.route('/api/transactions', methods=['POST'])
@login_required
def add_transaction():
    try:
        data = request.json
        tx = Transaction(
            user_id=session['user_id'],
            amount=float(data['amount']),
            category=data['category'],
            tx_type=data['tx_type'],
            note=data.get('note', '')
        )
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
    txs_data = [{
        'amount': t.amount, 
        'tx_type': t.tx_type, 
        'tx_date': t.tx_date, 
        'category': t.category
    } for t in txs]
    
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
    
    return jsonify({
        'balance': total_income - total_expense,
        'income': total_income,
        'expense': total_expense,
        'monthly': monthly
    })

@app.route('/api/predict/<int:user_id>')
@login_required
def predict(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
        
    txs = Transaction.query.filter_by(user_id=user_id).all()
    txs_data = [{
        'amount': t.amount, 
        'tx_type': t.tx_type, 
        'category': t.category
    } for t in txs]
    
    budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
    score = calculate_health_score(txs_data, budgets)
    forecast = forecast_spending(txs_data)
    categories = {}
    for t in txs_data:
        if t['tx_type'] == 'expense':
            categories[t['category']] = categories.get(t['category'], 0) + t['amount']
    advice = generate_advice(txs_data, budgets)
    
    return jsonify({
        'score': score,
        'predictions': {'weekly': forecast, 'categories': categories},
        'advice': advice
    })

@app.route('/api/budgets/<int:user_id>', methods=['GET'])
@login_required
def get_budgets(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
        
    budgets = Budget.query.filter_by(user_id=user_id).all()
    return jsonify([{
        'category': b.category, 
        'limit': b.limit_amount
    } for b in budgets])

@app.route('/api/budgets/<int:user_id>', methods=['POST'])
@login_required
def set_budget(user_id):
    if user_id != session['user_id']:
        return jsonify({'error': 'Access denied'}), 403
        
    data = request.json
    budget = Budget.query.filter_by(user_id=user_id, category=data['category']).first()
    if budget:
        budget.limit_amount = float(data['limit'])
    else:
        budget = Budget(user_id=user_id, category=data['category'], limit_amount=float(data['limit']))
        db.session.add(budget)
    db.session.commit()
    return jsonify({'message': 'Budget saved'})

@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    """Chatbot endpoint that provides investment and financial advice"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Get user's financial data
        user_id = session['user_id']
        txs = Transaction.query.filter_by(user_id=user_id).all()
        txs_data = [{
            'amount': t.amount, 
            'tx_type': t.tx_type, 
            'category': t.category
        } for t in txs]
        budgets = {b.category: b.limit_amount for b in Budget.query.filter_by(user_id=user_id).all()}
        
        # Generate AI response based on user message
        response = generate_chatbot_response(user_message, txs_data, budgets)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"Sorry, I encountered an error: {str(e)}"})

def generate_chatbot_response(user_message, transactions, budgets):
    """Generate intelligent chatbot response with market and investment insights"""
    if not transactions:
        return "👋 Welcome! I don't see any transactions yet. Add some transactions so I can analyze your spending and provide personalized investment advice."
    
    msg = user_message.lower()
    
    # Calculate user financial metrics
    expenses = [t for t in transactions if t['tx_type'] == 'expense']
    income_total = sum(t['amount'] for t in transactions if t['tx_type'] == 'income')
    expense_total = sum(e['amount'] for e in expenses)
    balance = income_total - expense_total
    health_score = calculate_health_score(transactions, budgets)
    forecast = forecast_spending(transactions)
    avg_forecast = sum(forecast.values()) / 4 if forecast else 0
    
    # Category breakdown
    cat_spending = {}
    for e in expenses:
        cat_spending[e['category']] = cat_spending.get(e['category'], 0) + e['amount']
    top_category = max(cat_spending.items(), key=lambda x: x[1]) if cat_spending else ("None", 0)
    
    # Investment keywords
    if any(word in msg for word in ['invest', 'stock', 'where to invest', 'crypto', 'etf', 'mutual fund', 'portfolio']):
        surplus = balance
        return f"📈 **AI Investment Strategy**\n\nBased on your ₱{balance:,.2f} surplus and health score {health_score}/100:\n\n" + \
               f"• **Recommended Allocation:**\n" + \
               f"  - 40% to S&P500 ETF (VOO/SPY) - Historical return ~10% annually\n" + \
               f"  - 30% to high-yield savings account (4-5% APY) for emergency fund\n" + \
               f"  - 20% to blue-chip dividend stocks (JNJ, KO, PG)\n" + \
               f"  - 10% to skill development or leisure\n\n" + \
               f"• **Monthly Investment Potential:** ₱{max(0, round(surplus * 0.3)):,.2f}\n" + \
               f"• **Compounding Estimate:** At 7% annual return, ₱{max(0, round(surplus * 0.3)):,.2f}/month grows to ₱{round(max(0, surplus * 0.3) * 12 * 12.5):,.2f} in 10 years."
    
    # Forecast/prediction keywords
    elif any(word in msg for word in ['forecast', 'predict', 'next month', 'spending trend']):
        return f"🤖 **Personalized ML Forecast**\n\nBased on your {len(expenses)} transactions:\n\n" + \
               f"• {', '.join([f'{k}: ₱{v:,.2f}' for k, v in forecast.items()])}\n\n" + \
               f"• **Average Weekly Spend:** ₱{avg_forecast:,.2f}\n" + \
               f"• **Top Category:** {top_category[0]} (₱{top_category[1]:,.2f})\n\n" + \
               f"💡 *Reducing {top_category[0]} by 15% could free up ₱{round(top_category[1] * 0.15):,.2f} monthly for investments.*"
    
    # Chart/trend analysis
    elif any(word in msg for word in ['chart', 'trend', 'category', 'spending pattern']):
        return f"📊 **Real-time Spending Analysis**\n\n" + \
               f"• **Largest Expense Category:** {top_category[0]} (₱{top_category[1]:,.2f})\n" + \
               f"• **Income vs Expense:** {'Surplus' if balance > 0 else 'Deficit'} of ₱{abs(balance):,.2f}\n" + \
               f"• **Health Score:** {health_score}/100 ({'Excellent' if health_score > 80 else 'Good' if health_score > 60 else 'Needs improvement'})\n\n" + \
               f"📈 *Technical Analysis: Your spending shows a {'positive' if income_total > expense_total else 'negative'} trend.*"
    
    # Budget advice
    elif any(word in msg for word in ['advice', 'tip', 'budget', 'save', 'saving']):
        advice_list = generate_advice(transactions, budgets)
        main_tip = advice_list[0] if advice_list else {'msg': 'Set up automatic transfers to savings.'}
        return f"💡 **Smart Financial Tip**\n\n{main_tip.get('msg', '')}\n\n" + \
               f"• **Current Savings Rate:** {max(0, round((balance/income_total)*100 if income_total > 0 else 0))}%\n" + \
               f"• **Recommended Action:** Automate 20% of income to investment accounts before spending."
    
    # Retirement planning
    elif any(word in msg for word in ['retire', 'retirement', 'future']):
        monthly_surplus = max(0, balance / 12)
        years = 20
        future_value = monthly_surplus * 12 * ((1 + 0.07) ** years - 1) / 0.07 if monthly_surplus > 0 else 0
        return f"⏳ **Retirement Projection**\n\nBased on your current finances:\n\n" + \
               f"• **Monthly Surplus:** ₱{monthly_surplus:,.2f}\n" + \
               f"• **20-Year Growth at 7%:** ₱{future_value:,.2f}\n\n" + \
               f"• **Recommendation:** Increase savings rate to 25% for earlier retirement."
    
    # Default response
    else:
        return f"✨ **Cognitive AI Report**\n\n" + \
               f"• **Balance:** ₱{balance:,.2f}\n" + \
               f"• **Health Score:** {health_score}/100\n" + \
               f"• **Weekly Forecast:** ₱{avg_forecast:,.2f}\n" + \
               f"• **Top Category:** {top_category[0]} (₱{top_category[1]:,.2f})\n\n" + \
               f"💬 **Try asking:**\n" + \
               f"  • 'Where to invest my money?'\n" + \
               f"  • 'Show me spending forecast'\n" + \
               f"  • 'Budget advice'\n" + \
               f"  • 'Retirement planning'"

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
