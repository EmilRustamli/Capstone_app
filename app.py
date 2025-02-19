from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import secrets
import random
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from stock_api import update_stock_prices

app = Flask(__name__)
mail = Mail(app)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Mail Settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'rustamliemiluni@gmail.com'
app.config['MAIL_PASSWORD'] = 'ganh bjho orkp tkkw'  # From personal google account

# Add this near the top with other configurations
app.config['SECRET_KEY'] = secrets.token_hex(16)

db = SQLAlchemy(app)

# Initialize mail after all configurations
mail.init_app(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_confirmed = db.Column(db.Boolean, default=False)
    confirmation_code = db.Column(db.String(6), unique=True)

class PortfolioItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ticker = db.Column(db.String(10), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    
    user = db.relationship('User', backref=db.backref('portfolio_items', lazy=True))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    
    # Check if user exists and is already confirmed
    existing_user = User.query.filter_by(email=data['email']).first()
    if existing_user and existing_user.is_confirmed:
        return jsonify({'error': 'Email already registered'}), 400
    
    # Delete any unconfirmed registration for this email
    if existing_user and not existing_user.is_confirmed:
        db.session.delete(existing_user)
        db.session.commit()
    
    # Check if username is taken by a confirmed user
    existing_username = User.query.filter_by(username=data['username']).first()
    if existing_username and existing_username.is_confirmed:
        return jsonify({'error': 'Username already taken'}), 400

    # Create new user with confirmation code
    confirmation_code = ''.join(random.choices('0123456789', k=6))
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    
    new_user = User(
        username=data['username'],
        email=data['email'],
        password=hashed_password,
        confirmation_code=confirmation_code
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    # Send confirmation email with code
    msg = Message('Your Confirmation Code',
                  sender='rustamliemiluni@gmail.com',
                  recipients=[data['email']])
    msg.body = f'Your confirmation code is: {confirmation_code}'
    mail.send(msg)
    
    return jsonify({'message': 'Registration successful! Please check your email for the confirmation code.'})

@app.route('/verify-code', methods=['POST'])
def verify_code():
    data = request.json
    user = User.query.filter_by(email=data['email'], confirmation_code=data['code']).first()
    
    if user:
        user.is_confirmed = True
        user.confirmation_code = None
        db.session.commit()
        return jsonify({'message': 'Email confirmed! You can now login.'})
    return jsonify({'error': 'Invalid confirmation code'}), 400

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if 'user_email' not in session:
        return redirect(url_for('home'))
    
    # Get user info
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        session.pop('user_email', None)
        return redirect(url_for('home'))
        
    return render_template('dashboard.html', username=user.username)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    
    if user and check_password_hash(user.password, data['password']):
        if not user.is_confirmed:
            return jsonify({'error': 'Please confirm your email first'}), 401
        session['user_email'] = user.email
        return jsonify({'message': 'Login successful!', 'redirect': '/dashboard'})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('home'))

@app.route('/account')
def account():
    if 'user_email' not in session:
        return redirect(url_for('home'))
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        session.pop('user_email', None)
        return redirect(url_for('home'))
        
    return render_template('account.html', user=user)

@app.route('/update-password', methods=['POST'])
def update_password():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    user = User.query.filter_by(email=session['user_email']).first()
    
    if user and check_password_hash(user.password, data['current_password']):
        user.password = generate_password_hash(data['new_password'], method='pbkdf2:sha256')
        db.session.commit()
        return jsonify({'message': 'Password updated successfully!'})
    
    return jsonify({'error': 'Current password is incorrect'}), 400

@app.route('/portfolio')
def portfolio():
    if 'user_email' not in session:
        return redirect(url_for('home'))
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        session.pop('user_email', None)
        return redirect(url_for('home'))
    
    portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
    return render_template('portfolio.html', username=user.username, portfolio_items=portfolio_items)

@app.route('/add-portfolio-item', methods=['POST'])
def add_portfolio_item():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    ticker = data['ticker'].upper()
    try:
        amount = float(data['amount'])
        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid amount'}), 400
    
    # Verify stock exists in company_info.json
    stock_info = get_stock_info(ticker)
    if not stock_info:
        return jsonify({'error': 'Invalid stock ticker'}), 400
    
    user = User.query.filter_by(email=session['user_email']).first()
    
    # Check if user already has this stock
    existing_item = PortfolioItem.query.filter_by(
        user_id=user.id,
        ticker=ticker
    ).first()
    
    if existing_item:
        # Update existing item's amount
        existing_item.amount += amount
        db.session.commit()
        
        return jsonify({
            'message': 'Portfolio item updated',
            'item': {
                'id': existing_item.id,
                'ticker': existing_item.ticker,
                'amount': existing_item.amount,
                'name': stock_info['name']
            }
        })
    else:
        # Create new item
        new_item = PortfolioItem(
            user_id=user.id,
            ticker=ticker,
            amount=amount
        )
        
        db.session.add(new_item)
        db.session.commit()
        
        return jsonify({
            'message': 'Portfolio item added',
            'item': {
                'id': new_item.id,
                'ticker': new_item.ticker,
                'amount': new_item.amount,
                'name': stock_info['name']
            }
        })

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    
    if user:
        # Generate confirmation code
        reset_code = ''.join(random.choices('0123456789', k=6))
        user.confirmation_code = reset_code
        db.session.commit()
        
        # Send email with code
        msg = Message('Password Reset Code',
                     sender='rustamliemiluni@gmail.com',
                     recipients=[user.email])
        msg.body = f'Your password reset code is: {reset_code}'
        mail.send(msg)
        
        return jsonify({'message': 'Password reset code sent to your email.'})
    
    return jsonify({'error': 'Email not found'}), 404

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    user = User.query.filter_by(email=data['email'], confirmation_code=data['code']).first()
    
    if user:
        user.password = generate_password_hash(data['new_password'], method='pbkdf2:sha256')
        user.confirmation_code = None
        db.session.commit()
        return jsonify({'message': 'Password reset successful! You can now login.'})
    
    return jsonify({'error': 'Invalid code'}), 400

@app.route('/search-stocks')
def search_stocks_route():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    
    results = search_stocks(query)
    return jsonify(results)

@app.route('/get-companies')
def get_companies():
    try:
        with open('company_info.json', 'r') as f:
            companies = json.load(f)
        return jsonify(companies)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this function to load company info
def get_stock_info(ticker):
    try:
        with open('company_info.json', 'r') as f:
            company_info = json.load(f)
        return company_info.get(ticker.upper())
    except:
        return None

def search_stocks(query):
    try:
        with open('company_info.json', 'r') as f:
            company_info = json.load(f)
        
        query = query.upper()
        results = []
        for ticker, info in company_info.items():
            if query in ticker.upper() or query in info['name'].upper():
                results.append({
                    'ticker': ticker,
                    'name': info['name']
                })
        return results[:10]  # Return top 10 matches
    except:
        return []

@app.route('/reset-portfolio', methods=['POST'])
def reset_portfolio():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        user = User.query.filter_by(email=session['user_email']).first()
        # Delete all portfolio items for this user
        PortfolioItem.query.filter_by(user_id=user.id).delete()
        db.session.commit()
        return jsonify({'message': 'Portfolio reset successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to reset portfolio'}), 500

@app.route('/edit-portfolio-item', methods=['POST'])
def edit_portfolio_item():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    try:
        amount = float(data['amount'])
        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid amount'}), 400

    user = User.query.filter_by(email=session['user_email']).first()
    
    try:
        item = PortfolioItem.query.filter_by(
            id=data['id'],
            user_id=user.id
        ).first()
        
        if not item:
            return jsonify({'error': 'Item not found'}), 404
            
        item.amount = amount
        db.session.commit()
        
        return jsonify({
            'message': 'Portfolio item updated',
            'amount': item.amount
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/delete-portfolio-item', methods=['POST'])
def delete_portfolio_item():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    user = User.query.filter_by(email=session['user_email']).first()
    
    try:
        item = PortfolioItem.query.filter_by(
            id=data['id'],
            user_id=user.id
        ).first()
        
        if not item:
            return jsonify({'error': 'Item not found'}), 404
            
        db.session.delete(item)
        db.session.commit()
        
        return jsonify({'message': 'Portfolio item deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/get-top-stocks')
def get_top_stocks():
    try:
        with open('company_info.json', 'r') as f:
            companies = json.load(f)
        
        # Sort companies by market cap and get top 12
        top_companies = sorted(
            [{'ticker': k, **v} for k, v in companies.items()],
            key=lambda x: float(x.get('marketCap', 0)),
            reverse=True
        )[:12]
        
        return jsonify(top_companies)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create the instance folder if it doesn't exist
    if not os.path.exists('instance'):
        os.makedirs('instance')
    
    # Only create tables if they don't exist
    with app.app_context():
        db.create_all()
    
    app.run(debug=True) 