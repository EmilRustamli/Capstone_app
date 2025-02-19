from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import secrets
import random
from werkzeug.security import generate_password_hash, check_password_hash
import os

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
app.config['MAIL_PASSWORD'] = 'ganh bjho orkp tkkw'  # You'll need to generate this from Google Account

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
    user = User.query.filter_by(email=session['user_email']).first()
    
    new_item = PortfolioItem(
        user_id=user.id,
        ticker=data['ticker'].upper(),
        amount=float(data['amount'])
    )
    
    db.session.add(new_item)
    db.session.commit()
    
    return jsonify({
        'message': 'Portfolio item added',
        'item': {
            'id': new_item.id,
            'ticker': new_item.ticker,
            'amount': new_item.amount
        }
    })

if __name__ == '__main__':
    # Create the instance folder if it doesn't exist
    if not os.path.exists('instance'):
        os.makedirs('instance')
    
    # Only create tables if they don't exist
    with app.app_context():
        db.create_all()
    
    app.run(debug=True) 