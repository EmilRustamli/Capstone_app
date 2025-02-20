from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import secrets
import random
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from stock_downloader import TOP_TICKERS  # Only import TOP_TICKERS
import yfinance as yf
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm
from flask_session import Session  # Add this to requirements.txt
import time
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Mail Settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'rustamliemiluni@gmail.com'
app.config['MAIL_PASSWORD'] = 'ganh bjho orkp tkkw'  # From personal google account

# Session and Security configurations
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)
Session(app)  # Initialize Flask-Session here

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

class Portfolio:
    # Class-level attributes to store data
    stock_data = pd.DataFrame()

    @classmethod
    def load_data(cls, csv_path):
        """Load pre-downloaded stock data from a CSV file."""
        cls.stock_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    def __init__(self, tickers, weights, start_date, end_date):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()
        self.returns = self.data.pct_change().dropna()
        self.benchmark_returns = self._fetch_benchmark()

    def _fetch_data(self):
        """Filter the preloaded data for the selected tickers and date range."""
        if Portfolio.stock_data.empty:
            raise ValueError("Stock data is not loaded. Use Portfolio.load_data(csv_path) first")

        filtered_data = Portfolio.stock_data.loc[self.start_date:self.end_date, self.tickers]
        if filtered_data.isnull().values.any():
            raise ValueError("Missing data for selected tickers in the specified date range.")
        return filtered_data

    def _fetch_benchmark(self):
        """Simulate a benchmark by averaging returns of all loaded stocks."""
        benchmark = Portfolio.stock_data.mean(axis=1).loc[self.start_date:self.end_date]
        return benchmark.pct_change().dropna().values.reshape(-1, 1)

    def get_risk_metrics(self):
        return {
            'Annualized Return': self.calculate_annualized_return() * 100,
            'Annualized Volatility': self.calculate_annualized_volatility() * 100,
            'R²': self.calculate_r_squared() * 100,
            'Idiosyncratic Risk': self.calculate_idiosyncratic_risk() * 100,
            'VaR': self.calculate_var() * 100,
            'CVaR': self.calculate_cvar() * 100
        }

    def calculate_portfolio_return_series(self):
        return self.returns.dot(self.weights)

    def calculate_annualized_return(self):
        daily_return = np.dot(self.weights, self.returns.mean())
        return daily_return * 252

    def calculate_annualized_volatility(self):
        covariance_matrix = self.returns.cov() * 252
        portfolio_variance = np.dot(self.weights.T, np.dot(covariance_matrix, self.weights))
        return np.sqrt(portfolio_variance)

    def calculate_r_squared(self):
        port_ret_series = self.calculate_portfolio_return_series()
        lin_reg = LinearRegression().fit(self.benchmark_returns, port_ret_series.values.reshape(-1, 1))
        return lin_reg.score(self.benchmark_returns, port_ret_series.values.reshape(-1, 1))

    def calculate_idiosyncratic_risk(self):
        port_ret_series = self.calculate_portfolio_return_series()
        lin_reg = LinearRegression().fit(self.benchmark_returns, port_ret_series.values.reshape(-1, 1))
        residuals = port_ret_series.values - (self.benchmark_returns.flatten() * lin_reg.coef_[0][0] + lin_reg.intercept_[0])
        return np.std(residuals)

    def calculate_var(self, confidence_level=0.95):
        portfolio_mean = self.calculate_annualized_return() / 252
        portfolio_std = self.calculate_annualized_volatility() / np.sqrt(252)
        z_score = norm.ppf(1 - confidence_level)
        return -(portfolio_mean + z_score * portfolio_std) * np.sqrt(252)

    def calculate_cvar(self, confidence_level=0.95):
        portfolio_mean = self.calculate_annualized_return() / 252
        portfolio_std = self.calculate_annualized_volatility() / np.sqrt(252)
        z_score = norm.ppf(1 - confidence_level)
        conditional_var = portfolio_mean + portfolio_std * (norm.pdf(z_score) / (1 - confidence_level))
        return -conditional_var * np.sqrt(252)

    def explain_risk_metrics(self):
        metrics = self.get_risk_metrics()
        return {
            'Annualized Return': f"Expected yearly return based on historical data.",
            'Annualized Volatility': f"Expected risk or variability in portfolio returns over a year.",
            'R²': f"Proportion of portfolio variance due to market movements.",
            'Idiosyncratic Risk': f"Unique portfolio risk not explained by the market.",
            'VaR': f"Maximum potential loss over a specified period with a 95% confidence.",
            'CVaR': f"Average loss expected in the worst 5% of scenarios."
        }

# Load the data when app starts
try:
    Portfolio.load_data('stock_data.csv')
    print("Successfully loaded stock data")
except Exception as e:
    print(f"Error loading stock data: {str(e)}")

@app.route('/')
def home():
    if 'user_email' in session:
        return redirect(url_for('dashboard'))
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
        
        # Debug: Print all TOP_TICKERS and which ones are missing
        logger.info(f"Looking for these tickers: {TOP_TICKERS}")
        missing_tickers = [t for t in TOP_TICKERS if t not in companies]
        if missing_tickers:
            logger.warning(f"Missing tickers in company_info.json: {missing_tickers}")
        
        # Get info for our predefined TOP_TICKERS in the exact order
        top_companies = []
        for ticker in TOP_TICKERS:
            if ticker in companies:
                company_data = companies[ticker].copy()
                company_data['ticker'] = ticker
                top_companies.append(company_data)
                logger.info(f"Added {ticker} to top companies")
            else:
                logger.warning(f"Ticker {ticker} not found in company_info.json")
        
        logger.info(f"Returning {len(top_companies)} top companies out of {len(TOP_TICKERS)} total")
        return jsonify(top_companies)
    except Exception as e:
        logger.error(f"Error in get_top_stocks: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-portfolio-stocks')
def get_portfolio_stocks():
    if 'user_email' not in session:
        return jsonify([])
    
    try:
        user = User.query.filter_by(email=session['user_email']).first()
        portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
        
        # Get stock info from company_info.json
        with open('company_info.json', 'r') as f:
            company_info = json.load(f)
        
        portfolio_stocks = []
        for item in portfolio_items:
            if item.ticker in company_info:
                stock_info = company_info[item.ticker].copy()
                stock_info['amount'] = item.amount  # Add amount owned
                portfolio_stocks.append(stock_info)
        
        return jsonify(portfolio_stocks)
    except Exception as e:
        print(f"Error getting portfolio stocks: {str(e)}")
        return jsonify([])

# Add this helper function to get companies data
def get_all_companies():
    try:
        with open('company_info.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading companies: {str(e)}")
        return {}

@app.route('/evaluate-portfolio', methods=['POST'])
def evaluate_portfolio():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        # Convert dates to YYYY-MM-DD format
        try:
            start_date = pd.to_datetime(data['start_date']).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(data['end_date']).strftime('%Y-%m-%-d')
            print(f"Evaluating portfolio from {start_date} to {end_date}")
        except Exception as e:
            print(f"Date error: {str(e)}")
            return jsonify({'error': 'Please use YYYY-MM-DD date format'}), 400

        # Get user's portfolio
        user = User.query.filter_by(email=session['user_email']).first()
        portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
        
        if not portfolio_items:
            return jsonify({'error': 'Portfolio is empty'}), 400

        # Create portfolio dictionary treating amounts as share numbers
        portfolio_input = {}
        for item in portfolio_items:
            # Use amount as share count (ignoring $ value)
            portfolio_input[item.ticker] = int(item.amount)

        print(f"Portfolio to evaluate: {portfolio_input}")

        # Calculate weights based on share counts
        tickers = list(portfolio_input.keys())
        shares = list(portfolio_input.values())
        total_shares = sum(shares)
        weights = [share / total_shares for share in shares]

        try:
            portfolio = Portfolio(tickers, weights, start_date, end_date)
            risk_metrics = portfolio.get_risk_metrics()
            explanations = portfolio.explain_risk_metrics()
            
            print(f"Evaluation successful. Metrics: {risk_metrics}")
            
            return jsonify({
                'metrics': risk_metrics,
                'explanations': explanations
            })
            
        except ValueError as e:
            print(f"Portfolio evaluation error: {str(e)}")
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return jsonify({'error': 'Failed to evaluate portfolio'}), 500

@app.errorhandler(403)
def forbidden_error(error):
    session.clear()  # Clear any corrupted session
    return redirect(url_for('home'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()  # Rollback any failed database transactions
    return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    # Create necessary directories
    if not os.path.exists('instance'):
        os.makedirs('instance')
    if not os.path.exists('flask_session'):
        os.makedirs('flask_session')
    
    # Initialize database
    with app.app_context():
        db.create_all()
    
    app.run(debug=True, port=5001) 