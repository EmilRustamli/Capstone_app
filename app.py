from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import secrets
import random
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import yfinance as yf
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import norm
from flask_session import Session  # Add this to requirements.txt
import time
from sklearn.linear_model import LinearRegression
import schedule
import threading
import pytz
from pypfopt import EfficientFrontier, risk_models, expected_returns

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

# Define TOP_TICKERS here since we're not using stock_downloader anymore
TOP_TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 
               'TSLA', 'AVGO', 'BRK-B', 'TSM', 'WMT', '2222.SR']


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
    trade_data = pd.DataFrame()

    @classmethod
    def load_data(cls, csv_path):
        """Load pre-downloaded stock data from a CSV file."""
        try:
            if os.environ.get('VERCEL_ENV') == 'production':
                print("Running in production environment - using lightweight data source")
                # In production, create a minimal dataset with the most important stocks
                minimal_data = {}
                
                # Use yfinance to fetch data for a limited set of stocks (TOP_TICKERS)
                for ticker in TOP_TICKERS:
                    try:
                        data = yf.download(ticker, period="5y")['Adj Close']
                        minimal_data[ticker] = data
                    except Exception as e:
                        print(f"Error fetching data for {ticker}: {str(e)}")
                
                if minimal_data:
                    # Create a DataFrame with the downloaded data
                    cls.trade_data = pd.DataFrame(minimal_data)
                    print(f"Loaded data for {len(minimal_data)} tickers in production")
                else:
                    raise ValueError("Failed to load any stock data in production")
            else:
                # In development, use the local CSV
                cls.trade_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                print(f"Loaded data from CSV in development")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Create minimal fallback data if loading fails
            fallback_data = {}
            for ticker in TOP_TICKERS[:5]:  # Use only 5 tickers for fallback
                try:
                    data = yf.download(ticker, period="2y")['Adj Close']
                    fallback_data[ticker] = data
                except:
                    continue
            
            if fallback_data:
                cls.trade_data = pd.DataFrame(fallback_data)
                print(f"Using fallback data with {len(fallback_data)} tickers")
            else:
                # Last resort - create empty DataFrame with minimal structure
                cls.trade_data = pd.DataFrame()
                print("Using empty DataFrame as last resort")

    @classmethod
    def monte_carlo_simulation(cls, portfolio, num_simulations=100, confidence_interval=0.95, 
                              start_date=None, end_date=None, prediction_days=90):
        """
        Perform Monte Carlo simulation for the portfolio.
        
        Args:
            portfolio: Dict of {ticker: investment_amount}
            num_simulations: Number of simulation paths to generate
            confidence_interval: Confidence interval for the prediction bounds
            start_date: Start date for historical data
            end_date: End date for historical data
            prediction_days: Number of days to predict into the future
            
        Returns:
            dict: Monte Carlo simulation results
        """
        try:
            # Convert start_date and end_date to datetime objects
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            #start_date_dt = end_date_dt - pd.Timedelta(days=365)
            
            # Filter the data for the specified date range
            tickers = list(portfolio.keys())
            filtered_data = cls.trade_data.loc[start_date_dt:end_date_dt, tickers]
            
            # Calculate daily returns
            returns = filtered_data.pct_change().dropna()
            
            # Calculate the covariance matrix
            cov_matrix = returns.cov()
            
            # Create a multivariate normal distribution
            mean_returns = returns.mean().values
            
            # Get the last prices for each ticker
            last_prices = filtered_data.iloc[-1]
            
            # Calculate initial portfolio value and shares
            total_investment = sum(portfolio.values())
            shares = {ticker: portfolio[ticker] / last_prices[ticker] for ticker in tickers}
            
            # Generate future dates for the prediction period
            last_date = filtered_data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='B')
            
            # Combine historical and future dates
            all_dates = pd.DatetimeIndex(filtered_data.index.tolist() + future_dates.tolist())
            
            # Separate historical data index for resampling later
            historical_dates = filtered_data.index
            
            # Create an array to store all simulations
            all_simulations = np.zeros((num_simulations, len(all_dates)))
            
            # Fill in the historical data for all simulations
            historical_values = sum(filtered_data[ticker] * shares[ticker] for ticker in tickers)
            for i in range(num_simulations):
                all_simulations[i, :len(historical_dates)] = historical_values.values
            
            # Simulate future prices using Geometric Brownian Motion
            for i in range(num_simulations):
                # Generate correlated random returns
                random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, prediction_days)
                
                # Simulate future prices
                future_prices = np.zeros((prediction_days, len(tickers)))
                for j in range(len(tickers)):
                    future_prices[0, j] = last_prices[tickers[j]]
                    for k in range(1, prediction_days):
                        future_prices[k, j] = future_prices[k-1, j] * (1 + random_returns[k-1, j])
                
                # Calculate portfolio values
                for k in range(prediction_days):
                    all_simulations[i, len(historical_dates) + k] = sum(future_prices[k, j] * shares[tickers[j]] 
                                                                     for j in range(len(tickers)))
            
            # Resample all dates to weekly - selecting the last value of each week
            all_dates_df = pd.DataFrame(index=all_dates)
            weekly_dates = all_dates_df.resample('W').last().index
            
            # Create a dataframe to store the weekly resampled simulations
            weekly_simulations = pd.DataFrame(index=all_dates, data={f'sim_{i}': all_simulations[i] for i in range(num_simulations)})
            weekly_simulations = weekly_simulations.resample('W').last()
            
            # Convert simulations back to numpy array
            resampled_sims = np.array([weekly_simulations[f'sim_{i}'].values for i in range(num_simulations)])
            
            # Calculate statistics from the simulations at each time point
            median_path = np.median(resampled_sims, axis=0)
            mean_path = np.mean(resampled_sims, axis=0)
            
            # Calculate confidence intervals
            lower_bound = np.percentile(resampled_sims, (1 - confidence_interval) * 100 / 2, axis=0)
            upper_bound = np.percentile(resampled_sims, 100 - (1 - confidence_interval) * 100 / 2, axis=0)
            
            # Calculate the historical path (resampled to weekly)
            historical_weekly_values = historical_values.resample('W').last()
            
            # Calculate final value statistics
            final_values = resampled_sims[:, -1]
            final_mean_value = np.mean(final_values)
            final_median_value = np.median(final_values)
            final_lower_bound = np.percentile(final_values, (1 - confidence_interval) * 100 / 2)
            final_upper_bound = np.percentile(final_values, 100 - (1 - confidence_interval) * 100 / 2)
            
            # Calculate VaR
            initial_value = historical_values.iloc[-1]
            changes = final_values - initial_value
            var_95 = np.percentile(changes, 5)
            
            # Probability of gain
            probability_of_gain = np.mean(final_values > initial_value)
            
            # Max potential gain
            max_gain = np.max(final_values) - initial_value
            
            # Prepare results
            return {
                'dates': weekly_dates.strftime('%Y-%m-%d').tolist(),
                'median_path': median_path.tolist(),
                'mean_path': mean_path.tolist(),
                'upper_bound': upper_bound.tolist(),
                'lower_bound': lower_bound.tolist(),
                'historical_path': historical_weekly_values.values.tolist(),
                'sample_paths': resampled_sims[:20].tolist(),  # Include only 20 sample paths to reduce payload size
                'final_mean_value': float(final_mean_value),
                'final_median_value': float(final_median_value),
                'final_lower_bound': float(final_lower_bound),
                'final_upper_bound': float(final_upper_bound),
                'var_95': float(var_95),
                'probability_of_gain': float(probability_of_gain),
                'max_gain': float(max_gain)
            }
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {str(e)}")
            raise

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
        if Portfolio.trade_data.empty:
            raise ValueError("Stock data is not loaded. Use Portfolio.load_data(csv_path) first")

        # Convert string dates to datetime objects if they're not already
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        # Find nearest previous valid dates if the exact dates don't exist
        valid_dates = Portfolio.trade_data.index
        
        if start_date not in valid_dates:
            # Find the nearest previous date that exists in the dataset
            valid_previous_dates = valid_dates[valid_dates < start_date]
            if len(valid_previous_dates) == 0:
                raise ValueError(f"No valid dates before start_date {start_date}")
            start_date = valid_previous_dates[-1]
            print(f"Adjusted start_date to nearest previous valid date: {start_date}")
        
        if end_date not in valid_dates:
            # Find the nearest previous date that exists in the dataset
            valid_previous_dates = valid_dates[valid_dates < end_date]
            if len(valid_previous_dates) == 0:
                raise ValueError(f"No valid dates before end_date {end_date}")
            end_date = valid_previous_dates[-1]
            print(f"Adjusted end_date to nearest previous valid date: {end_date}")

        filtered_data = Portfolio.trade_data.loc[start_date:end_date, self.tickers]
        if filtered_data.isnull().values.any():
            raise ValueError("Missing data for selected tickers in the specified date range.")
        return filtered_data

    def _fetch_benchmark(self):
        """Simulate a benchmark by averaging returns of all loaded stocks."""
        benchmark = Portfolio.trade_data.mean(axis=1).loc[self.start_date:self.end_date]
        return benchmark.pct_change().dropna().values.reshape(-1, 1)

    def get_risk_metrics(self):
        return {
            'Annualized Return': self.calculate_annualized_return() * 100,
            'Annualized Volatility': self.calculate_annualized_volatility() * 100,
            'Sharpe Ratio': self.calculate_annualized_return() / self.calculate_annualized_volatility(),
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

    @classmethod
    def optimize_user_portfolio(cls, portfolio_input, start_date, end_date, optimize_for="sharpe", build_new=False, full_tickers_list=None):
        try:
            # Get historical data
            if build_new:
                # Filter tickers that exist in our trade_data
                available_tickers = [ticker for ticker in full_tickers_list 
                                   if ticker in cls.trade_data.columns]
                tickers = available_tickers
            else:
                tickers = list(portfolio_input.keys())

            # Filter data for the specified date range
            data = cls.trade_data.loc[start_date:end_date, tickers]
            
            # Calculate current portfolio weights - always use actual portfolio metrics
            total_value = sum(portfolio_input.values())
            current_portfolio_tickers = list(portfolio_input.keys())
            current_portfolio_weights = {ticker: value/total_value for ticker, value in portfolio_input.items()}
            
            # Get data for current portfolio evaluation
            current_portfolio_data = cls.trade_data.loc[start_date:end_date, current_portfolio_tickers]
            current_returns = current_portfolio_data.pct_change()
            current_mean_returns = current_returns.mean() * 252
            current_cov_matrix = current_returns.cov() * 252
            
            # Calculate current portfolio metrics - this will always be the same regardless of build_new
            current_return = sum(current_portfolio_weights[ticker] * current_mean_returns[ticker] 
                             for ticker in current_portfolio_weights)
            current_volatility = np.sqrt(
                sum(sum(current_portfolio_weights[ticker1] * current_portfolio_weights[ticker2] * current_cov_matrix.loc[ticker1, ticker2]
                    for ticker2 in current_portfolio_weights)
                    for ticker1 in current_portfolio_weights)
            )
            
            # For optimization, use different weights based on build_new
            if build_new:
                # For new portfolio, start with equal weights for optimization
                num_stocks = len(tickers)
                optimization_weights = {ticker: 1.0/num_stocks for ticker in tickers}
            else:
                # Use current weights for optimization
                optimization_weights = current_portfolio_weights
            
            # Optimize portfolio
            optimized_weights = cls.optimize_with_pypfopt(
                data=data,
                original_weights=optimization_weights,
                original_return=current_return,
                original_volatility=current_volatility,
                optimize_for=optimize_for
            )

            # Calculate metrics for optimized portfolio
            returns = data.pct_change()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            opt_return = sum(optimized_weights[ticker] * mean_returns[ticker] 
                           for ticker in optimized_weights)
            opt_volatility = np.sqrt(
                sum(sum(optimized_weights[ticker1] * optimized_weights[ticker2] * cov_matrix.loc[ticker1, ticker2]
                    for ticker2 in optimized_weights)
                    for ticker1 in optimized_weights)
            )
            opt_sharpe = opt_return / opt_volatility

            # Calculate dollar allocations
            optimized_allocation = {
                ticker: weight * total_value 
                for ticker, weight in optimized_weights.items()
                if weight > 0.01  # Filter out very small allocations
            }

            return {
                'metrics': {
                    'Current Return': current_return * 100,
                    'Current Volatility': current_volatility * 100,
                    'Current Sharpe': (current_return / current_volatility),
                    'Optimized Return': opt_return * 100,
                    'Optimized Volatility': opt_volatility * 100,
                    'Optimized Sharpe': opt_sharpe
                },
                'allocation': optimized_allocation
            }

        except Exception as e:
            print(f"Error in optimize_user_portfolio: {str(e)}")
            raise

    @classmethod
    def optimize_with_pypfopt(cls, data, original_weights, original_return, original_volatility, optimize_for="sharpe"):
        try:
            # Calculate expected returns and sample covariance
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            # Check if any assets have expected returns exceeding risk-free rate
            # PyPortfolioOpt assumes a risk-free rate of 0.02 (2%) by default
            risk_free_rate = 0.02
            if not any(mu > risk_free_rate):
                print("Warning: No assets have expected returns exceeding the risk-free rate. Returning original weights.")
                # Return the original weights since optimization would fail
                return original_weights

            try:
                # Try OSQP first (faster)
                ef = EfficientFrontier(mu, S, solver="OSQP")
                
                # Optimize based on the specified target
                if optimize_for == "sharpe":
                    ef.max_sharpe()
                elif optimize_for == "return":
                    target_volatility = original_volatility * 1.2
                    ef.efficient_risk(target_volatility)
                elif optimize_for == "volatility":
                    ef.min_volatility()
                    
                weights = ef.clean_weights()
                
            except Exception as e:
                print(f"OSQP solver failed: {str(e)}, trying SCS solver...")
                
                # Fallback to SCS if OSQP fails
                ef = EfficientFrontier(mu, S, solver="SCS")
                
                if optimize_for == "sharpe":
                    ef.max_sharpe()
                elif optimize_for == "return":
                    target_volatility = original_volatility * 1.2
                    ef.efficient_risk(target_volatility)
                elif optimize_for == "volatility":
                    ef.min_volatility()
                    
                weights = ef.clean_weights()
            
            return weights
            
        except Exception as e:
            print(f"Error in optimize_with_pypfopt: {str(e)}")
            # In case of any error, return the original weights
            return original_weights

    @classmethod
    def calculate_portfolio_value(cls, portfolio, data, start_date):
        """
        Calculate portfolio value over time starting from start_date
        portfolio: dict of {ticker: amount}
        data: DataFrame of price data
        start_date: date to start calculating from
        """
        # Convert start_date to datetime if it's a string
        start_date_dt = pd.to_datetime(start_date)
        
        # Find nearest previous valid date if start_date doesn't exist in the dataset
        valid_dates = data.index
        if start_date_dt not in valid_dates:
            # Find the nearest previous date that exists in the dataset
            valid_previous_dates = valid_dates[valid_dates < start_date_dt]
            if len(valid_previous_dates) == 0:
                raise ValueError(f"No valid dates before start_date {start_date_dt}")
            start_date_dt = valid_previous_dates[-1]
            print(f"Adjusted calculation start_date to nearest previous valid date: {start_date_dt}")
        
        # Get data after adjusted start_date
        future_data = data[data.index >= start_date_dt]
        
        # Calculate initial prices and shares
        initial_prices = data.loc[start_date_dt]
        shares = {ticker: amount/initial_prices[ticker] for ticker, amount in portfolio.items()}
        
        # Calculate portfolio values
        portfolio_values = sum(future_data[ticker] * shares[ticker] for ticker in portfolio)
        
        # Normalize to start at 1
        normalized_values = portfolio_values / portfolio_values.iloc[0]
        
        return normalized_values

# Load the data when app starts
try:
    Portfolio.load_data('trade_data.csv')
    print("Successfully loaded trade data")
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
    session.clear()
    response = redirect(url_for('home'))
    
    # Clear cache and prevent back navigation
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, post-check=0, pre-check=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Clear-Site-Data'] = '"cache", "cookies", "storage"'
    
    # Clear all cookies
    for cookie in request.cookies:
        response.delete_cookie(cookie)
    
    return response

@app.before_request
def check_session():
    if 'user_email' in session:
        # Check if the route requires authentication
        protected_routes = ['dashboard', 'portfolio', 'education', 'account']
        if request.endpoint in protected_routes:
            # Verify user exists and session is valid
            user = User.query.filter_by(email=session['user_email']).first()
            if not user:
                session.clear()
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

@app.route('/education')
def education():
    if 'user_email' not in session:
        return redirect(url_for('home'))
    
    user = User.query.filter_by(email=session['user_email']).first()
    if not user:
        session.pop('user_email', None)
        return redirect(url_for('home'))
    
    return render_template('education.html', username=user.username)

@app.route('/add-portfolio-item', methods=['POST'])
def add_portfolio_item():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        ticker = data['ticker'].upper()
        amount = float(data['amount'])
        
        # Check if ticker exists in stock_data.json
        with open('stock_data.json', 'r') as f:
            stock_data = json.load(f)
        
        if ticker not in stock_data:
            return jsonify({'error': 'Invalid ticker symbol'}), 400

        user = User.query.filter_by(email=session['user_email']).first()
        
        # Check if stock already exists in portfolio
        existing_item = PortfolioItem.query.filter_by(
            user_id=user.id,
            ticker=ticker
        ).first()
        
        if existing_item:
            # Add to existing amount
            existing_item.amount += amount
            db.session.commit()
            return jsonify({'message': 'Stock amount updated in portfolio'})
            
        # Add new stock to portfolio
        portfolio_item = PortfolioItem(
            user_id=user.id,
            ticker=ticker,
            amount=amount
        )
        
        db.session.add(portfolio_item)
        db.session.commit()
        
        return jsonify({'message': 'Stock added to portfolio'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

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
        # In production, use a minimal approach
        if os.environ.get('VERCEL_ENV') == 'production':
            # Use TOP_TICKERS as a minimal dataset
            results = []
            query = query.upper()
            for ticker in TOP_TICKERS:
                if query in ticker:
                    results.append({
                        'ticker': ticker,
                        'name': f"{ticker} Stock" # Simple placeholder name
                    })
            return results[:10]
        else:
            # In development, use the local JSON file
            with open('stock_data.json', 'r') as f:
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
    except Exception as e:
        print(f"Error in search_stocks: {str(e)}")
        # Minimal fallback
        return [{'ticker': t, 'name': f"{t} Stock"} for t in TOP_TICKERS if query.upper() in t][:10]

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
        # In production environment, don't try to load the JSON file
        if os.environ.get('VERCEL_ENV') == 'production':
            logger.info("Production environment detected, using generated data")
            # Create simplified data for top tickers
            top_companies = []
            for ticker in TOP_TICKERS:
                # Create basic info for each ticker
                company_data = {
                    'ticker': ticker,
                    'name': f"{ticker} Stock",
                    'sector': 'Technology',
                    'industry': 'Various',
                    'country': 'USA',
                    'exchange': 'NASDAQ/NYSE'
                }
                top_companies.append(company_data)
            return jsonify(top_companies)
        else:
            # In development, use the local JSON file
            with open('stock_data.json', 'r') as f:
                companies = json.load(f)
            
            # Debug: Print all TOP_TICKERS and which ones are missing
            logger.info(f"Looking for these tickers: {TOP_TICKERS}")
            
            # Get info for our predefined TOP_TICKERS in the exact order
            top_companies = []
            for ticker in TOP_TICKERS:
                if ticker in companies:
                    company_data = companies[ticker].copy()
                    top_companies.append(company_data)
            
            return jsonify(top_companies)
    except Exception as e:
        logger.error(f"Error in get_top_stocks: {str(e)}")
        # Provide fallback data
        fallback_companies = []
        for ticker in TOP_TICKERS:
            fallback_companies.append({
                'ticker': ticker,
                'name': f"{ticker} Stock",
                'sector': 'Unknown',
                'industry': 'Unknown'
            })
        return jsonify(fallback_companies)

@app.route('/get-portfolio-stocks')
def get_portfolio_stocks():
    if 'user_email' not in session:
        return jsonify([])
    
    try:
        user = User.query.filter_by(email=session['user_email']).first()
        portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
        
        # In production environment, don't try to load the JSON file
        if os.environ.get('VERCEL_ENV') == 'production':
            print("Production environment detected, using generated portfolio data")
            # Create simplified data for portfolio stocks
            portfolio_stocks = []
            for item in portfolio_items:
                # Create basic info for each ticker
                stock_info = {
                    'ticker': item.ticker,
                    'name': f"{item.ticker} Stock",
                    'sector': 'Various',
                    'industry': 'Various',
                    'amount': item.amount
                }
                portfolio_stocks.append(stock_info)
            return jsonify(portfolio_stocks)
        else:
            # In development, use the local JSON file
            with open('stock_data.json', 'r') as f:
                stock_data = json.load(f)
            
            portfolio_stocks = []
            for item in portfolio_items:
                if item.ticker in stock_data:
                    stock_info = stock_data[item.ticker].copy()
                    stock_info['amount'] = item.amount
                    portfolio_stocks.append(stock_info)
                else:
                    # Fallback for tickers not in the data
                    stock_info = {
                        'ticker': item.ticker,
                        'name': f"{item.ticker} Stock",
                        'amount': item.amount
                    }
                    portfolio_stocks.append(stock_info)
        
            return jsonify(portfolio_stocks)
    except Exception as e:
        print(f"Error getting portfolio stocks: {str(e)}")
        # Provide fallback data
        fallback_stocks = []
        for item in portfolio_items:
            fallback_stocks.append({
                'ticker': item.ticker,
                'name': f"{item.ticker} Stock",
                'amount': item.amount
            })
        return jsonify(fallback_stocks)

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
            end_date = pd.to_datetime(data['end_date']).strftime('%Y-%m-%d')
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

@app.route('/optimize-portfolio', methods=['POST'])
def optimize_portfolio():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        optimize_for = data.get('optimize_for', 'sharpe')  # Default to sharpe
        build_new = data.get('build_new', False)  # Default to False
        
        print(f"Optimizing portfolio with parameters: start_date={start_date}, end_date={end_date}, optimize_for={optimize_for}, build_new={build_new}")
        
        # Get user's portfolio
        user = User.query.filter_by(email=session['user_email']).first()
        portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
        
        if not portfolio_items:
            return jsonify({'error': 'Portfolio is empty'}), 400

        # Create portfolio dictionary with dollar values
        portfolio_input = {item.ticker: float(item.amount) for item in portfolio_items}
        print(f"Portfolio input: {portfolio_input}")
        
        # Load full ticker list
        full_tickers = pd.read_csv('updated_tickers.csv')['0'].tolist()
        
        try:
            # Call the optimization function
            optimized_portfolio = Portfolio.optimize_user_portfolio(
                portfolio_input=portfolio_input,
                start_date=start_date,
                end_date=end_date,
                optimize_for=optimize_for,
                build_new=build_new,
                full_tickers_list=full_tickers
            )
            
            print(f"Optimization successful: {optimized_portfolio}")
            return jsonify({
                'success': True,
                'optimized_portfolio': optimized_portfolio,
                'message': 'Portfolio optimization completed successfully'
            })
            
        except Exception as e:
            print(f"Detailed optimization error: {str(e)}")
            return jsonify({'error': f'Failed to optimize portfolio: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Detailed general error: {str(e)}")
        return jsonify({'error': f'Failed to process request: {str(e)}'}), 500

@app.errorhandler(403)
def forbidden_error(error):
    session.clear()  # Clear any corrupted session
    return redirect(url_for('home'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()  # Rollback any failed database transactions
    return jsonify({'error': str(error)}), 500

# def update_all_stock_data():
#     """Update data for all stocks in portfolios and TOP_TICKERS"""
#     try:
#         # Get all unique tickers from portfolios
#         portfolio_tickers = PortfolioItem.query.with_entities(
#             PortfolioItem.ticker).distinct().all()
#         portfolio_tickers = [item[0] for item in portfolio_tickers]
        
#         # Combine with TOP_TICKERS
#         all_tickers = list(set(TOP_TICKERS + portfolio_tickers))
        
#         # Update data for all tickers
#         update_stock_data(all_tickers)
#         print("Successfully updated all stock data")
#     except Exception as e:
#         print(f"Error updating stock data: {e}")

@app.route('/calculate-future-performance', methods=['POST'])
def calculate_future_performance():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    try:
        data = request.json
        
        # Ensure consistent date formatting
        try:
            start_date_str = pd.to_datetime(data['start_date']).strftime('%Y-%m-%d')
            end_date_str = pd.to_datetime(data['end_date']).strftime('%Y-%m-%d')
            include_build_new = data['include_build_new']
            
            # Get valid dates from the dataset (dates that exist in the index)
            df = Portfolio.trade_data
            valid_dates = df.index
            
            # Find nearest previous valid dates if the exact dates don't exist
            start_date = pd.to_datetime(start_date_str)
            if start_date not in valid_dates:
                # Find the nearest previous date that exists in the dataset
                valid_previous_dates = valid_dates[valid_dates < start_date]
                if len(valid_previous_dates) == 0:
                    return jsonify({'error': 'No valid dates before the specified start date'}), 400
                start_date = valid_previous_dates[-1]
                print(f"Adjusted start_date to nearest previous valid date: {start_date}")
            
            end_date = pd.to_datetime(end_date_str)
            if end_date not in valid_dates:
                # Find the nearest previous date that exists in the dataset
                valid_previous_dates = valid_dates[valid_dates < end_date]
                if len(valid_previous_dates) == 0:
                    return jsonify({'error': 'No valid dates before the specified end date'}), 400
                end_date = valid_previous_dates[-1]
                print(f"Adjusted end_date to nearest previous valid date: {end_date}")
            
            # Convert dates back to string format for consistency
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"Date parsing error: {str(e)}")
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD format.'}), 400

        # Get current user's portfolio
        user = User.query.filter_by(email=session['user_email']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
        if not portfolio_items:
            return jsonify({'error': 'Portfolio is empty'}), 400
            
        current_portfolio = {item.ticker: item.amount for item in portfolio_items}
        
        # Use the trade data loaded at app startup
        df = Portfolio.trade_data
        
        portfolios = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # Add current portfolio performance without optimization
        base_values = Portfolio.calculate_portfolio_value(current_portfolio, df, end_date_str)
        
        # Calculate metrics for current portfolio without optimization
        # Create a Portfolio object to evaluate the current portfolio
        tickers = list(current_portfolio.keys())
        total_value = sum(current_portfolio.values())
        weights = [current_portfolio[ticker]/total_value for ticker in tickers]
        
        try:
            portfolio_eval = Portfolio(tickers, weights, start_date_str, end_date_str)
            metrics = portfolio_eval.get_risk_metrics()
            
            # Format metrics for the details modal
            current_metrics = {
                'metrics': {
                    'Current Return': metrics['Annualized Return'],
                    'Current Volatility': metrics['Annualized Volatility'],
                    'Current Sharpe': metrics['Sharpe Ratio'],
                    'Optimized Return': metrics['Annualized Return'],  # Same as current
                    'Optimized Volatility': metrics['Annualized Volatility'],  # Same as current
                    'Optimized Sharpe': metrics['Sharpe Ratio']  # Same as current
                },
                'allocation': current_portfolio  # Use the actual current allocation
            }
        except Exception as e:
            print(f"Error evaluating current portfolio: {str(e)}")
            # Fallback to a simple structure if evaluation fails
            current_metrics = {
                'metrics': {
                    'Current Return': 0,
                    'Current Volatility': 0,
                    'Current Sharpe': 0,
                    'Optimized Return': 0,
                    'Optimized Volatility': 0,
                    'Optimized Sharpe': 0
                },
                'allocation': current_portfolio
            }
        
        portfolios.append({
            'name': 'User Portfolio',
            'values': base_values.tolist(),
            'color': colors[0],
            'optimizationData': current_metrics
        })
        
        # Load full ticker list for build_new portfolios
        full_tickers = pd.read_csv('updated_tickers.csv')['0'].tolist()
        
        # Get and store the current portfolio metrics first to ensure consistency
        # This will be used for all optimization methods
        current_portfolio_metrics = None
        
        # First get current portfolio metrics that will stay consistent
        try:
            first_result = Portfolio.optimize_user_portfolio(
                portfolio_input=current_portfolio,
                start_date=start_date_str,
                end_date=end_date_str,
                optimize_for='sharpe',
                build_new=False,
                full_tickers_list=full_tickers
            )
            
            current_portfolio_metrics = {
                'Current Return': first_result['metrics']['Current Return'],
                'Current Volatility': first_result['metrics']['Current Volatility'],
                'Current Sharpe': first_result['metrics']['Current Sharpe']
            }
        except Exception as e:
            print(f"Error getting baseline portfolio metrics: {str(e)}")
            current_portfolio_metrics = {
                'Current Return': 0,
                'Current Volatility': 0,
                'Current Sharpe': 0
            }
        
        # Optimize for different strategies
        strategies = ['sharpe', 'return', 'volatility']
        for i, strategy in enumerate(strategies):
            try:
                optimized_result = Portfolio.optimize_user_portfolio(
                    portfolio_input=current_portfolio,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    optimize_for=strategy,
                    build_new=False,
                    full_tickers_list=full_tickers
                )
                
                # Keep the current portfolio metrics consistent
                optimized_result['metrics']['Current Return'] = current_portfolio_metrics['Current Return']
                optimized_result['metrics']['Current Volatility'] = current_portfolio_metrics['Current Volatility']
                optimized_result['metrics']['Current Sharpe'] = current_portfolio_metrics['Current Sharpe']
                
                values = Portfolio.calculate_portfolio_value(optimized_result['allocation'], df, end_date_str)
                portfolios.append({
                    'name': f'Adjusted Portfolio ({strategy})',
                    'values': values.tolist(),
                    'color': colors[i+1],
                    'optimizationData': optimized_result
                })
            except Exception as e:
                print(f"Error optimizing portfolio with strategy {strategy}: {str(e)}")
                # Skip adding this portfolio if optimization fails
                continue
                
            if include_build_new:
                try:
                    new_portfolio_result = Portfolio.optimize_user_portfolio(
                        portfolio_input=current_portfolio,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        optimize_for=strategy,
                        build_new=True,
                        full_tickers_list=full_tickers
                    )
                    
                    # Keep the current portfolio metrics consistent
                    new_portfolio_result['metrics']['Current Return'] = current_portfolio_metrics['Current Return']
                    new_portfolio_result['metrics']['Current Volatility'] = current_portfolio_metrics['Current Volatility']
                    new_portfolio_result['metrics']['Current Sharpe'] = current_portfolio_metrics['Current Sharpe']
                    
                    values = Portfolio.calculate_portfolio_value(new_portfolio_result['allocation'], df, end_date_str)
                    portfolios.append({
                        'name': f'Built New Portfolio ({strategy})',
                        'values': values.tolist(),
                        'color': colors[i+4],
                        'optimizationData': new_portfolio_result
                    })
                except Exception as e:
                    print(f"Error building new portfolio with strategy {strategy}: {str(e)}")
                    # Skip adding this portfolio if optimization fails
                    continue
        
        # Check if we have any portfolios after attempting optimization
        if len(portfolios) <= 1:  # Only the original portfolio is available
            return jsonify({'error': 'Unable to optimize portfolio with the selected date range. Try a different period with positive returns.'}), 400
        
        # Resample data to weekly points before calculating values
        weekly_df = Portfolio.trade_data.resample('W').last()  # Resample to weekly data
        
        # For future dates, we need to ensure we find dates after our end_date
        # First convert end_date back to datetime to compare with index
        end_date_dt = pd.to_datetime(end_date)
        future_dates = weekly_df.index[weekly_df.index >= end_date_dt]
        
        if len(future_dates) == 0:
            return jsonify({'error': 'No future dates available after the specified end date'}), 400
        
        return jsonify({
            'dates': future_dates.strftime('%Y-%m-%d').tolist(),
            'portfolios': portfolios
        })
        
    except Exception as e:
        print(f"Error in calculate_future_performance: {str(e)}")
        return jsonify({'error': f'Error calculating future performance: {str(e)}'}), 500

@app.route('/update-portfolio', methods=['POST'])
def update_portfolio():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        allocation = data.get('allocation', {})
        
        if not allocation:
            return jsonify({'success': False, 'error': 'No allocation provided'}), 400
        
        # Get the current user
        user = User.query.filter_by(email=session['user_email']).first()
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        # Delete existing portfolio items for this user
        PortfolioItem.query.filter_by(user_id=user.id).delete()
        
        # Add new portfolio items
        for ticker, amount in allocation.items():
            new_item = PortfolioItem(ticker=ticker, amount=amount, user_id=user.id)
            db.session.add(new_item)
        
        db.session.commit()
        
        return jsonify({'success': True})
    
    except Exception as e:
        db.session.rollback()
        print(f"Error updating portfolio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/monte-carlo-prediction', methods=['POST'])
def monte_carlo_prediction():
    if 'user_email' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.json
        
        # Extract parameters from request
        start_date_str = pd.to_datetime(data['start_date']).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(data['end_date']).strftime('%Y-%m-%d')
        num_simulations = int(data.get('num_simulations', 100))
        confidence_interval = float(data.get('confidence_interval', 0.95))
        
        # Get portfolio data from the request or user's database record
        if 'portfolio' in data and data['portfolio']:
            # Use the provided portfolio
            portfolio_data = {item['ticker']: item['amount'] for item in data['portfolio']}
        else:
            # Get user's portfolio from database
            user = User.query.filter_by(email=session['user_email']).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
                
            portfolio_items = PortfolioItem.query.filter_by(user_id=user.id).all()
            if not portfolio_items:
                return jsonify({'error': 'Portfolio is empty'}), 400
                
            portfolio_data = {item.ticker: item.amount for item in portfolio_items}
        
        # Validate portfolio data
        if not portfolio_data:
            return jsonify({'error': 'No portfolio data available'}), 400
        
        # Validate tickers
        valid_tickers = [ticker for ticker in portfolio_data.keys() if ticker in Portfolio.trade_data.columns]
        if not valid_tickers:
            return jsonify({'error': 'No valid tickers in portfolio'}), 400
        
        # Filter out invalid tickers
        valid_portfolio = {ticker: portfolio_data[ticker] for ticker in valid_tickers}
        
        # Get the last date in the dataset for determining prediction period
        last_date = Portfolio.trade_data.index[-1]
        may_29_2025 = pd.to_datetime('2025-05-29')
        
        # Calculate business days between last date and may_29_2025
        prediction_days = len(pd.date_range(start=last_date, end=may_29_2025, freq='B')) - 1
        
        # Run Monte Carlo simulation
        result = Portfolio.monte_carlo_simulation(
            portfolio=valid_portfolio,
            num_simulations=num_simulations,
            confidence_interval=confidence_interval,
            start_date=start_date_str,
            end_date=end_date_str,
            prediction_days=prediction_days
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in monte_carlo_prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

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