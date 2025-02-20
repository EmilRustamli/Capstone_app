import yfinance as yf
import pandas as pd
from pathlib import Path
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Top stocks we want to track frequently
TOP_TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOG', '2222.SR', 'META', 
               'TSLA', 'AVGO', 'BRK-B', 'TSM', 'WMT']

def read_tickers(file_path='tickers.csv'):
    """Read all tickers from CSV file and add missing ones."""
    try:
        # First read existing tickers
        df = pd.read_csv(file_path)
        existing_tickers = df['Tickers'].tolist()
        logger.info(f"Found {len(existing_tickers)} existing tickers")
        
        # Define missing tickers
        missing_tickers = ['2222.SR', 'BRK-B', 'TSM']
        
        # Add missing tickers
        added_tickers = []
        for ticker in missing_tickers:
            if ticker not in existing_tickers:
                added_tickers.append(ticker)
                
        if added_tickers:
            # Create new dataframe with all tickers
            new_df = pd.DataFrame({'Tickers': existing_tickers + added_tickers})
            # Save back to CSV
            new_df.to_csv(file_path, index=False)
            logger.info(f"Added {len(added_tickers)} new tickers: {added_tickers}")
            return new_df['Tickers'].tolist()
        
        return existing_tickers
    except Exception as e:
        logger.error(f"Error reading/updating tickers: {str(e)}")
        return []

def get_portfolio_tickers():
    """Get unique tickers from all user portfolios"""
    from app import db, PortfolioItem
    try:
        portfolio_items = PortfolioItem.query.with_entities(PortfolioItem.ticker).distinct().all()
        return [item[0] for item in portfolio_items]
    except Exception as e:
        logger.error(f"Error getting portfolio tickers: {str(e)}")
        return []

def get_company_info(tickers):
    """Get company info including price, market cap, and change percentage."""
    company_info = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Getting info for {ticker}")
            
            # Add longer delay between requests
            time.sleep(2)  # Increased from 0.5 to 2 seconds
            
            # Download data for single ticker
            data = yf.download(ticker, period='2d', progress=False)
            if data.empty:
                logger.error(f"Received empty data for {ticker}")
                continue
                
            stock = yf.Ticker(ticker)
            time.sleep(2)  # Increased from 1 to 2 seconds
            
            info = stock.info
            if not info:
                logger.error(f"No info received for {ticker}")
                continue
            
            # Get current and previous day prices
            current_price = data['Close'][-1]
            prev_price = data['Close'][-2]
            
            # Calculate percentage change
            change_percent = ((current_price - prev_price) / prev_price) * 100
            
            # Get market cap in billions
            market_cap = info.get('marketCap', 0) / 1_000_000_000

            company_info[ticker] = {
                "ticker": ticker,
                "name": info.get('longName', 'N/A'),
                "price": f"{current_price:.2f}",
                "marketCap": f"{market_cap:.0f}",
                "change": f"{change_percent:.2f}"
            }
            
            logger.info(f"Successfully processed {ticker}")
            time.sleep(0.5)  # Add delay between tickers
            
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {str(e)}")
            continue
    
    return company_info

def update_company_info(new_data):
    """Update company_info.json with new data while preserving existing data"""
    try:
        # Read existing data
        with open('company_info.json', 'r') as f:
            existing_data = json.load(f)
        
        # Update only the tickers we have new data for
        existing_data.update(new_data)
        
        # Save back to file
        with open('company_info.json', 'w') as f:
            json.dump(existing_data, f, indent=4)
            
        logger.info(f"Updated {len(new_data)} stocks in company_info.json")
    except Exception as e:
        logger.error(f"Error updating company_info.json: {str(e)}")

def main():
    """Update TOP_TICKERS and portfolio stocks"""
    try:
        # First update tickers.csv
        all_tickers = read_tickers()
        logger.info(f"Total tickers after update: {len(all_tickers)}")
        
        # Then get portfolio tickers
        portfolio_tickers = get_portfolio_tickers()
        
        # Combine all tickers we need to update
        tickers = list(set(TOP_TICKERS + portfolio_tickers))
        
        # Verify tickers are valid
        tickers = [t.strip() for t in tickers if t and isinstance(t, str)]
        logger.info(f"Updating these tickers: {tickers}")
        
        # Get new data for selected tickers
        new_data = get_company_info(tickers)
        
        if new_data:
            update_company_info(new_data)
        else:
            logger.error("No data received to update")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 