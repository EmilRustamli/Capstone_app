import yfinance as yf
import pandas as pd
from pathlib import Path
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_tickers(file_path='tickers.csv'):
    """Read tickers from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df['Tickers'].tolist()
    except Exception as e:
        logger.error(f"Error reading tickers: {str(e)}")
        return []

def get_company_info(tickers):
    """Get company names for the tickers using yfinance."""
    company_info = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Getting info for {ticker}")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_info[ticker] = {
                "ticker": ticker,
                "name": info.get('longName', 'N/A')
            }
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {str(e)}")
            continue
    
    return company_info

def main():
    # Read tickers from CSV
    tickers = read_tickers()
    logger.info(f"Found {len(tickers)} tickers")
    
    # Get company info
    company_info = get_company_info(tickers)
    logger.info(f"Retrieved info for {len(company_info)} companies")
    
    # Save to JSON
    with open('company_info.json', 'w') as f:
        json.dump(company_info, f, indent=4)
    logger.info("Saved company info to company_info.json")

if __name__ == "__main__":
    main() 