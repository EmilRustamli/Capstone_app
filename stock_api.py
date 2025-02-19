import yfinance as yf
from datetime import datetime

def get_stock_quote(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and previous close
        current_price = info.get('currentPrice', 0)
        previous_close = info.get('previousClose', 0)
        
        # Calculate percentage change
        if previous_close:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        else:
            change_percent = 0
            
        return {
            'price': current_price,
            'change_percent': change_percent
        }
    except Exception as e:
        print(f"Error fetching quote for {ticker}: {str(e)}")
        return None

def update_stock_prices(companies):
    """Update prices for all companies"""
    updated_companies = companies.copy()
    
    # Get all tickers at once for better performance
    tickers = list(companies.keys())
    
    try:
        # Fetch data for all tickers in one batch
        print(f"Fetching data for tickers: {tickers}")  # Debug print
        data = yf.download(tickers, period='2d', interval='1d', group_by='ticker', progress=False)
        print("Data fetched successfully")  # Debug print
        
        for ticker in companies:
            try:
                if len(tickers) == 1:
                    current_price = data['Close'][-1]
                    prev_price = data['Close'][-2]
                else:
                    current_price = data[ticker]['Close'][-1]
                    prev_price = data[ticker]['Close'][-2]
                
                change_percent = ((current_price - prev_price) / prev_price) * 100
                
                updated_companies[ticker].update({
                    'price': f"{current_price:.2f}",
                    'change': f"{change_percent:.2f}"
                })
                print(f"Updated {ticker}: Price=${current_price:.2f}, Change={change_percent:.2f}%")  # Debug print
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                # If there's an error, keep the existing price and change
                continue
                
    except Exception as e:
        print(f"Error fetching batch data: {str(e)}")
        # If there's an error fetching data, return the original companies data
        return companies
    
    return updated_companies 