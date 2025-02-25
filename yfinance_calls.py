import yfinance as yf
import json

def fetch_stock_data(ticker):
    """
    Fetches stock data for a single ticker.
    Returns a dictionary with ticker, name, price, marketCap, and change (%) for today.
    If regular market data is missing (e.g. after hours), falls back to previous close and
    calculates change using previous close and previous-previous close.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Try to get the long name; fallback to short name if unavailable.
        name = info.get('longName', info.get('shortName', 'N/A'))
        
        # Get regular market price. If not available, fallback to previous close.
        regular_price = info.get('RegularMarketPrice')
        if regular_price is None or regular_price == 'N/A':
            regular_price = info.get('previousClose', 'N/A')
        
        # Get regular market change percent.
        regular_change = info.get('RegularMarketChangePercent')
        if regular_change is None or regular_change == 'N/A':
            # If regular change percent is missing, try calculating it using historical data.
            hist = stock.history(period='5d')
            if not hist.empty and len(hist) >= 2:
                # Use the last two available trading days.
                previous_close = hist['Close'].iloc[-1]
                previous_previous_close = hist['Close'].iloc[-2]
                if previous_previous_close != 0:
                    regular_change = ((previous_close - previous_previous_close) / previous_previous_close) * 100
                else:
                    regular_change = 'N/A'
            else:
                regular_change = 'N/A'
        
        data = {
            "ticker": ticker,
            "name": name,
            "price": regular_price,
            "marketCap": info.get('marketCap', 'N/A'),
            "change": regular_change,  # today's change in percent
            "AH price": info.get('postMarketPrice', 'N/A'),  # After Hours price
            "AH change": info.get('postMarketChangePercent', 'N/A')  # After Hours price change
        }
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {
            "ticker": ticker,
            "name": "N/A",
            "price": "N/A",
            "marketCap": "N/A",
            "change": "N/A"
        }

def pull_all_stock_data(ticker_list):
    """
    Pulls stock data for all tickers in ticker_list,
    saves the results in 'stock_data.json',
    and returns the data as a dictionary.
    """
    stock_data = {}
    for ticker in ticker_list:
        stock_data[ticker] = fetch_stock_data(ticker)
    with open('stock_data.json', 'w') as json_file:
        json.dump(stock_data, json_file, indent=4)
    print("Stock data has been saved to stock_data.json")
    return stock_data

def update_stock_data(tickers):
    """
    Updates the existing stock_data.json with new data for the given tickers.
    For each ticker provided, fetches the latest data and updates the file.
    Returns a dictionary containing only the updated tickers' data.
    """
    try:
        with open('stock_data.json', 'r') as json_file:
            stock_data = json.load(json_file)
    except FileNotFoundError:
        stock_data = {}

    for ticker in tickers:
        stock_data[ticker] = fetch_stock_data(ticker)
    
    with open('stock_data.json', 'w') as json_file:
        json.dump(stock_data, json_file, indent=4)
    print(f"Updated data for tickers: {', '.join(tickers)}")
    
    updated_data = {ticker: stock_data[ticker] for ticker in tickers}
    return updated_data

