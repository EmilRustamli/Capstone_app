import yfinance as yf
import json
import time

def fetch_stock_data(ticker):
    """
    Fetches stock data for a single ticker.
    Returns a dictionary with ticker, name, price, marketCap, and change (%) for today.
    """
    try:
        print(f"Fetching data for {ticker}...")
        time.sleep(2)  # Add delay to avoid rate limiting
        
        stock = yf.Ticker(ticker)
        info = stock.info
        # Try to get the long name; fallback to short name if unavailable.
        name = info.get('longName', info.get('shortName', 'N/A'))
        data = {
            "ticker": ticker,
            "name": name,
            "price": info.get('regularMarketPrice', 'N/A'),
            "marketCap": info.get('marketCap', 'N/A'),
            "change": info.get('regularMarketChangePercent', 'N/A')  # today's change in percent
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
        time.sleep(2)  # Add delay between requests
    # Save the entire stock data to a JSON file.
    with open('stock_data.json', 'w') as json_file:
        json.dump(stock_data, json_file, indent=4)
    print("Stock data has been saved to stock_data.json")
    return stock_data

def update_stock_data(tickers):
    """
    Updates the existing stock_data.json with new data for the given tickers.
    For each ticker provided, fetches the latest data and updates the file.
    """
    # Load existing data if file exists; otherwise, start with an empty dict.
    try:
        with open('stock_data.json', 'r') as json_file:
            stock_data = json.load(json_file)
    except FileNotFoundError:
        stock_data = {}

    # Update each specified ticker with fresh data.
    for ticker in tickers:
        stock_data[ticker] = fetch_stock_data(ticker)
        time.sleep(2)  # Add delay between requests
    
    # Write the updated data back to the JSON file.
    with open('stock_data.json', 'w') as json_file:
        json.dump(stock_data, json_file, indent=4)
    print(f"Updated data for tickers: {', '.join(tickers)}")
    return stock_data

if __name__ == "__main__":
    # Test with some sample tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    print("Testing stock data fetching...")
    result = pull_all_stock_data(test_tickers)
    print("\nFetched data:")
    for ticker, data in result.items():
        print(f"{ticker}: {data}") 