import schedule
import time
import datetime
import pytz
import yfinance as yf
import json

# --- Existing Functions ---

def fetch_stock_data(ticker):
    """
    Fetches stock data for a single ticker.
    Returns a dictionary with ticker, name, price, marketCap, and change (%) for today.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Try to get the long name; fallback to short name if unavailable.
        name = info.get('longName', info.get('shortName', 'N/A'))
        data = {
            "ticker": ticker,
            "name": name,
            "price": info.get('postMarketPrice', 'N/A'),
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
    # Save the entire stock data to a JSON file.
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
    # Load existing data if file exists; otherwise, start with an empty dict.
    try:
        with open('stock_data.json', 'r') as json_file:
            stock_data = json.load(json_file)
    except FileNotFoundError:
        stock_data = {}

    # Update each specified ticker with fresh data.
    for ticker in tickers:
        stock_data[ticker] = fetch_stock_data(ticker)
    
    # Write the updated data back to the JSON file.
    with open('stock_data.json', 'w') as json_file:
        json.dump(stock_data, json_file, indent=4)
    print(f"Updated data for tickers: {', '.join(tickers)}")
    
    # Return only the data for the updated tickers.
    updated_data = {ticker: stock_data[ticker] for ticker in tickers}
    return updated_data

# --- Scheduling Code ---

# Define the list of all stocks you want to update at 9:30 AM and 4:00 PM ET.
all_stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "FB"]  # update as needed

def update_major_stocks():
    """Update major stocks every minute."""
    print(f"{datetime.datetime.now()}: Updating major stocks...")
    updated = update_stock_data(["AAPL", "GOOG", "MSFT"])
    print("Major stocks updated:", updated)

def update_all_stocks_job():
    """Update all stocks (all_stocks list) at scheduled times."""
    # For clarity, we also print the current ET time.
    et = pytz.timezone('US/Eastern')
    now_et = datetime.datetime.now(et).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now_et} ET: Updating all stocks...")
    updated = update_stock_data(all_stocks)
    print("All stocks updated:", updated)

# Schedule major stocks update every minutes.
schedule.every(15).minutes.do(update_major_stocks)

# Schedule all_stocks update at 9:30 AM and 4:00 PM ET, Monday through Friday.
# If your system time is not ET, you may need to adjust these times accordingly.
for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
    getattr(schedule.every(), day).at("09:30").do(update_all_stocks_job)
    getattr(schedule.every(), day).at("16:00").do(update_all_stocks_job)

print("Scheduler is running...")
