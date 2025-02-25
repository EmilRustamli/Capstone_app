from yfinance_calls import update_stock_data

TOP_TICKERS = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'GOOG', 'META', 
               'TSLA', 'AVGO', 'BRK-B', 'TSM', 'WMT', '2222.SR']

update_stock_data(TOP_TICKERS)

# Somehow it just runs app instead of actually running the function
# I need to figure out why and fix it

