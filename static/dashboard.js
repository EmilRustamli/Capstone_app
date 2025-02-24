// Function to fetch and display portfolio stocks
function loadPortfolioStocks() {
    fetch('/get-portfolio-stocks')
        .then(response => response.json())
        .then(stocks => {
            const stocksContainer = document.getElementById('portfolioStocks');
            if (!stocks || stocks.length === 0) {
                stocksContainer.innerHTML = `
                    <div class="no-stocks-message">
                        <p>No stocks in your portfolio. Visit the Portfolio page to add stocks.</p>
                    </div>`;
                return;
            }

            let html = '';
            stocks.forEach(stock => {
                const changeClass = parseFloat(stock.change) >= 0 ? 'positive' : 'negative';
                const changeSymbol = parseFloat(stock.change) >= 0 ? '+' : '';
                
                html += `
                    <div class="stock-card">
                        <div class="stock-header">
                            <h3>${stock.ticker}</h3>
                            <span class="stock-name">${stock.name}</span>
                        </div>
                        <div class="stock-details">
                            <div class="stock-price">$${parseFloat(stock.price).toFixed(2)}</div>
                            <div class="stock-change ${changeClass}">
                                ${changeSymbol}${parseFloat(stock.change).toFixed(2)}%
                            </div>
                            <div class="stock-amount">
                                Shares: ${stock.amount}
                            </div>
                        </div>
                    </div>`;
            });
            stocksContainer.innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading portfolio stocks:', error);
            document.getElementById('portfolioStocks').innerHTML = `
                <div class="error-message">
                    <p>Error loading portfolio stocks. Please try again later.</p>
                </div>`;
        });
}

// Load stocks when page loads
document.addEventListener('DOMContentLoaded', loadPortfolioStocks); 