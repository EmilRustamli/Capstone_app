async function handleAddPortfolioItem(event) {
    event.preventDefault();
    const form = event.target;
    const ticker = document.getElementById('ticker').value;
    const amount = document.getElementById('amount').value;

    try {
        const response = await fetch('/add-portfolio-item', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ticker, amount })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Add new item to the list
            const portfolioItems = document.getElementById('portfolioItems');
            const newItem = document.createElement('div');
            newItem.className = 'portfolio-item';
            newItem.innerHTML = `
                <span class="ticker">${data.item.ticker}</span>
                <span class="amount">$${data.item.amount.toFixed(2)}</span>
            `;
            portfolioItems.appendChild(newItem);
            form.reset();
        }
    } catch (error) {
        console.error('Portfolio add error:', error);
    }
} 