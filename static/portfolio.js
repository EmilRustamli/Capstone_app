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

// Monte Carlo Prediction
let monteCarloChart = null;

function showMonteCarloPrediction() {
    document.getElementById('monteCarloModal').style.display = 'flex';
    
    // Set default dates
    const today = new Date();
    const startDateEl = document.getElementById('mcStartDate');
    const endDateEl = document.getElementById('mcEndDate');
    
    // Set start date to 90 days ago
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 90);
    startDateEl.value = formatDateForInput(startDate);
    
    // Set end date to today
    endDateEl.value = formatDateForInput(today);
    
    // Hide results section until analysis is run
    document.getElementById('mcResults').style.display = 'none';
}

function closeMonteCarloModal() {
    document.getElementById('monteCarloModal').style.display = 'none';
    // Clean up the chart when closing the modal
    if (monteCarloChart) {
        monteCarloChart.destroy();
        monteCarloChart = null;
    }
}

function resetMonteCarloZoom() {
    if (monteCarloChart) {
        console.log('Resetting Monte Carlo chart zoom...');
        monteCarloChart.resetZoom();
        console.log('Monte Carlo chart zoom reset');
    } else {
        console.error('Monte Carlo chart not initialized');
    }
}

function runMonteCarloPrediction() {
    const startDate = document.getElementById('mcStartDate').value;
    const endDate = document.getElementById('mcEndDate').value;
    const numSimulations = document.getElementById('numSimulations').value;
    const confidenceInterval = document.getElementById('confidenceInterval').value;
    
    if (!startDate || !endDate) {
        alert('Please select both start and end dates.');
        return;
    }
    
    // Show loading state
    const resultsDiv = document.getElementById('mcResults');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<h3>Running Monte Carlo Simulations...</h3><div class="loading-spinner"></div><p>This may take a minute as we generate multiple portfolio optimizations.</p>';
    
    // Ensure any existing chart is properly destroyed
    if (monteCarloChart) {
        monteCarloChart.destroy();
        monteCarloChart = null;
    }
    
    // Add debugging info
    console.log('Starting Monte Carlo prediction - Charts.js version:', Chart.version);
    
    // Collect all tickers and investment amounts for the current portfolio
    const portfolio = [];
    document.querySelectorAll('.portfolio-item').forEach(item => {
        const ticker = item.querySelector('.ticker').textContent;
        const amount = parseFloat(item.querySelector('.amount').textContent.replace('$', '').replace(',', ''));
        portfolio.push({ ticker, amount });
    });
    
    // Set target prediction end date to June 30, 2025
    const predictionEndDate = '2025-06-30';
    
    // Step 1: First get portfolio optimizations (similar to calculate_future_performance)
    fetch('/calculate-future-performance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate,
            include_build_new: true
        }),
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || 'An error occurred getting portfolio optimizations');
            });
        }
        return response.json();
    })
    .then(async portfoliosData => {
        // Get all optimized portfolios from the response
        const portfoliosToSimulate = portfoliosData.portfolios;
        
        // Limit to 7 portfolios if we have more
        const limitedPortfolios = portfoliosToSimulate.slice(0, 7);
        
        // Create an array to store simulation results
        const simulationResults = [];
        
        // Add placeholder data to track progress
        resultsDiv.innerHTML = '<h3>Running Monte Carlo Simulations...</h3><div class="loading-spinner"></div><div id="mcProgress">Running simulation 1 of ' + limitedPortfolios.length + '</div>';
        
        // Run Monte Carlo simulations for each portfolio
        for (let i = 0; i < limitedPortfolios.length; i++) {
            // Update progress
            document.getElementById('mcProgress').innerText = `Running simulation ${i+1} of ${limitedPortfolios.length}: ${limitedPortfolios[i].name}`;
            
            // Convert portfolio optimization data to format needed for Monte Carlo
            const currentOptPortfolio = limitedPortfolios[i];
            const portfolioAllocation = currentOptPortfolio.optimizationData.allocation;
            
            // Convert allocation object to array of objects with ticker and amount
            const portfolioArray = Object.entries(portfolioAllocation).map(([ticker, amount]) => ({
                ticker,
                amount
            }));
            
            try {
                // Run Monte Carlo simulation for this portfolio
                const mcResult = await fetch('/monte-carlo-prediction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        start_date: startDate,
                        end_date: endDate,
                        prediction_end_date: predictionEndDate,
                        portfolio: portfolioArray,
                        num_simulations: numSimulations,
                        confidence_interval: confidenceInterval,
                        use_known_data: true
                    }),
                }).then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to run Monte Carlo simulation for optimized portfolio');
                    }
                    return response.json();
                });
                
                // Store the result with portfolio info
                simulationResults.push({
                    name: currentOptPortfolio.name,
                    color: currentOptPortfolio.color,
                    mcData: mcResult,
                    optimizationData: currentOptPortfolio.optimizationData
                });
            } catch (error) {
                console.error(`Error running Monte Carlo for portfolio ${i}:`, error);
                // Continue with other portfolios even if one fails
            }
        }
        
        // Display the combined Monte Carlo simulation results
        displayMultiPortfolioMonteCarloResults(simulationResults);
    })
    .catch(error => {
        console.error('Error:', error);
        resultsDiv.innerHTML = `<h3>Error</h3><p>${error.message}</p>`;
    });
}

function displayMonteCarloResults(data) {
    const resultsDiv = document.getElementById('mcResults');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `
        <h3>Monte Carlo Portfolio Prediction</h3>
        <div class="chart-container">
            <canvas id="monteCarloChart"></canvas>
        </div>
        <div class="chart-actions">
            <button class="reset-zoom-btn" onclick="resetMonteCarloZoom()">Reset Zoom</button>
        </div>
    `;
    
    // Clean up any existing chart
    if (monteCarloChart) {
        monteCarloChart.destroy();
    }
    
    // Create Monte Carlo chart
    const ctx = document.getElementById('monteCarloChart').getContext('2d');
    
    // Extract data for chart
    const dates = data.dates;
    const median = data.median_path;
    const upperBound = data.upper_bound;
    const lowerBound = data.lower_bound;
    const historicalData = data.historical_path;
    const knownFutureData = data.known_future_path || [];
    
    // Determine where known data ends and predictions begin
    const lastKnownDataDate = data.last_known_data_date;
    const lastKnownDataIndex = dates.findIndex(date => date === lastKnownDataDate);
    
    // Find the value at the historical end date (last point of historical data)
    const normalizeValue = historicalData[historicalData.length - 1];
    
    // Normalize all data series based on this value
    const normalizedMedian = median.map(val => val / normalizeValue);
    const normalizedUpperBound = upperBound.map(val => val / normalizeValue);
    const normalizedLowerBound = lowerBound.map(val => val / normalizeValue);
    const normalizedHistorical = historicalData.map(val => val / normalizeValue);
    const normalizedKnownFuture = knownFutureData.length > 0 ? knownFutureData.map(val => val / normalizeValue) : [];
    
    // Find max and min values for y-axis scaling (using normalized values)
    const allValues = [
        ...normalizedMedian, 
        ...normalizedUpperBound, 
        ...normalizedLowerBound, 
        ...normalizedHistorical,
        ...normalizedKnownFuture
    ].filter(val => !isNaN(val));
    
    const maxValue = Math.max(...allValues);
    const minValue = Math.min(...allValues);
    const dataRange = maxValue - minValue;
    const padding = dataRange * 0.1;
    const yMax = Math.round((maxValue + padding) * 100) / 100;
    const yMin = Math.round((minValue - padding) * 100) / 100;
    
    // Create datasets array
    const datasets = [];
    
    // Add historical data (from start_date to end_date)
    datasets.push({
        label: 'Historical Data',
        data: normalizedHistorical,
        borderColor: 'rgba(0, 0, 255, 1)',
        borderWidth: 2,
        pointRadius: 0,
        fill: false
    });
    
    // Add known future data if available (from end_date to last known date)
    if (normalizedKnownFuture && normalizedKnownFuture.length > 0) {
        datasets.push({
            label: 'Known Future Data',
            data: normalizedKnownFuture,
            borderColor: 'rgba(0, 128, 0, 1)', // Green color for known future data
            borderWidth: 2,
            pointRadius: 0,
            fill: false
        });
    }
    
    // Add prediction lines (median and bounds)
    datasets.push({
        label: 'Median Prediction',
        data: normalizedMedian,
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
        pointRadius: 0,
        fill: false
    });
    
    datasets.push({
        label: 'Upper Bound (95%)',
        data: normalizedUpperBound,
        borderColor: 'rgba(75, 192, 192, 0.4)',
        borderWidth: 2,
        pointRadius: 0,
        borderDash: [5, 5],
        fill: false
    });
    
    datasets.push({
        label: 'Lower Bound (95%)',
        data: normalizedLowerBound,
        borderColor: 'rgba(75, 192, 192, 0.4)',
        borderWidth: 2,
        pointRadius: 0,
        borderDash: [5, 5],
        fill: false
    });
    
    // Create the chart
    monteCarloChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Monte Carlo Portfolio Mean Projections',
                    font: {
                        size: 18
                    },
                    padding: 20
                },
                tooltip: {
                    enabled: false, // Disable built-in tooltips, we're using custom ones
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: {
                            size: 14
                        },
                        padding: 20
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true,
                            modifierKey: 'ctrl'
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy'
                    },
                    limits: {
                        y: {
                            min: yMin,
                            max: yMax
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        font: {
                            size: 14
                        }
                    },
                    ticks: {
                        maxTicksLimit: 12,
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Relative Portfolio Value (End Date = 1.0)',
                        font: {
                            size: 14
                        }
                    },
                    min: yMin,
                    max: yMax,
                    ticks: {
                        font: {
                            size: 12
                        },
                        callback: function(value) {
                            // Format as relative value and percentage
                            if (value === 1) return '1.0 (0%)';
                            const percentChange = ((value - 1) * 100).toFixed(0);
                            const sign = percentChange >= 0 ? '+' : '';
                            return `${value.toFixed(2)} (${sign}${percentChange}%)`;
                        }
                    }
                }
            },
            elements: {
                line: {
                    tension: 0.2
                },
                point: {
                    radius: 0, // Hidden by default
                    hoverRadius: 5 // Show on hover
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: false,
                axis: 'x',
                includeInvisible: false
            },
            events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
            onHover: function(evt, activeElements) {
                // For user experience, change cursor to pointer when over data points
                if (evt && evt.native) {
                    evt.native.target.style.cursor = activeElements.length ? 'pointer' : 'default';
                }
            }
        }
    });
    
    // Also normalize the statistics for display
    const statsDiv = document.createElement('div');
    statsDiv.className = 'mc-statistics';
    statsDiv.innerHTML = `
        <h4>Simulation Statistics</h4>
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-label">Expected Final Value:</span>
                <span class="stat-value">${formatCurrency(data.final_mean_value)} (${((data.final_mean_value / normalizeValue - 1) * 100).toFixed(2)}%)</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">95% Confidence Range:</span>
                <span class="stat-value">${formatCurrency(data.final_lower_bound)} - ${formatCurrency(data.final_upper_bound)}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">VaR (95%):</span>
                <span class="stat-value">${formatCurrency(Math.abs(data.var_95))}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Probability of Gain:</span>
                <span class="stat-value">${(data.probability_of_gain * 100).toFixed(2)}%</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Max Potential Gain:</span>
                <span class="stat-value">${formatCurrency(data.max_gain)} (${((data.max_gain / normalizeValue) * 100).toFixed(2)}%)</span>
            </div>
        </div>
    `;
    
    document.querySelector('#mcResults').appendChild(statsDiv);
}

function displayMultiPortfolioMonteCarloResults(simulationResults) {
    if (!simulationResults || simulationResults.length === 0) {
        const resultsDiv = document.getElementById('mcResults');
        resultsDiv.innerHTML = '<h3>Error</h3><p>No valid portfolio simulations were generated</p>';
        return;
    }

    console.log('Displaying Monte Carlo results:', simulationResults.length, 'portfolios');

    const resultsDiv = document.getElementById('mcResults');
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = `
        <h3>Monte Carlo Portfolio Mean Projections</h3>
        <div class="chart-container" style="position: relative;">
            <canvas id="monteCarloChart"></canvas>
            <div id="mc-tooltip" style="position: absolute; display: none; background-color: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px; pointer-events: none; z-index: 10000; width: auto; min-width: 150px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);"></div>
        </div>
        <div class="chart-actions">
            <button class="reset-zoom-btn" onclick="resetMonteCarloZoom()">Reset Zoom</button>
        </div>
        <div id="simulationStats"></div>
    `;
    
    // Clean up any existing chart
    if (monteCarloChart) {
        monteCarloChart.destroy();
    }
    
    // Create Monte Carlo chart
    const ctx = document.getElementById('monteCarloChart').getContext('2d');
    
    // Use the dates from the first simulation for all (they should be the same)
    const firstSim = simulationResults[0].mcData;
    const dates = firstSim.dates;
    
    // Find the historical end date value (should be the same for all simulations)
    // We'll use this for normalization
    const normalizeValue = firstSim.historical_path[firstSim.historical_path.length - 1];
    
    // Create datasets for the chart
    const datasets = [];
    
    // Store portfolio data for cmd+click details display
    const portfolioData = {};
    
    // Create color palette for different portfolios if their colors aren't provided
    const portfolioColors = [
        'rgba(0, 0, 255, 1)',      // Blue
        'rgba(255, 165, 0, 1)',     // Orange
        'rgba(0, 128, 0, 1)',       // Green
        'rgba(255, 0, 0, 1)',       // Red
        'rgba(128, 0, 128, 1)',     // Purple
        'rgba(165, 42, 42, 1)',     // Brown
        'rgba(0, 0, 0, 1)'          // Black
    ];
    
    // Find max and min values across all simulations for y-axis scaling
    let allValues = [];
    
    // Process each portfolio simulation
    simulationResults.forEach((result, portfolioIndex) => {
        const portfolioName = result.name;
        const mcData = result.mcData;
        const portfolioColor = result.color || portfolioColors[portfolioIndex % portfolioColors.length];
        
        // Store portfolio data for detail display
        portfolioData[portfolioName] = {
            name: portfolioName,
            color: portfolioColor,
            optimizationData: {
                // Use the original optimizationData from the portfolio
                original_return: result.optimizationData?.metrics?.['Current Return'] / 100 || 0,
                original_volatility: result.optimizationData?.metrics?.['Current Volatility'] / 100 || 0,
                original_sharpe: result.optimizationData?.metrics?.['Current Sharpe'] || 0,
                return: result.optimizationData?.metrics?.['Optimized Return'] / 100 || 0,
                volatility: result.optimizationData?.metrics?.['Optimized Volatility'] / 100 || 0,
                sharpe: result.optimizationData?.metrics?.['Optimized Sharpe'] || 0,
                allocation: result.optimizationData?.allocation || {}
            }
        };
        
        // For this portfolio, normalize all series based on the value at historical end date
        const normalizedMean = mcData.mean_path.map(val => val / normalizeValue);
        const normalizedUpperBound = mcData.upper_bound.map(val => val / normalizeValue);
        const normalizedLowerBound = mcData.lower_bound.map(val => val / normalizeValue);
        const normalizedHistorical = mcData.historical_path.map(val => val / normalizeValue);
        
        // Collect all values for axis scaling
        allValues = allValues.concat(
            normalizedMean,
            normalizedUpperBound,
            normalizedLowerBound,
            normalizedHistorical
        ).filter(val => !isNaN(val));
        
        // Make the color lighter for the bounds
        const lighterColor = portfolioColor.replace('1)', '0.4)');
        
        // Add the mean line for this portfolio (combines historical and prediction)
        datasets.push({
            label: `${portfolioName}`,
            data: normalizedMean,
            borderColor: portfolioColor,
            borderWidth: 2,
            pointRadius: 0,
            // Use a custom styling function to make only future portions dashed
            segment: {
                borderDash: ctx => {
                    // If we're past the length of the historical data, use dashed line
                    // Otherwise use solid line
                    return ctx.p0.parsed.x >= normalizedHistorical.length - 1 ? [5, 5] : []
                }
            },
            fill: false
        });
        
        // Add confidence interval bounds (with proper grouping for legend)
        datasets.push({
            label: `${portfolioName} - 95% CI`,
            data: normalizedUpperBound,  // Use upper bound data for the legend item
            borderColor: lighterColor,
            borderWidth: 1.5,
            pointRadius: 0,
            borderDash: [5, 5],
            fill: false,
            hidden: true  // Hidden by default
        });
        
        // Add lower bound (with different label but same styling)
        datasets.push({
            label: `${portfolioName} - 95% CI (Lower)`,  // This won't appear in legend due to filter
            data: normalizedLowerBound,
            borderColor: lighterColor,
            borderWidth: 1.5,
            pointRadius: 0,
            borderDash: [5, 5],
            fill: false,
            hidden: true  // Hidden by default
        });
    });
    
    // Calculate y-axis scale with padding
    const maxValue = Math.max(...allValues);
    const minValue = Math.min(...allValues);
    const dataRange = maxValue - minValue;
    const padding = dataRange * 0.1;
    const yMax = Math.round((maxValue + padding) * 100) / 100;
    const yMin = Math.round((minValue - padding) * 100) / 100;
    
    // Create the chart
    monteCarloChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Monte Carlo Portfolio Mean Projections',
                    font: {
                        size: 18
                    },
                    padding: 20
                },
                tooltip: {
                    enabled: false, // Disable built-in tooltips, we're using custom ones
                },
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        filter: function(item) {
                            // Don't show the lower bound CI items in legend 
                            return !item.text.includes('CI (Lower)');
                        },
                        font: {
                            size: 12
                        },
                        padding: 10
                    },
                    onClick: function(e, legendItem, legend) {
                        const index = legendItem.datasetIndex;
                        const chart = legend.chart;
                        const meta = chart.getDatasetMeta(index);
                        
                        // Check if we're clicking on a CI item (which is always the upper bound)
                        if (legendItem.text.includes('95% CI')) {
                            // We're clicking a CI item, toggle both upper and lower bounds
                            meta.hidden = !meta.hidden;
                            
                            // Find and toggle the corresponding lower bound
                            // The lower bound is always the next dataset after the upper bound
                            const lowerBoundIndex = index + 1;
                            const lowerBoundMeta = chart.getDatasetMeta(lowerBoundIndex);
                            lowerBoundMeta.hidden = meta.hidden;
                        } else {
                            // Check if Command/Ctrl key is pressed (show portfolio details)
                            if (e.native && (e.native.metaKey || e.native.ctrlKey)) {
                                // Extract portfolio name (without the " - 95% CI" if present)
                                const portfolioName = legendItem.text.split(' - ')[0];
                                
                                if (portfolioData[portfolioName]) {
                                    // Use the existing showPortfolioDetails function
                                    const portfolio = portfolioData[portfolioName];
                                    
                                    // Debug portfolio data structure
                                    console.log('Portfolio data for modal:', portfolio);
                                    
                                    // Format the portfolio data to match what showPortfolioDetails expects
                                    const optimizationData = {
                                        metrics: {
                                            'Current Return': portfolio.optimizationData.original_return ? portfolio.optimizationData.original_return * 100 : 0,
                                            'Current Volatility': portfolio.optimizationData.original_volatility ? portfolio.optimizationData.original_volatility * 100 : 0,
                                            'Current Sharpe': portfolio.optimizationData.original_sharpe || 0,
                                            'Optimized Return': portfolio.optimizationData.return ? portfolio.optimizationData.return * 100 : 0,
                                            'Optimized Volatility': portfolio.optimizationData.volatility ? portfolio.optimizationData.volatility * 100 : 0,
                                            'Optimized Sharpe': portfolio.optimizationData.sharpe || 0
                                        },
                                        allocation: portfolio.optimizationData.allocation || {}
                                    };
                                    
                                    // Call showPortfolioDetails function directly
                                    // Set z-index to ensure portfolio details modal is on top
                                    document.getElementById('portfolioDetailsModal').style.zIndex = 1100;
                                    document.getElementById('monteCarloModal').style.zIndex = 1000;
                                    
                                    if (typeof showPortfolioDetails === 'function') {
                                        showPortfolioDetails(optimizationData, portfolioName);
                                    } else {
                                        // Fallback - display information directly (simpler version)
                                        const modal = document.getElementById('portfolioDetailsModal');
                                        if (modal) {
                                            const modalContent = modal.querySelector('.modal-content');
                                            if (modalContent) {
                                                modalContent.innerHTML = `
                                                    <div class="optimization-summary">
                                                        <h3>${portfolioName}</h3>
                                                        <p>Portfolio details not available. Please check console for data structure.</p>
                                                    </div>
                                                `;
                                                modal.style.display = 'flex';
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Regular dataset, use default behavior
                                meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;
                            }
                        }
                        
                        chart.update();
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true,
                            modifierKey: 'ctrl'
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'xy'
                    },
                    limits: {
                        y: {
                            min: yMin,
                            max: yMax
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        font: {
                            size: 14
                        }
                    },
                    ticks: {
                        maxTicksLimit: 12,
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Relative Portfolio Value (End Date = 1.0)',
                        font: {
                            size: 14
                        }
                    },
                    min: yMin,
                    max: yMax,
                    ticks: {
                        font: {
                            size: 12
                        },
                        callback: function(value) {
                            // Format as relative value and percentage
                            if (value === 1) return '1.0 (0%)';
                            const percentChange = ((value - 1) * 100).toFixed(0);
                            const sign = percentChange >= 0 ? '+' : '';
                            return `${value.toFixed(2)} (${sign}${percentChange}%)`;
                        }
                    }
                }
            },
            elements: {
                line: {
                    tension: 0.2
                },
                point: {
                    radius: 0, // Hidden by default
                    hoverRadius: 5 // Show on hover
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: false,
                axis: 'x',
                includeInvisible: false
            },
            events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
            onHover: function(evt, activeElements) {
                // For user experience, change cursor to pointer when over data points
                if (evt && evt.native) {
                    evt.native.target.style.cursor = activeElements.length ? 'pointer' : 'default';
                }
            }
        }
    });
    
    // Create statistics tables for each portfolio
    const statsDiv = document.getElementById('simulationStats');
    statsDiv.innerHTML = '<h4>Simulation Statistics</h4>';
    
    // Create table with comparisons
    const statsTable = document.createElement('table');
    statsTable.className = 'simulation-stats-table';
    
    // Table header
    let headerRow = document.createElement('thead');
    headerRow.innerHTML = `
        <tr>
            <th>Portfolio</th>
            <th>Expected Final Value <span class="info-icon" title="The mean (average) expected portfolio value at the end of the simulation period. The dashed lines in the chart represent these mean projections.">&#9432;</span></th>
            <th>95% Confidence Range <span class="info-icon" title="The range within which the final portfolio value is expected to fall with 95% probability, based on the simulation results.">&#9432;</span></th>
            <th>VaR (95%) <span class="info-icon" title="Value at Risk (95%): The maximum expected loss at a 95% confidence level. This represents the worst-case scenario with 95% confidence.">&#9432;</span></th>
            <th>Probability of Gain <span class="info-icon" title="The percentage of simulations where the final portfolio value was higher than the initial investment. Indicates likelihood of positive returns.">&#9432;</span></th>
            <th>Max Potential Gain <span class="info-icon" title="The highest possible portfolio value projected across all simulations. Represents the best-case scenario.">&#9432;</span></th>
        </tr>
    `;
    statsTable.appendChild(headerRow);
    
    // Add rows for each portfolio
    simulationResults.forEach(result => {
        const portfolioName = result.name;
        const mcData = result.mcData;
        const portfolioColor = result.color;
        
        const row = document.createElement('tr');
        row.style.borderLeft = `4px solid ${portfolioColor}`;
        
        row.innerHTML = `
            <td>${portfolioName}</td>
            <td>${formatCurrency(mcData.final_mean_value)} (${((mcData.final_mean_value / normalizeValue - 1) * 100).toFixed(2)}%)</td>
            <td>${formatCurrency(mcData.final_lower_bound)} - ${formatCurrency(mcData.final_upper_bound)}</td>
            <td>${formatCurrency(Math.abs(mcData.var_95))}</td>
            <td>${(mcData.probability_of_gain * 100).toFixed(2)}%</td>
            <td>${formatCurrency(mcData.max_gain)} (${((mcData.max_gain / normalizeValue) * 100).toFixed(2)}%)</td>
        `;
        
        statsTable.appendChild(row);
    });
    
    statsDiv.appendChild(statsTable);

    // Add a direct event handler to the chart canvas as a backup tooltip method
    const chartCanvas = document.getElementById('monteCarloChart');
    chartCanvas.addEventListener('mousemove', function(e) {
        // Get tooltip element
        const tooltip = document.getElementById('mc-tooltip');
        if (!tooltip) return;
        
        // Get bounding rectangle for position calculations
        const rect = chartCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Only show tooltip if we're inside the chart area
        if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
            // Get nearest data points
            const points = monteCarloChart.getElementsAtEventForMode(e, 'nearest', { intersect: false }, true);
            
            if (points.length > 0) {
                // Get the data point
                const datasetIndex = points[0].datasetIndex;
                const dataIndex = points[0].index;
                
                // Find which portfolio this belongs to
                const portfolioIndex = Math.floor(datasetIndex / 3);
                const result = simulationResults[portfolioIndex];
                
                // Make sure we have data
                if (result && result.mcData && result.mcData.mean_path) {
                    // Build tooltip content
                    const date = dates[dataIndex];
                    const value = result.mcData.mean_path[dataIndex] / normalizeValue;
                    const percentChange = ((value - 1) * 100).toFixed(2);
                    const sign = percentChange >= 0 ? '+' : '';
                    
                    tooltip.innerHTML = `
                        <div style="font-weight: bold; margin-bottom: 8px;">${date}</div>
                        <div>
                            <span style="color: ${result.color}; font-weight: bold;">${result.name}:</span>
                            <span>${value.toFixed(2)} (${sign}${percentChange}%)</span>
                        </div>
                    `;
                    
                    // Position tooltip
                    tooltip.style.display = 'block';
                    tooltip.style.left = (x + 15) + 'px';
                    tooltip.style.top = (y - 20) + 'px';
                }
            }
        }
    });
    
    // Hide tooltip on mouseleave
    chartCanvas.addEventListener('mouseleave', function() {
        const tooltip = document.getElementById('mc-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    });

    // Implementation of custom tooltips with Chart.js API
    const tooltip = document.getElementById('mc-tooltip');
    const chartElement = document.getElementById('monteCarloChart');
    
    // Only add the listener if both elements exist
    if (chartElement && tooltip) {
        chartElement.addEventListener('mousemove', function(event) {
            // Get mouse position relative to chart
            const rect = chartElement.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Get the data point at mouse position
            const activePoints = monteCarloChart.getElementsAtEventForMode(
                event,
                'index',
                { intersect: false },
                false
            );
            
            if (activePoints.length > 0) {
                // Show tooltip
                tooltip.style.display = 'block';
                
                // Get data point index
                const dataIndex = activePoints[0].index;
                const date = dates[dataIndex];
                
                // Generate tooltip content
                let content = `<div style="font-weight: bold; margin-bottom: 8px;">${date}</div>`;
                
                // Add data for each visible portfolio
                simulationResults.forEach((result, i) => {
                    // Check if this dataset is visible
                    const datasetIndex = i * 3; // Each portfolio has 3 datasets (mean, upper CI, lower CI)
                    const meta = monteCarloChart.getDatasetMeta(datasetIndex);
                    
                    if (!meta.hidden) {
                        const name = result.name;
                        const value = result.mcData.mean_path[dataIndex] / normalizeValue;
                        const percentChange = ((value - 1) * 100).toFixed(2);
                        const sign = percentChange >= 0 ? '+' : '';
                        const color = result.color || portfolioColors[i % portfolioColors.length];
                        
                        content += `
                            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                <span style="color: ${color}; font-weight: bold; margin-right: 12px;">${name}:</span>
                                <span>${value.toFixed(2)} (${sign}${percentChange}%)</span>
                            </div>
                        `;
                    }
                });
                
                // Update tooltip content and position
                tooltip.innerHTML = content;
                
                // Position the tooltip
                const tooltipWidth = tooltip.offsetWidth;
                const tooltipHeight = tooltip.offsetHeight;
                
                // Adjust position to keep tooltip within chart
                let tooltipX = x + 10; // 10px offset from cursor
                let tooltipY = y + 10;
                
                // Make sure tooltip doesn't go off right edge
                if (tooltipX + tooltipWidth > rect.width) {
                    tooltipX = x - tooltipWidth - 10;
                }
                
                // Make sure tooltip doesn't go off bottom edge
                if (tooltipY + tooltipHeight > rect.height) {
                    tooltipY = y - tooltipHeight - 10;
                }
                
                // Set position
                tooltip.style.left = `${tooltipX}px`;
                tooltip.style.top = `${tooltipY}px`;
            } else {
                // Hide tooltip when not over a data point
                tooltip.style.display = 'none';
            }
        });
        
        // Hide tooltip when mouse leaves chart
        chartElement.addEventListener('mouseleave', function() {
            tooltip.style.display = 'none';
        });
    }
}

// Helper function to clear the current portfolio
function clearPortfolio() {
    const portfolioItems = document.getElementById('portfolioItems');
    portfolioItems.innerHTML = '';
}

// Helper function to add a portfolio item programmatically
async function addPortfolioItem(ticker, amount) {
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
        }
    } catch (error) {
        console.error('Portfolio add error:', error);
    }
}

// Helper function to format date for input fields
function formatDateForInput(date) {
    return date.toISOString().split('T')[0];
}

// Helper function to format currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
} 