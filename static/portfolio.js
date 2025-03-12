// Apply CSS styles for custom chart legend and tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Create a style element
    const styleElement = document.createElement('style');
    
    // Add the CSS rules
    styleElement.textContent = `
        .chart-legend-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 8px;
            margin: 20px 0;
        }
        
        .legend-pill {
            display: flex;
            align-items: center;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 20px;
            padding: 5px 12px;
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.2s, box-shadow 0.2s;
            user-select: none;
        }
        
        .legend-pill:hover {
            background-color: rgba(0, 0, 0, 0.1);
        }
        
        .legend-pill.active {
            background-color: rgba(24, 26, 27, 0.95);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            color: white;
        }
        
        .legend-pill-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .legend-pill-ci {
            position: relative;
        }
        
        .legend-ci-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 8px;
            display: none;
            z-index: 1000;
            min-width: 150px;
        }
        
        .legend-ci-dropdown.show {
            display: block;
        }
        
        .ci-item {
            display: flex;
            align-items: center;
            padding: 6px 8px;
            border-radius: 4px;
            margin-bottom: 4px;
            cursor: pointer;
        }
        
        .ci-item:last-child {
            margin-bottom: 0;
        }
        
        .ci-item:hover {
            background-color: rgba(0, 0, 0, 0.05);
        }
        
        .ci-item.active {
            background-color: rgba(0, 0, 0, 0.1);
        }
        
        .ci-item-color {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .dropdown-indicator {
            margin-left: 5px;
            font-size: 10px;
            transition: transform 0.2s;
        }
        
        .legend-pill.open .dropdown-indicator {
            transform: rotate(180deg);
        }
    `;
    
    // Add the style element to the document head
    document.head.appendChild(styleElement);
});

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
    const modal = document.getElementById('monteCarloModal');
    modal.style.display = 'flex';
    modal.classList.add('active');
    
    // Set default dates
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 90);
    
    // Apply Apple-style layout if not already applied
    applyMonteCarloAppleStyleLayout();
    
    // Get the existing inputs after applying the layout
    const startDateEl = document.getElementById('mcStartDate');
    const endDateEl = document.getElementById('mcEndDate');
    
    // Set their values
    if (startDateEl) startDateEl.value = formatDateForInput(startDate);
    if (endDateEl) endDateEl.value = formatDateForInput(today);
    
    // Hide results section until analysis is run
    const resultsEl = document.getElementById('mcResults');
    if (resultsEl) resultsEl.style.display = 'none';
}

function closeMonteCarloModal() {
    const modal = document.getElementById('monteCarloModal');
    modal.style.display = 'none';
    modal.classList.remove('active');
    
    // Clean up the chart when closing the modal
    if (monteCarloChart) {
        monteCarloChart.destroy();
        monteCarloChart = null;
    }
}

function resetMonteCarloZoom() {
    if (monteCarloChart) {
        console.log('Resetting Monte Carlo chart zoom');
        monteCarloChart.resetZoom();
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
    // Make sure the zoom plugin is recognized
    if (!Chart.registry.getPlugin('zoom')) {
        console.warn('Zoom plugin not found in Chart.js registry. Adding it now.');
        // Try to manually register the zoom plugin if available in window
        if (window.ChartZoom) {
            Chart.register(window.ChartZoom);
            console.log('Registered zoom plugin manually.');
        }
    }
    
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
                    display: false,
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
                            enabled: true, // Enable wheel zoom with modifier key
                            modifierKey: 'meta', // Use meta key (Command on Mac)
                            speed: 0.1 // Add speed for smoother zooming
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
                <span class="stat-value">${formatCurrency(data.max_gain)} (${(((data.max_gain + normalizeValue) / normalizeValue - 1) * 100).toFixed(2)}%)</span>
            </div>
        </div>
    `;
    
    document.querySelector('#mcResults').appendChild(statsDiv);

    // Add debug logging for the zoom plugin
    console.log('Monte Carlo chart created with zoom plugin:', 
                monteCarloChart.options.plugins.zoom,
                'Chart.js plugins:', Chart.registry.plugins);
    
    // After creating monteCarloChart
    
    // Ensure wheel events are properly handled for the chart canvas
    const mcCanvas = document.getElementById('monteCarloChart');
    if (mcCanvas) {
        mcCanvas.addEventListener('wheel', function(e) {
            // Only enable zoom when meta key (Command on Mac) is pressed
            if (e.metaKey) {
                // Allow chart zoom plugin to handle the event
                console.log('Command+wheel detected on Monte Carlo chart');
            } else {
                // Otherwise allow normal scrolling
                e.stopPropagation();
            }
        }, { passive: false });
    }
    
    // Create and add simulation statistics
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
        <h3 style="text-align: center; margin-top: 20px;">Monte Carlo Portfolio Mean Projections</h3>
        <div id="custom-legend" class="chart-legend-pills"></div>
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
                    display: false,
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
                    display: false, // Disable built-in legend, we're using custom one
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: {
                            enabled: true, // Enable wheel zoom when used with modifier key
                            modifierKey: 'meta', // Use meta key (Command on Mac)
                            speed: 0.1 // Add speed for smoother zooming
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
                y: {
                    title: {
                    display: true,
                        text: 'Portfolio Value (Normalized)',
                        font: {
                            size: 14
                        }
                    },
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
                    },
                    min: yMin,
                    max: yMax
                },
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
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
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
            events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
            onHover: function(evt, activeElements) {
                // For user experience, change cursor to pointer when over data points
                if (evt && evt.native) {
                    evt.native.target.style.cursor = activeElements.length ? 'pointer' : 'default';
                }
            }
        }
    });
    
    // Create custom legend
    createCustomMCLegend(monteCarloChart, simulationResults, portfolioData);
    
    // Create statistics tables for each portfolio
    createSimulationStats(simulationResults, normalizeValue, dates);
    
    // Add tooltips
    setupMonteCarloTooltips(monteCarloChart, simulationResults, dates, normalizeValue);
    
    // Debug and ensure wheel events work properly
    console.log('Multi-portfolio Monte Carlo chart created with zoom plugin:', 
                monteCarloChart.options.plugins.zoom);
    
    // Ensure wheel events are properly handled
    const mcCanvas = document.getElementById('monteCarloChart');
    if (mcCanvas) {
        // Remove any existing listeners to avoid duplicates
        mcCanvas.removeEventListener('wheel', handleMonteCarloWheel);
        
        // Add wheel event listener
        mcCanvas.addEventListener('wheel', handleMonteCarloWheel, { passive: false });
        
        console.log('Added wheel event listener to Monte Carlo chart');
    }
    
    // Create simulation statistics
    createSimulationStats(simulationResults, normalizeValue, dates);
}

// Wheel event handler for Monte Carlo chart
function handleMonteCarloWheel(e) {
    // Only enable zoom when meta key (Command on Mac) is pressed
    if (e.metaKey) {
        // Let the event propagate to Chart.js zoom plugin
        console.log('Command+wheel on Monte Carlo chart - allowing zoom');
    } else {
        // For normal scrolling, prevent Chart.js from handling it
        // but allow the browser to scroll normally
        e.stopPropagation();
    }
}

// Helper function to create custom legend with pill-style buttons and CI dropdown
function createCustomMCLegend(chart, simulationResults, portfolioData) {
    const legendContainer = document.getElementById('custom-legend');
    if (!legendContainer) return;
    
    // Clear any existing legend
    legendContainer.innerHTML = '';
    
    // Group datasets by portfolio (excluding CI datasets which we'll handle separately)
    const portfolioGroups = {};
    const ciGroups = {};
    
    chart.data.datasets.forEach((dataset, index) => {
        const label = dataset.label;
        
        // Skip lower bound CI items
        if (label.includes('CI (Lower)')) return;
        
        if (label.includes('95% CI')) {
            // This is a CI item, store it separately
            const portfolioName = label.split(' - ')[0];
            if (!ciGroups[portfolioName]) {
                ciGroups[portfolioName] = [];
            }
            ciGroups[portfolioName].push({
                index: index,
                label: label,
                color: dataset.borderColor
            });
            
            // Make sure CI items start hidden by default
            chart.getDatasetMeta(index).hidden = true;
            
            // Also hide the corresponding lower bound
            chart.getDatasetMeta(index + 1).hidden = true;
        } else {
            // This is a main portfolio line
            portfolioGroups[label] = {
                index: index,
                color: dataset.borderColor
            };
        }
    });
    
    // Create a pill for each main portfolio
    Object.keys(portfolioGroups).forEach(portfolioName => {
        const { index, color } = portfolioGroups[portfolioName];
        const isHidden = chart.getDatasetMeta(index).hidden;
        
        // Create the pill element
        const pill = document.createElement('div');
        pill.className = `legend-pill ${isHidden ? '' : 'active'}`;
        pill.innerHTML = `
            <span class="legend-pill-color" style="background-color: ${color}"></span>
            <span>${portfolioName}</span>
        `;
        
        // Add click event to toggle visibility
        pill.addEventListener('click', (e) => {
            // Check if Command/Ctrl key is pressed (show portfolio details)
            if (e.metaKey || e.ctrlKey) {
                // Show portfolio details if available
                if (portfolioData[portfolioName]) {
                    // Extract detailed portfolio data
                    const portfolio = portfolioData[portfolioName];
                    
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
                    
                    // Ensure the Monte Carlo modal is behind the portfolio details modal
                    document.getElementById('portfolioDetailsModal').style.zIndex = 1100;
                    document.getElementById('monteCarloModal').style.zIndex = 1000;
                    
                    // Show the portfolio details
                    showPortfolioDetails(optimizationData, portfolioName);
                }
            } else {
                // Regular click - toggle visibility
                const meta = chart.getDatasetMeta(index);
                meta.hidden = !meta.hidden;
                
                // Toggle active class
                pill.classList.toggle('active');
                
                // Update chart
                chart.update();
            }
        });
        
        // Add pill to container
        legendContainer.appendChild(pill);
    });
    
    // Create a special CI pill with dropdown
    if (Object.keys(ciGroups).length > 0) {
        // Create the CI pill
        const ciPill = document.createElement('div');
        ciPill.className = 'legend-pill legend-pill-ci';
        ciPill.style.position = 'relative'; // Make ciPill a positioning container
        ciPill.innerHTML = `
            <span>CI</span>
            <span class="dropdown-indicator">&#9207;</span>
        `;
        
        // Create dropdown container
        const dropdown = document.createElement('div');
        dropdown.className = 'legend-ci-dropdown';
        dropdown.style.zIndex = 9999; // Keep high z-index
        // Ensure dropdown positioning
        dropdown.style.position = 'absolute';
        dropdown.style.top = '100%';
        dropdown.style.left = '0';
        dropdown.style.marginTop = '6px';
        
        // Add dropdown items for each CI
        Object.keys(ciGroups).forEach(portfolioName => {
            ciGroups[portfolioName].forEach(ci => {
                const meta = chart.getDatasetMeta(ci.index);
                const isActive = !meta.hidden;
                
                const item = document.createElement('div');
                item.className = `ci-item ${isActive ? 'active' : ''}`;
                item.dataset.index = ci.index;
                
                // Find lower bound dataset (next index)
                const lowerBoundIndex = ci.index + 1;
                
                item.innerHTML = `
                    <span class="ci-item-color" style="background-color: ${ci.color}"></span>
                    <span>${portfolioName} CI</span>
                `;
                
                // Add click event to toggle visibility
                item.addEventListener('click', (e) => {
                    e.stopPropagation(); // Prevent closing dropdown
                    e.preventDefault(); // Prevent any default behavior
                    
                    // Toggle upper bound
                    const metaUpper = chart.getDatasetMeta(ci.index);
                    metaUpper.hidden = !metaUpper.hidden;
                    
                    // Toggle lower bound
                    const metaLower = chart.getDatasetMeta(lowerBoundIndex);
                    metaLower.hidden = metaUpper.hidden;
                    
                    // Toggle active class
                    item.classList.toggle('active');
                    
                    // Update chart
                    chart.update();
                });
                
                dropdown.appendChild(item);
            });
        });
        
        // Add dropdown to pill
        ciPill.appendChild(dropdown);
        
        // Toggle dropdown on click
        ciPill.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent document click from immediately closing dropdown
            dropdown.classList.toggle('show');
            ciPill.classList.toggle('open');
            
            // When showing dropdown, move it to the document body for better interaction
            if (dropdown.classList.contains('show')) {
                // Calculate position relative to viewport
                const rect = ciPill.getBoundingClientRect();
                
                // Preserve the dropdown's position relative to the CI button
                const dropdownClone = dropdown.cloneNode(true);
                
                // Set fixed position for the clone
                dropdownClone.style.position = 'fixed';
                dropdownClone.style.top = rect.bottom + 'px';
                dropdownClone.style.left = rect.left + 'px';
                dropdownClone.style.marginTop = '6px';
                
                // Copy all event listeners
                Array.from(dropdown.querySelectorAll('.ci-item')).forEach((item, index) => {
                    const newItem = dropdownClone.querySelectorAll('.ci-item')[index];
                    newItem.addEventListener('click', (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        
                        // Get dataset index
                        const datasetIndex = parseInt(item.dataset.index, 10);
                        const lowerBoundIndex = datasetIndex + 1;
                        
                        // Toggle upper bound
                        const metaUpper = chart.getDatasetMeta(datasetIndex);
                        metaUpper.hidden = !metaUpper.hidden;
                        
                        // Toggle lower bound
                        const metaLower = chart.getDatasetMeta(lowerBoundIndex);
                        metaLower.hidden = metaUpper.hidden;
                        
                        // Toggle active class on both the original and cloned items
                        item.classList.toggle('active');
                        newItem.classList.toggle('active');
                        
                        // Update chart
                        chart.update();
                    });
                });
                
                // Store a reference to the original dropdown on the clone
                dropdownClone.dataset.originalDropdownId = 'ciDropdown';
                dropdown.id = 'ciDropdown';
                
                // Hide the original dropdown
                dropdown.style.display = 'none';
                
                // Add the clone to the document body
                document.body.appendChild(dropdownClone);
                
                // Add scroll event listener to update dropdown position
                const updateDropdownPosition = () => {
                    const updatedRect = ciPill.getBoundingClientRect();
                    dropdownClone.style.top = updatedRect.bottom + 'px';
                    dropdownClone.style.left = updatedRect.left + 'px';
                };
                
                // Store the listener so we can remove it later
                window.ciDropdownScrollHandler = updateDropdownPosition;
                
                // Add scroll event listener
                window.addEventListener('scroll', window.ciDropdownScrollHandler, true);
            } else {
                // When hiding, remove any clones and show the original
                const dropdownClone = document.querySelector('.legend-ci-dropdown[data-original-dropdown-id="ciDropdown"]');
                if (dropdownClone) {
                    dropdownClone.remove();
                }
                
                // Remove scroll event listener
                if (window.ciDropdownScrollHandler) {
                    window.removeEventListener('scroll', window.ciDropdownScrollHandler, true);
                    window.ciDropdownScrollHandler = null;
                }
                
                dropdown.style.display = '';
            }
        });
        
        // Remove any existing click handler to avoid duplicates
        document.removeEventListener('click', handleClickOutsideDropdown);
        
        // Define the click handler as a named function so we can remove it later
        function handleClickOutsideDropdown(e) {
            if (!ciPill.contains(e.target)) {
                // Hide both the original dropdown and any clones
                dropdown.classList.remove('show');
                ciPill.classList.remove('open');
                
                const dropdownClone = document.querySelector('.legend-ci-dropdown[data-original-dropdown-id="ciDropdown"]');
                if (dropdownClone) {
                    dropdownClone.remove();
                }
                
                // Remove scroll event listener
                if (window.ciDropdownScrollHandler) {
                    window.removeEventListener('scroll', window.ciDropdownScrollHandler, true);
                    window.ciDropdownScrollHandler = null;
                }
                
                dropdown.style.display = '';
            }
        }
        
        // Close dropdown when clicking outside
        document.addEventListener('click', handleClickOutsideDropdown);
        
        // Add CI pill to container
        legendContainer.appendChild(ciPill);
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

// Enhance Evaluate Portfolio Modal UI
function evaluatePortfolio() {
    const modal = document.getElementById('evaluateModal');
    modal.style.display = 'flex';
    modal.classList.add('active');
    
    // Set default dates if not already set
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 90);
    
    // Apply Apple-style layout if not already applied
    applyEvaluateAppleStyleLayout();
    
    // Hide results section until analysis is run
    document.getElementById('evaluationResults').style.display = 'none';
    document.getElementById('initialButtons').style.display = 'flex';
    document.getElementById('resultButtons').style.display = 'none';
}

function applyEvaluateAppleStyleLayout() {
    // Check if already styled
    if (document.querySelector('#evaluateModal .evaluate-apple-style')) {
        return;
    }
    
    // Get the modal content
    const modalContent = document.querySelector('#evaluateModal .modal-content');
    
    // Remove any existing input container but keep results div
    const evaluationResults = document.getElementById('evaluationResults');
    const existingInputs = document.querySelector('#evaluateModal .evaluate-inputs');
    if (existingInputs) {
        existingInputs.remove();
    }
    
    // Create a clean info banner
    const infoBanner = document.createElement('div');
    infoBanner.className = 'info-banner';
    infoBanner.innerHTML = `
        <p class="evaluation-description">Select a date range to evaluate your portfolio's historical performance and risk metrics.</p>
    `;
    
    // Create a clean form container
    const formContainer = document.createElement('div');
    formContainer.className = 'apple-mc-container';
    
    // Create date inputs row
    const dateRow = document.createElement('div');
    dateRow.className = 'apple-mc-row';
    
    // Start Date
    const startDateContainer = document.createElement('div');
    startDateContainer.className = 'apple-mc-column';
    const startDateLabel = document.createElement('label');
    startDateLabel.setAttribute('for', 'startDate');
    startDateLabel.textContent = 'Start Date';
    const startDateInput = document.createElement('input');
    startDateInput.id = 'startDate';
    startDateInput.className = 'apple-input';
    startDateInput.type = 'date';
    startDateInput.required = true;
    
    startDateContainer.appendChild(startDateLabel);
    startDateContainer.appendChild(startDateInput);
    dateRow.appendChild(startDateContainer);
    
    // End Date
    const endDateContainer = document.createElement('div');
    endDateContainer.className = 'apple-mc-column';
    const endDateLabel = document.createElement('label');
    endDateLabel.setAttribute('for', 'endDate');
    endDateLabel.textContent = 'End Date';
    const endDateInput = document.createElement('input');
    endDateInput.id = 'endDate';
    endDateInput.className = 'apple-input';
    endDateInput.type = 'date';
    endDateInput.required = true;
    
    endDateContainer.appendChild(endDateLabel);
    endDateContainer.appendChild(endDateInput);
    dateRow.appendChild(endDateContainer);
    
    // Add row to form container
    formContainer.appendChild(dateRow);
    
    // Add the Apple style marker
    const styleMarker = document.createElement('div');
    styleMarker.className = 'evaluate-apple-style';
    styleMarker.style.display = 'none';
    
    // Add all elements to modal content
    modalContent.insertBefore(infoBanner, modalContent.firstChild);
    modalContent.insertBefore(formContainer, evaluationResults);
    modalContent.appendChild(styleMarker);
    
    // Set initial dates
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 90);
    
    startDateInput.value = formatDateForInput(startDate);
    endDateInput.value = formatDateForInput(today);
}

function closeEvaluateModal() {
    const modal = document.getElementById('evaluateModal');
    modal.style.display = 'none';
    modal.classList.remove('active');
    
    document.getElementById('evaluationResults').style.display = 'none';
    document.getElementById('initialButtons').style.display = 'flex';
    document.getElementById('resultButtons').style.display = 'none';
}

// Enhance Backtest Performance Modal UI
function showFuturePerformance() {
    const modal = document.getElementById('futureModal');
    modal.style.display = 'flex';
    modal.classList.add('active');
    
    // Update modal title to "Backtesting"
    const modalTitle = document.querySelector('#futureModal h2');
    if (modalTitle) {
        modalTitle.textContent = "Backtesting";
        modalTitle.className = "backtest-title";
    }
    
    // Apply Apple-style layout if not already applied
    applyFutureAppleStyleLayout();
    
    // Hide results section until analysis is run
    document.getElementById('futureResults').style.display = 'none';
    
    // Clean up any existing chart
    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }
}

function applyFutureAppleStyleLayout() {
    // Check if already styled
    if (document.querySelector('#futureModal .future-apple-style')) {
        return;
    }
    
    // Get the modal content
    const modalContent = document.querySelector('#futureModal .modal-content');
    
    // Remove any existing input container but keep results div
    const futureResults = document.getElementById('futureResults');
    const existingInputs = document.querySelector('#futureModal .future-inputs');
    if (existingInputs) {
        existingInputs.remove();
    }
    
    // Create a clean form container
    const formContainer = document.createElement('div');
    formContainer.className = 'apple-mc-container';
    
    // Create date inputs row
    const dateRow = document.createElement('div');
    dateRow.className = 'apple-mc-row';
    
    // Start Date
    const startDateContainer = document.createElement('div');
    startDateContainer.className = 'apple-mc-column';
    const startDateLabel = document.createElement('label');
    startDateLabel.setAttribute('for', 'futureStartDate');
    startDateLabel.textContent = 'Start Date';
    const startDateInput = document.createElement('input');
    startDateInput.id = 'futureStartDate';
    startDateInput.className = 'apple-input';
    startDateInput.type = 'date';
    startDateInput.required = true;
    
    startDateContainer.appendChild(startDateLabel);
    startDateContainer.appendChild(startDateInput);
    dateRow.appendChild(startDateContainer);
    
    // End Date
    const endDateContainer = document.createElement('div');
    endDateContainer.className = 'apple-mc-column';
    const endDateLabel = document.createElement('label');
    endDateLabel.setAttribute('for', 'futureEndDate');
    endDateLabel.textContent = 'End Date';
    const endDateInput = document.createElement('input');
    endDateInput.id = 'futureEndDate';
    endDateInput.className = 'apple-input';
    endDateInput.type = 'date';
    endDateInput.required = true;
    
    endDateContainer.appendChild(endDateLabel);
    endDateContainer.appendChild(endDateInput);
    dateRow.appendChild(endDateContainer);
    
    // Create checkbox row
    const checkboxRow = document.createElement('div');
    checkboxRow.className = 'apple-mc-row';
    
    // Checkbox container
    const checkboxContainer = document.createElement('div');
    checkboxContainer.className = 'apple-mc-column apple-checkbox-column checkbox-container';
    
    // Create the checkbox and label with proper positioning
    const checkboxLabel = document.createElement('label');
    checkboxLabel.className = 'apple-checkbox-label';
    checkboxLabel.setAttribute('for', 'includeBuildNew');
    
    // Create the label text first
    const labelText = document.createElement('span');
    labelText.textContent = 'Include Build New Portfolios';
    checkboxLabel.appendChild(labelText);
    
    // Then create and append the checkbox after the text
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = 'includeBuildNew';
    checkbox.className = 'apple-checkbox';
    checkboxLabel.appendChild(checkbox);
    
    checkboxContainer.appendChild(checkboxLabel);
    checkboxRow.appendChild(checkboxContainer);
    
    // Add rows to form container
    formContainer.appendChild(dateRow);
    formContainer.appendChild(checkboxRow);
    
    // Add the Apple style marker
    const styleMarker = document.createElement('div');
    styleMarker.className = 'future-apple-style';
    styleMarker.style.display = 'none';
    
    // Add all elements to modal content
    modalContent.insertBefore(formContainer, futureResults);
    modalContent.appendChild(styleMarker);
    
    // Remove the default date values - let users fill them manually
}

function closeFutureModal() {
    const modal = document.getElementById('futureModal');
    modal.style.display = 'none';
    modal.classList.remove('active');
    
    if (performanceChart) {
        performanceChart.destroy();
        performanceChart = null;
    }
}

// Function to apply Apple-style layout to Monte Carlo modal
function applyMonteCarloAppleStyleLayout() {
    // Check if already styled
    if (document.querySelector('#monteCarloModal .mc-apple-style')) {
        return;
    }
    
    // Get the modal content
    const modalContent = document.querySelector('#monteCarloModal .modal-content');
    
    // Get the results div to preserve
    const mcResults = document.getElementById('mcResults');
    
    // Remove any existing input container
    const existingInputs = document.querySelector('#monteCarloModal .mc-inputs');
    if (existingInputs) {
        existingInputs.remove();
    }
    
    // Create a clean form container
    const formContainer = document.createElement('div');
    formContainer.className = 'mc-inputs apple-mc-container';
    
    // Create description
    const description = document.createElement('p');
    description.className = 'mc-description';
    description.textContent = 'Select a historical date range for analysis. The simulation will use this data to project performance from the end date through June 2025, including actual data where available.';
    
    // Create date inputs row
    const dateRow = document.createElement('div');
    dateRow.className = 'date-inputs';
    
    // Start Date
    const startDateContainer = document.createElement('div');
    const startDateLabel = document.createElement('label');
    startDateLabel.setAttribute('for', 'mcStartDate');
    startDateLabel.textContent = 'Historical Start Date:';
    const startDateInput = document.createElement('input');
    startDateInput.id = 'mcStartDate';
    startDateInput.type = 'date';
    startDateInput.required = true;
    
    startDateContainer.appendChild(startDateLabel);
    startDateContainer.appendChild(startDateInput);
    dateRow.appendChild(startDateContainer);
    
    // End Date
    const endDateContainer = document.createElement('div');
    const endDateLabel = document.createElement('label');
    endDateLabel.setAttribute('for', 'mcEndDate');
    endDateLabel.textContent = 'Historical End Date:';
    const endDateInput = document.createElement('input');
    endDateInput.id = 'mcEndDate';
    endDateInput.type = 'date';
    endDateInput.required = true;
    
    endDateContainer.appendChild(endDateLabel);
    endDateContainer.appendChild(endDateInput);
    dateRow.appendChild(endDateContainer);
    
    // Create a wrapper row for the options with fixed styling
    const optionsRow = document.createElement('div');
    optionsRow.className = 'mc-options';
    optionsRow.style.display = 'flex';
    optionsRow.style.justifyContent = 'space-between';
    optionsRow.style.gap = '20px';
    
    // Create another row specifically for the inputs
    const inputsRow = document.createElement('div');
    inputsRow.style.display = 'flex';
    inputsRow.style.gap = '20px';
    inputsRow.style.width = '100%';
    
    // Define common styles for both inputs
    const commonInputStyles = {
        width: '100%',
        boxSizing: 'border-box',
        height: '38px',
        border: '1px solid #ced4da',
        borderRadius: '4px',
        padding: '8px 12px',
        fontSize: '16px',
        backgroundColor: '#fff',
        color: '#333'
    };
    
    // Number of Simulations
    const simulationsContainer = document.createElement('div');
    simulationsContainer.style.flex = '1';
    const simulationsLabel = document.createElement('label');
    simulationsLabel.setAttribute('for', 'numSimulations');
    simulationsLabel.textContent = 'Number of Simulations:';
    simulationsLabel.style.display = 'block';
    simulationsLabel.style.marginBottom = '5px';
    
    const simulationsInput = document.createElement('input');
    simulationsInput.id = 'numSimulations';
    simulationsInput.type = 'number';
    simulationsInput.value = '500';
    simulationsInput.min = '100';
    simulationsInput.max = '1000';
    
    // Apply common styles to simulations input
    Object.assign(simulationsInput.style, commonInputStyles);
    
    simulationsContainer.appendChild(simulationsLabel);
    simulationsContainer.appendChild(simulationsInput);
    inputsRow.appendChild(simulationsContainer);
    
    // Confidence Interval
    const confidenceContainer = document.createElement('div');
    confidenceContainer.style.flex = '1';
    const confidenceLabel = document.createElement('label');
    confidenceLabel.setAttribute('for', 'confidenceInterval');
    confidenceLabel.textContent = 'Confidence Interval:';
    confidenceLabel.style.display = 'block';
    confidenceLabel.style.marginBottom = '5px';
    
    const confidenceSelect = document.createElement('select');
    confidenceSelect.id = 'confidenceInterval';
    
    // Apply common styles to confidence select
    Object.assign(confidenceSelect.style, commonInputStyles);
    
    // Additional select-specific styles to make it look more like the number input
    confidenceSelect.style.appearance = 'none'; // Remove default select appearance
    confidenceSelect.style.WebkitAppearance = 'none'; // For Safari
    confidenceSelect.style.MozAppearance = 'none'; // For Firefox
    confidenceSelect.style.backgroundImage = 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23007CB2%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")';
    confidenceSelect.style.backgroundRepeat = 'no-repeat';
    confidenceSelect.style.backgroundPosition = 'right 8px center';
    confidenceSelect.style.backgroundSize = '12px';
    confidenceSelect.style.paddingRight = '28px'; // Make room for the custom arrow
    
    // Add options to select
    const option90 = document.createElement('option');
    option90.value = '0.90';
    option90.textContent = '90%';
    confidenceSelect.appendChild(option90);
    
    const option95 = document.createElement('option');
    option95.value = '0.95';
    option95.textContent = '95%';
    option95.selected = true;
    confidenceSelect.appendChild(option95);
    
    const option99 = document.createElement('option');
    option99.value = '0.99';
    option99.textContent = '99%';
    confidenceSelect.appendChild(option99);
    
    confidenceContainer.appendChild(confidenceLabel);
    confidenceContainer.appendChild(confidenceSelect);
    inputsRow.appendChild(confidenceContainer);
    
    // Add the inputs row to the options row
    optionsRow.appendChild(inputsRow);
    
    // Add elements to the form container
    formContainer.appendChild(description);
    formContainer.appendChild(dateRow);
    formContainer.appendChild(optionsRow);
    
    // Add the style marker
    const styleMarker = document.createElement('div');
    styleMarker.className = 'mc-apple-style';
    styleMarker.style.display = 'none';
    formContainer.appendChild(styleMarker);
    
    // Add the form container to the modal content
    modalContent.insertBefore(formContainer, mcResults);
}

// Helper function to create simulation statistics table
function createSimulationStats(simulationResults, normalizeValue, dates) {
    const statsDiv = document.getElementById('simulationStats');
    statsDiv.innerHTML = '<h4 style="text-align: center;">Simulation Statistics</h4>';
    
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
        
        // Calculate percentage changes for better visual indicators
        const expectedValuePercent = ((mcData.final_mean_value / normalizeValue - 1) * 100).toFixed(2);
        const probGainPercent = (mcData.probability_of_gain * 100).toFixed(2);
        const maxGainPercent = (((mcData.max_gain + normalizeValue) / normalizeValue - 1) * 100).toFixed(2);
        
        // Determine positive/negative classes based on values
        const expectedValueClass = parseFloat(expectedValuePercent) >= 0 ? 'positive' : 'negative';
        const probGainClass = parseFloat(probGainPercent) >= 90 ? 'positive' : parseFloat(probGainPercent) >= 70 ? 'neutral' : 'negative';
        const maxGainClass = parseFloat(maxGainPercent) >= 15 ? 'positive' : 'neutral';
        
        row.innerHTML = `
            <td>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 12px; height: 12px; background-color: ${portfolioColor}; border-radius: 50%; margin-right: 8px;"></span>
                    <span style="font-weight: 500;">${portfolioName}</span>
                </div>
            </td>
            <td>
                <div>${formatCurrency(mcData.final_mean_value)}</div>
                <div class="percent-change ${expectedValueClass}">${expectedValuePercent >= 0 ? '+' : ''}${expectedValuePercent}%</div>
            </td>
            <td>
                <div>${formatCurrency(mcData.final_lower_bound)} - ${formatCurrency(mcData.final_upper_bound)}</div>
            </td>
            <td>
                <div>${formatCurrency(Math.abs(mcData.var_95))}</div>
            </td>
            <td>
                <div class="probability ${probGainClass}">${probGainPercent}%</div>
            </td>
            <td>
                <div>${formatCurrency(mcData.max_gain)}</div>
                <div class="percent-change ${maxGainClass}">${maxGainPercent >= 0 ? '+' : ''}${maxGainPercent}%</div>
            </td>
        `;
        
        statsTable.appendChild(row);
    });
    
    statsDiv.appendChild(statsTable);
}

// Helper function to set up tooltips for the Monte Carlo chart
function setupMonteCarloTooltips(chart, simulationResults, dates, normalizeValue) {
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
            const activePoints = chart.getElementsAtEventForMode(
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
                    const meta = chart.getDatasetMeta(datasetIndex);
                    
                    if (!meta.hidden) {
                        const name = result.name;
                        const value = result.mcData.mean_path[dataIndex] / normalizeValue;
                        const percentChange = ((value - 1) * 100).toFixed(2);
                        const sign = percentChange >= 0 ? '+' : '';
                        const color = result.color || '#4a90e2';
                        
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

function resetFutureZoom() {
    if (performanceChart) {
        performanceChart.resetZoom();
    }
}

function calculateFuturePerformance() {
    const startDate = document.getElementById('futureStartDate').value;
    const endDate = document.getElementById('futureEndDate').value;
    const includeBuildNew = document.getElementById('includeBuildNew').checked;

    if (!startDate || !endDate) {
        alert('Please select both start and end dates');
        return;
    }

    if (new Date(startDate) >= new Date(endDate)) {
        alert('Start date must be before end date');
        return;
    }

    // Show loading state
    const analyzeBtn = document.querySelector('.analyze-btn');
    const resultsDiv = document.getElementById('futureResults');
    analyzeBtn.textContent = 'Analyzing...';
    analyzeBtn.disabled = true;
    resultsDiv.innerHTML = '<div class="loading-spinner"></div>';
    resultsDiv.style.display = 'block';

    fetch('/calculate-future-performance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate,
            include_build_new: includeBuildNew
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;

        if (data.error) {
            resultsDiv.innerHTML = `<div class="error-message">
                <p>Error: ${data.error}</p>
                <p>Try selecting a different date range. The selected period may have unusual market conditions or negative returns.</p>
            </div>`;
            return;
        }

        // Check if we have valid portfolios data
        if (!data.portfolios || data.portfolios.length === 0 || !data.dates || data.dates.length === 0) {
            resultsDiv.innerHTML = `<div class="error-message">
                <p>Error: No portfolio data available for the selected period.</p>
                <p>Try selecting a different date range with more positive market performance.</p>
            </div>`;
            return;
        }
        
        const maxValue = Math.max(...data.portfolios.flatMap(p => p.values));
        const minValue = Math.min(...data.portfolios.flatMap(p => p.values));

        // Calculate a more dynamic and tighter y-axis range
        // Find the range of the data
        const dataRange = maxValue - minValue;
        // Add just 10% padding above and below
        const padding = dataRange * 0.1;
        // Round to 2 decimal places for clean limits
        const yMax = Math.ceil((maxValue + padding) * 100) / 100;
        const yMin = Math.floor((minValue - padding) * 100) / 100;

        // Reset the results div with the proper structure
        resultsDiv.innerHTML = `
            <h3 style="text-align: center; margin-top: 20px;">Portfolio Performance Comparison</h3>
            <div id="backtest-custom-legend" class="chart-legend-pills"></div>
            <div class="chart-container" style="position: relative;">
                <canvas id="performanceChart"></canvas>
                <div id="backtest-tooltip" style="position: absolute; display: none; background-color: rgba(0,0,0,0.8); color: white; padding: 10px; border-radius: 5px; pointer-events: none; z-index: 10000; width: auto; min-width: 150px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);"></div>
            </div>
            <div class="chart-actions">
                <button class="reset-zoom-btn" onclick="resetFutureZoom()">Reset Zoom</button>
            </div>
        `;
        resultsDiv.style.display = 'block';

        // Create the chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        if (performanceChart) {
            performanceChart.destroy();
        }

        // Store portfolio data for the custom legend
        const portfolioData = {};
        
        // Process portfolio data for the chart
        const datasets = data.portfolios.map((portfolio, index) => {
            // Store portfolio data for legend
            portfolioData[portfolio.name] = {
                name: portfolio.name,
                color: portfolio.color,
                optimizationData: portfolio.optimizationData,
                index: index  // Store the dataset index
            };
            
            return {
                label: portfolio.name,
                data: portfolio.values,
                borderColor: portfolio.color,
                fill: false,
                tension: 0.2,
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 10,
                optimizationData: portfolio.optimizationData
            };
        });

        performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'xy'
                        },
                        zoom: {
                            wheel: {
                                enabled: true, // Enable wheel zoom when used with modifier key
                                modifierKey: 'meta', // Use meta key (Command on Mac)
                                speed: 0.1 // Add speed for smoother zooming
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
                    },
                    title: {
                        display: false,
                        text: 'Portfolio Performance Over Time',
                        font: {
                            size: 18
                        },
                        padding: 20
                    },
                    tooltip: {
                        enabled: false // Disable built-in tooltips, we'll use custom ones
                    },
                    legend: {
                        display: false // Disable built-in legend, we'll use custom one
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
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
                        },
                        title: {
                            display: true,
                            text: 'Normalized Portfolio Value',
                            font: {
                                size: 14
                            }
                        }
                    },
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
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
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
                events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
                onHover: function(evt, activeElements) {
                    // For user experience, change cursor to pointer when over data points
                    if (evt && evt.native) {
                        evt.native.target.style.cursor = activeElements.length ? 'pointer' : 'default';
                    }
                }
            }
        });
        
        // Create custom legend
        createCustomBacktestLegend(performanceChart, portfolioData);
        
        // Setup tooltips
        setupBacktestTooltips(performanceChart, data.portfolios, data.dates);
    })
    .catch(error => {
        analyzeBtn.textContent = 'Analyze';
        analyzeBtn.disabled = false;
        console.error('Error:', error);
        alert('Failed to analyze future performance');
    });
}

function createCustomBacktestLegend(chart, portfolioData) {
    const legendContainer = document.getElementById('backtest-custom-legend');
    if (!legendContainer) return;
    
    // Clear any existing legend
    legendContainer.innerHTML = '';
    
    // Create a pill for each portfolio
    Object.keys(portfolioData).forEach(portfolioName => {
        const portfolio = portfolioData[portfolioName];
        
        // Create the pill element
        const pill = document.createElement('div');
        pill.className = 'legend-pill active';
        pill.innerHTML = `
            <span class="legend-pill-color" style="background-color: ${portfolio.color};"></span>
            <span>${portfolioName}</span>
        `;
        
        // Add click event to toggle visibility
        pill.addEventListener('click', (e) => {
            // Check if Command/Ctrl key is pressed (show portfolio details)
            if (e.metaKey || e.ctrlKey) {
                // Show portfolio details if available
                if (portfolio) {
                    // Extract detailed portfolio data
                    const optimizationData = portfolio.optimizationData;
                    
                    // The optimization data already has the correct structure, use it directly
                    // instead of trying to reformat it
                    
                    // Ensure the future performance modal is behind the portfolio details modal
                    document.getElementById('portfolioDetailsModal').style.zIndex = 1100;
                    document.getElementById('futureModal').style.zIndex = 1000;
                    
                    // Show the portfolio details
                    showPortfolioDetails(optimizationData, portfolioName);
                }
            } else {
                // Regular click - toggle visibility
                const meta = chart.getDatasetMeta(portfolio.index);
                meta.hidden = !meta.hidden;
                
                // Toggle active class
                pill.classList.toggle('active');
                
                // Update chart
                chart.update();
            }
        });
        
        // Add pill to container
        legendContainer.appendChild(pill);
    });
}

function setupBacktestTooltips(chart, portfolios, dates) {
    const tooltip = document.getElementById('backtest-tooltip');
    const chartElement = document.getElementById('performanceChart');
    
    // Only add the listener if both elements exist
    if (chartElement && tooltip) {
        chartElement.addEventListener('mousemove', function(event) {
            // Get mouse position relative to chart
            const rect = chartElement.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            
            // Get the data point at mouse position
            const activePoints = chart.getElementsAtEventForMode(
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
                portfolios.forEach((portfolio, i) => {
                    // Check if this dataset is visible
                    const datasetIndex = i;
                    const meta = chart.getDatasetMeta(datasetIndex);
                    
                    if (!meta.hidden) {
                        const name = portfolio.name;
                        const value = portfolio.values[dataIndex];
                        const percentChange = ((value - 1) * 100).toFixed(2);
                        const sign = percentChange >= 0 ? '+' : '';
                        const color = portfolio.color;
                        
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