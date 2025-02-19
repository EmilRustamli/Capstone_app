function showTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

async function handleLogin(event) {
    event.preventDefault();
    const form = event.target;
    const email = form.querySelector('input[type="email"]').value;
    const password = form.querySelector('input[type="password"]').value;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            // Remove the alert as it might be blocking the redirect
            // alert(data.message);
            form.reset();
            // Immediate redirect to dashboard
            window.location.href = data.redirect;
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
    }
}

async function handleRegister(event) {
    event.preventDefault();
    const form = event.target;
    const username = form.querySelector('input[type="text"]').value;
    const email = form.querySelector('input[type="email"]').value;
    const password = form.querySelector('input[type="password"]').value;

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();
        
        if (response.ok) {
            alert(data.message);
            // Show verification code input
            document.getElementById('verification-div').style.display = 'block';
            // Store email for verification
            localStorage.setItem('pendingVerificationEmail', email);
            form.reset();
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
    }
}

async function handleVerification(event) {
    event.preventDefault();
    const code = document.getElementById('verification-code').value;
    const email = localStorage.getItem('pendingVerificationEmail');

    try {
        const response = await fetch('/verify-code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, code })
        });

        const data = await response.json();
        
        if (response.ok) {
            alert(data.message);
            document.getElementById('verification-div').style.display = 'none';
            document.getElementById('verification-code').value = '';
            localStorage.removeItem('pendingVerificationEmail');
            showTab('login');
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
    }
} 