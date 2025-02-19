function showTab(tabName) {
    // Update content
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');
    
    // Update navigation
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-tab') === tabName) {
            link.classList.add('active');
        }
    });
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

function showForgotPassword(event) {
    event.preventDefault();
    document.getElementById('loginForm').style.display = 'none';
    document.getElementById('forgotPasswordDiv').style.display = 'block';
}

async function handleForgotPassword(event) {
    event.preventDefault();
    const form = event.target;
    const email = form.querySelector('input[type="email"]').value;

    try {
        const response = await fetch('/forgot-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email })
        });

        const data = await response.json();
        
        if (response.ok) {
            alert(data.message);
            document.getElementById('forgotPasswordDiv').style.display = 'none';
            document.getElementById('resetPasswordDiv').style.display = 'block';
            localStorage.setItem('resetPasswordEmail', email);
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
    }
}

async function handleResetPassword(event) {
    event.preventDefault();
    const form = event.target;
    const inputs = form.querySelectorAll('input');
    const code = inputs[0].value;
    const newPassword = inputs[1].value;
    const confirmPassword = inputs[2].value;
    const email = localStorage.getItem('resetPasswordEmail');

    if (newPassword !== confirmPassword) {
        alert('Passwords do not match!');
        return;
    }

    try {
        const response = await fetch('/reset-password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email,
                code,
                new_password: newPassword
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            alert(data.message);
            localStorage.removeItem('resetPasswordEmail');
            document.getElementById('resetPasswordDiv').style.display = 'none';
            document.getElementById('loginForm').style.display = 'block';
            form.reset();
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('An error occurred. Please try again.');
    }
}

function backToLogin() {
    document.getElementById('forgotPasswordDiv').style.display = 'none';
    document.getElementById('resetPasswordDiv').style.display = 'none';
    document.getElementById('loginForm').style.display = 'block';
} 