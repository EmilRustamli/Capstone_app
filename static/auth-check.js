// Authentication check script
function checkAuthStatus() {
    // Clear the logged out flag when we successfully authenticate
    // This ensures normal navigation works when properly logged in
    if (sessionStorage.getItem('userLoggedOut') === 'true') {
        // Only apply the strict checks if user was logged out
        fetch('/check-auth', {
            method: 'GET',
            credentials: 'same-origin', // Include cookies
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        })
        .then(response => {
            if (!response.ok) {
                // If not authenticated, redirect to login page
                window.location.href = '/';
                return null;
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                // If authenticated, clear the logged out flag to allow normal navigation
                if (data.authenticated === true) {
                    sessionStorage.removeItem('userLoggedOut');
                } else if (data.authenticated === false) {
                    // Still not authenticated, redirect
                    window.location.href = '/';
                }
            }
        })
        .catch(error => {
            console.error('Auth check failed:', error);
            // On error, redirect to login to be safe
            window.location.href = '/';
        });
    } else {
        // Normal authentication check when not coming from logout
        fetch('/check-auth', {
            method: 'GET',
            credentials: 'same-origin',
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        })
        .then(response => {
            if (!response.ok) {
                window.location.href = '/';
                return null;
            }
            return response.json();
        })
        .then(data => {
            if (data && data.authenticated === false) {
                window.location.href = '/';
            }
        })
        .catch(error => {
            console.error('Auth check failed:', error);
        });
    }
}

// Modified approach to prevent back button after logout
// This is less aggressive and only blocks the back button after logout
function preventBackAfterLogout() {
    // Only apply history manipulation if user was logged out
    if (sessionStorage.getItem('userLoggedOut') === 'true') {
        window.history.pushState(null, '', window.location.href);
        window.addEventListener('popstate', function() {
            window.history.pushState(null, '', window.location.href);
            // Re-check auth status when back button is pressed
            checkAuthStatus();
        });
    }
}

// Run auth check when page loads
document.addEventListener('DOMContentLoaded', function() {
    checkAuthStatus();
    preventBackAfterLogout();
}); 