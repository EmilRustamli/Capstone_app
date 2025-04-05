// Session verification script
document.addEventListener('DOMContentLoaded', function() {
    // Function to check if user is still logged in
    function checkSession() {
        // Get the current path to avoid redirect loops
        const currentPath = window.location.pathname;
        
        // Skip check if we're already on the home page
        if (currentPath === '/' || currentPath === '/index.html') {
            return;
        }
        
        fetch('/verify-session')
            .then(response => {
                if (!response.ok) {
                    // If session is invalid, redirect to login page
                    window.location.href = '/';
                    return null; // don't continue with response.json()
                }
                return response.json();
            })
            .then(data => {
                if (data && !data.valid) {
                    // If session is explicitly marked as invalid
                    window.location.href = '/';
                }
            })
            .catch(error => {
                console.error('Session check failed:', error);
                // Only redirect on session-related errors, not network errors
                if (error.name !== 'TypeError') {
                    window.location.href = '/';
                }
            });
    }
    
    // Run check on page load
    checkSession();
    
    // Also check when user becomes active after being away
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible') {
            checkSession();
        }
    });
}); 