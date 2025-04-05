function showLogoutModal(event) {
    event.preventDefault();
    document.getElementById('logoutModal').style.display = 'flex';
}

function closeLogoutModal() {
    document.getElementById('logoutModal').style.display = 'none';
}

function confirmLogout() {
    // Don't close the modal, instead update its content to show logging out
    const modal = document.getElementById('logoutModal');
    const modalContent = modal.querySelector('.modal');
    
    // Save the original content so we can restore it if needed
    if (!modal.originalContent) {
        modal.originalContent = modalContent.innerHTML;
    }
    
    // Replace the modal content with a loading message
    modalContent.innerHTML = `
        <h2>Logging out...</h2>
        <p>Please wait while we securely log you out.</p>
        <div class="logout-spinner"></div>
    `;
    
    // Set a flag to indicate logout happened BEFORE making the request
    sessionStorage.setItem('userLoggedOut', 'true');
    
    // Clear local storage and session storage except the logout flag
    const logoutFlag = sessionStorage.getItem('userLoggedOut');
    localStorage.clear();
    sessionStorage.clear();
    // Restore the logout flag
    sessionStorage.setItem('userLoggedOut', logoutFlag);
    
    // Perform logout with fetch to ensure it's complete
    fetch('/logout', {
        method: 'GET',
        credentials: 'same-origin',
        headers: {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    })
    .then(() => {
        // Redirect to home page
        window.location.href = '/';
    })
    .catch(() => {
        // Even if there's an error, redirect to home page
        window.location.href = '/';
    });
}

// When page loads, ensure logout modal is hidden
document.addEventListener('DOMContentLoaded', function() {
    // Make sure the logout modal is hidden by default
    const logoutModal = document.getElementById('logoutModal');
    if (logoutModal) {
        logoutModal.style.display = 'none';
    }
});

// Close modal if clicking outside
window.addEventListener('click', function(event) {
    const modal = document.getElementById('logoutModal');
    if (event.target === modal) {
        closeLogoutModal();
    }
}); 