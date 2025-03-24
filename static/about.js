// Initialize logout modal functionality
document.addEventListener('DOMContentLoaded', function() {
    // Make sure the logout modal is hidden by default
    const logoutModal = document.getElementById('logoutModal');
    if (logoutModal) {
        logoutModal.style.display = 'none';
    }
});

// These functions should match what's in logout.js to ensure consistency
function showLogoutModal(event) {
    event.preventDefault();
    document.getElementById('logoutModal').style.display = 'flex';
}

function closeLogoutModal() {
    document.getElementById('logoutModal').style.display = 'none';
}

function confirmLogout() {
    // Hide the modal first
    closeLogoutModal();
    
    // Perform logout
    window.location.href = '/logout';
}

// Close modal if clicking outside
window.addEventListener('click', function(event) {
    const modal = document.getElementById('logoutModal');
    if (event.target === modal) {
        closeLogoutModal();
    }
}); 