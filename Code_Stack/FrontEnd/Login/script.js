// ============================================
// LOGIN AUTHENTICATION SYSTEM
// ============================================

// Get subdomain from URL (e.g., "acmecorp" from "acmecorp.vigilsecure.com")
function getSubdomain() {
    const hostname = window.location.hostname;
    const parts = hostname.split('.');

    // If localhost or IP, return test subdomain
    if (hostname === 'localhost' || hostname.startsWith('127.0.0.1')) {
        return 'testclient';
    }

    // Return first part as subdomain (company name)
    return parts[0];
}

// Store subdomain for routing
const CLIENT_SUBDOMAIN = getSubdomain();
console.log('Client Subdomain:', CLIENT_SUBDOMAIN);

// ============================================
// UTILITY FUNCTIONS
// ============================================
function showStatus(element, type, message) {
    element.className = `form-status ${type}`;
    element.textContent = message;
    element.style.display = 'block';
}

// ============================================
// ADMIN AUTH TOKEN
// ============================================
let adminAuthToken = null;

// ============================================
// WAIT FOR DOM TO LOAD BEFORE ATTACHING EVENTS
// ============================================
window.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - Attaching event listeners...');

    // Clear only Vigil-specific tokens on login page load (don't clear all localStorage)
    const vigilKeys = ['userToken', 'adminToken', 'userRole', 'username', 'clientSubdomain', 'adminEmail', 'isAdmin', 'vigil_chat_sessions'];
    vigilKeys.forEach(key => localStorage.removeItem(key));

    // Display subdomain info in console for debugging
    console.log('='.repeat(60));
    console.log('VIGIL LOGIN PAGE');
    console.log('='.repeat(60));
    console.log('Client Subdomain:', CLIENT_SUBDOMAIN);
    console.log('Full Hostname:', window.location.hostname);
    console.log('Database Routing:', `${CLIENT_SUBDOMAIN}_db`);
    console.log('Admin Email:', `admin@${CLIENT_SUBDOMAIN}.vigilmail.com`);
    console.log('='.repeat(60));

    // ============================================
    // TOGGLE PASSWORD VISIBILITY
    // ============================================
    document.querySelectorAll('.toggle-password').forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.getAttribute('data-target');
            const input = document.getElementById(targetId);

            if (input.type === 'password') {
                input.type = 'text';
            } else {
                input.type = 'password';
            }
        });
    });

    // ============================================
    // USER LOGIN
    // ============================================
    const userLoginForm = document.getElementById('userLoginForm');
    if (userLoginForm) {
        console.log('User login form found, attaching handler');
        userLoginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('User login form submitted');

            const username = document.getElementById('userUsername').value;
            const password = document.getElementById('userPassword').value;
            const statusEl = document.getElementById('userStatus');

            // Show pending status
            showStatus(statusEl, 'pending', 'Authenticating...');

            try {
                // Call authentication API with subdomain routing
                const response = await fetch('/api/auth/user-login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Subdomain': CLIENT_SUBDOMAIN
                    },
                    body: JSON.stringify({
                        username,
                        password,
                        subdomain: CLIENT_SUBDOMAIN
                    })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    showStatus(statusEl, 'success', 'Login successful! Redirecting...');

                    // Store session token
                    localStorage.setItem('userToken', data.token);
                    localStorage.setItem('userRole', 'user');
                    localStorage.setItem('username', data.username);
                    localStorage.setItem('clientSubdomain', CLIENT_SUBDOMAIN);
                    localStorage.setItem('isAdmin', 'false');

                    // Redirect to main interface
                    setTimeout(() => {
                        window.location.href = '../Interface/index.html';
                    }, 1500);
                } else {
                    showStatus(statusEl, 'error', data.message || 'Invalid credentials');
                }
            } catch (error) {
                showStatus(statusEl, 'error', 'Connection error. Please try again.');
                console.error('User login error:', error);
            }
        });
    } else {
        console.error('ERROR: User login form not found!');
    }

    // ============================================
    // ADMIN LOGIN
    // ============================================
    const adminLoginForm = document.getElementById('adminLoginForm');
    if (adminLoginForm) {
        console.log('Admin login form found, attaching handler');
        adminLoginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log('Admin login form submitted');

            const username = document.getElementById('adminUsername').value;
            const password = document.getElementById('adminPassword').value;
            const statusEl = document.getElementById('adminStatus');
            const authVerification = document.getElementById('adminAuthVerification');

            // Show pending status
            showStatus(statusEl, 'pending', 'Authenticating...');

            try {
                // Call admin authentication API with subdomain routing
                const response = await fetch('/api/auth/admin-login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Subdomain': CLIENT_SUBDOMAIN
                    },
                    body: JSON.stringify({
                        username,
                        password,
                        subdomain: CLIENT_SUBDOMAIN
                    })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // Store temporary auth token
                    adminAuthToken = data.authToken;

                    // Show 6-digit code verification
                    showStatus(statusEl, 'success', 'Credentials verified. Check your secure email for the authentication code.');
                    authVerification.style.display = 'flex';

                    // Focus on code input
                    document.getElementById('adminAuthCode').focus();
                } else {
                    showStatus(statusEl, 'error', data.message || 'Invalid admin credentials');
                }
            } catch (error) {
                showStatus(statusEl, 'error', 'Connection error. Please try again.');
                console.error('Admin login error:', error);
            }
        });
    } else {
        console.error('ERROR: Admin login form not found!');
    }

    // ============================================
    // ADMIN 6-DIGIT CODE VERIFICATION
    // ============================================
    const verifyAdminCodeBtn = document.getElementById('verifyAdminCode');
    if (verifyAdminCodeBtn) {
        console.log('Admin verify button found, attaching handler');
        verifyAdminCodeBtn.addEventListener('click', async () => {
            const code = document.getElementById('adminAuthCode').value;
            const statusEl = document.getElementById('adminStatus');
            const username = document.getElementById('adminUsername').value;

            if (code.length !== 6 || !/^\d{6}$/.test(code)) {
                showStatus(statusEl, 'error', 'Please enter a valid 6-digit code');
                return;
            }

            showStatus(statusEl, 'pending', 'Verifying code...');

            try {
                const response = await fetch('/api/auth/admin-verify-code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Subdomain': CLIENT_SUBDOMAIN
                    },
                    body: JSON.stringify({
                        authToken: adminAuthToken,
                        code: code,
                        subdomain: CLIENT_SUBDOMAIN
                    })
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    showStatus(statusEl, 'success', 'Authentication successful! Redirecting...');

                    // Store admin session token
                    localStorage.setItem('adminToken', data.token);
                    localStorage.setItem('userRole', 'admin');
                    localStorage.setItem('username', data.username);
                    localStorage.setItem('clientSubdomain', CLIENT_SUBDOMAIN);
                    localStorage.setItem('adminEmail', data.adminEmail);
                    localStorage.setItem('isAdmin', 'true');

                    // Redirect to main interface
                    setTimeout(() => {
                        window.location.href = '../Interface/index.html';
                    }, 1500);
                } else {
                    showStatus(statusEl, 'error', data.message || 'Invalid authentication code');
                }
            } catch (error) {
                showStatus(statusEl, 'error', 'Verification error. Please try again.');
                console.error('Admin code verification error:', error);
            }
        });
    } else {
        console.error('ERROR: Admin verify button not found!');
    }

    // Auto-submit code when 6 digits entered
    const adminAuthCodeInput = document.getElementById('adminAuthCode');
    if (adminAuthCodeInput) {
        adminAuthCodeInput.addEventListener('input', (e) => {
            const code = e.target.value;
            if (code.length === 6 && /^\d{6}$/.test(code)) {
                document.getElementById('verifyAdminCode').click();
            }
        });
    }

    // ============================================
    // MODAL HANDLERS
    // ============================================

    // Forgot Password User
    const forgotPasswordUser = document.getElementById('forgotPasswordUser');
    if (forgotPasswordUser) {
        forgotPasswordUser.addEventListener('click', () => {
            document.getElementById('forgotPasswordModal').classList.add('active');
        });
    }

    // Forgot Password Admin
    const forgotPasswordAdmin = document.getElementById('forgotPasswordAdmin');
    if (forgotPasswordAdmin) {
        forgotPasswordAdmin.addEventListener('click', () => {
            document.getElementById('forgotPasswordModal').classList.add('active');
        });
    }

    // Create User Account
    const createUserAccount = document.getElementById('createUserAccount');
    if (createUserAccount) {
        createUserAccount.addEventListener('click', () => {
            document.getElementById('createAccountModal').classList.add('active');
        });
    }

    // Close Modals
    const closeForgotPassword = document.getElementById('closeForgotPassword');
    if (closeForgotPassword) {
        closeForgotPassword.addEventListener('click', () => {
            document.getElementById('forgotPasswordModal').classList.remove('active');
        });
    }

    const closeCreateAccount = document.getElementById('closeCreateAccount');
    if (closeCreateAccount) {
        closeCreateAccount.addEventListener('click', () => {
            document.getElementById('createAccountModal').classList.remove('active');
        });
    }

    // Close on overlay click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.classList.remove('active');
            }
        });
    });

    // ============================================
    // FORGOT PASSWORD FORM
    // ============================================
    const forgotPasswordForm = document.getElementById('forgotPasswordForm');
    if (forgotPasswordForm) {
        forgotPasswordForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('resetUsername').value;
            const adminName = document.getElementById('resetAdminName').value;
            const statusEl = document.getElementById('resetStatus');

            showStatus(statusEl, 'pending', 'Verifying account and sending reset code...');

            try {
                // Backend will:
                // 1. Verify username exists in database
                // 2. Verify username is linked to admin with provided name
                // 3. Get admin email from database (admin@{subdomain}.vigilmail.com)
                // 4. Send reset code to admin email
                const response = await fetch('/api/auth/forgot-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Subdomain': CLIENT_SUBDOMAIN
                    },
                    body: JSON.stringify({
                        username,
                        adminName,
                        subdomain: CLIENT_SUBDOMAIN
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    showStatus(statusEl, 'success', 'Reset code sent to administrator for approval.');
                } else {
                    showStatus(statusEl, 'error', data.message || 'Invalid username or admin name');
                }
            } catch (error) {
                showStatus(statusEl, 'error', 'Connection error');
            }
        });
    }

    // ============================================
    // CREATE ACCOUNT FORM
    // ============================================
    const createAccountForm = document.getElementById('createAccountForm');
    if (createAccountForm) {
        createAccountForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('newUsername').value;
            const adminName = document.getElementById('newAdminName').value;
            const password = document.getElementById('newPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            const statusEl = document.getElementById('createStatus');

            if (password !== confirmPassword) {
                showStatus(statusEl, 'error', 'Passwords do not match');
                return;
            }

            showStatus(statusEl, 'pending', 'Verifying admin and sending approval request...');

            try {
                // Backend will:
                // 1. Verify admin with provided name exists in database
                // 2. Create pending user account linked to admin
                // 3. Get admin email from database (admin@{subdomain}.vigilmail.com)
                // 4. Send approval request with verification code to admin email
                const response = await fetch('/api/auth/request-user-account', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Client-Subdomain': CLIENT_SUBDOMAIN
                    },
                    body: JSON.stringify({
                        username,
                        adminName,
                        password,
                        subdomain: CLIENT_SUBDOMAIN
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    showStatus(statusEl, 'success', 'Request sent to administrator! Awaiting approval via email.');
                } else {
                    showStatus(statusEl, 'error', data.message || 'Invalid admin name or error creating request');
                }
            } catch (error) {
                showStatus(statusEl, 'error', 'Connection error');
            }
        });
    }

    console.log('All event listeners attached successfully');
});
