// Dynamically set API base URL based on environment
const API_BASE_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000/api'
  : `${window.location.protocol}//${window.location.host}/api`;

// Helper function to safely get DOM elements
function getElement(id) {
    const el = document.getElementById(id);
    if (!el) {
        console.warn(`Element with id "${id}" not found`);
    }
    return el;
}

const messages = getElement('messages');
const input = getElement('input');
const sendBtn = getElement('sendBtn');

// Custom scrollbar elements
const customScrollbar = getElement('customScrollbar');
const customScrollbarThumb = getElement('customScrollbarThumb');

const docInput = getElement('docInput');
const photoInput = getElement('photoInput');
const audioInput = getElement('audioInput');
const videoInput = getElement('videoInput');

const docBtn = getElement('docBtn');
const photoBtn = getElement('photoBtn');
const audioBtn = getElement('audioBtn');
const videoBtn = getElement('videoBtn');
const expandBtn = getElement('expandBtn');

// Alerts elements
const alertsToggle = getElement('alertsToggle');
const alertsPanel = getElement('alertsPanel');
const alertsClose = getElement('alertsClose');
const alertList = getElement('alertList');
const alertDetailView = getElement('alertDetailView');
const alertBack = getElement('alertBack');
const alertChatBtn = getElement('alertChatBtn');

// Chat History elements
const historyToggle = getElement('historyToggle');
const historyPanel = getElement('historyPanel');
const historyClose = getElement('historyClose');
const historyList = getElement('historyList');
const historyNewChat = getElement('historyNewChat');

// Document Search elements
const docSearchInput = getElement('docSearchInput');
const docSearchOverlay = getElement('docSearchOverlay');
const docSearchResults = getElement('docSearchResults');
const docSearchClose = getElement('docSearchClose');

// Detail elements
const detailDot = getElement('detailDot');
const detailTitle = getElement('detailTitle');
const detailSummary = getElement('detailSummary');
const detailImpact = getElement('detailImpact');
const detailTimeline = getElement('detailTimeline');
const detailAffected = getElement('detailAffected');

// Logo elements
const logoContainer = getElement('logoContainer');
const logo = getElement('logo');

let isLoading = false;
let conversationHistory = [];
let chatActive = false;
let currentAlert = null;
let searchTimeout = null;
let currentSessionId = null;
let chatSessions = [];

// Attachment menu toggle
const attachContainer = getElement('attachContainer');
const attachTrigger = getElement('attachTrigger');

if (attachTrigger && attachContainer) {
    attachTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        attachContainer.classList.toggle('open');
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
        if (!attachContainer.contains(e.target)) {
            attachContainer.classList.remove('open');
        }
    });
}

// Load chat sessions from localStorage
function loadChatSessions() {
    try {
        const saved = localStorage.getItem('vigil_chat_sessions');
        if (saved) {
            chatSessions = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Error loading chat sessions:', e);
        chatSessions = [];
    }
    renderChatHistory();
}

// Save chat sessions to localStorage
function saveChatSessions() {
    try {
        localStorage.setItem('vigil_chat_sessions', JSON.stringify(chatSessions));
    } catch (e) {
        console.error('Error saving chat sessions:', e);
    }
}

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
}

// Create new chat session
function createNewSession() {
    const sessionId = generateSessionId();
    const session = {
        id: sessionId,
        title: 'New Chat',
        messages: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    };
    chatSessions.unshift(session);
    currentSessionId = sessionId;
    conversationHistory = [];
    saveChatSessions();
    renderChatHistory();
    return sessionId;
}

// Update current session with messages
function updateCurrentSession() {
    if (!currentSessionId) {
        createNewSession();
    }

    const session = chatSessions.find(s => s.id === currentSessionId);
    if (session) {
        session.messages = [...conversationHistory];
        session.updatedAt = new Date().toISOString();

        // Update title from first user message
        const firstUserMsg = conversationHistory.find(m => m.role === 'user');
        if (firstUserMsg && typeof firstUserMsg.content === 'string') {
            session.title = firstUserMsg.content.substring(0, 50) + (firstUserMsg.content.length > 50 ? '...' : '');
        }

        saveChatSessions();
        renderChatHistory();
    }
}

// Load a specific chat session
function loadSession(sessionId) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (session) {
        currentSessionId = sessionId;
        conversationHistory = session.messages || [];

        // Clear messages UI and reload
        messages.innerHTML = '';
        conversationHistory.forEach(msg => {
            if (msg.role === 'user') {
                addMessageToUI(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                if (typeof msg.content === 'object') {
                    addStructuredResponseToUI(msg.content);
                } else {
                    addMessageToUI(msg.content, 'assistant');
                }
            }
        });

        // Activate chat mode if there are messages
        if (conversationHistory.length > 0 && !chatActive) {
            chatActive = true;
            logoContainer.classList.add('chat-active');
            logo.style.transform = 'none';
        }

        historyPanel.classList.remove('active');
        renderChatHistory();
    }
}

// Render chat history list
function renderChatHistory() {
    if (!historyList) return;

    if (chatSessions.length === 0) {
        historyList.innerHTML = '<div class="history-empty">No chat history yet</div>';
        return;
    }

    historyList.innerHTML = chatSessions.map(session => `
        <div class="history-item ${session.id === currentSessionId ? 'active' : ''}" data-session-id="${session.id}">
            <div class="history-item-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <div class="history-item-content">
                <div class="history-item-title">${escapeHtml(session.title)}</div>
                <div class="history-item-date">${formatDate(session.updatedAt)}</div>
            </div>
        </div>
    `).join('');

    // Add click handlers
    historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            loadSession(item.dataset.sessionId);
        });
    });
}

// Helper to add message to UI without tracking
function addMessageToUI(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

// Helper to add structured response to UI without tracking
function addStructuredResponseToUI(data) {
    // Reuse the existing addStructuredResponse logic but return the element
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    // For simplicity, just show a summary
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="response-label">Analysis Result</span><div class="response-value">Click to view full analysis</div>';
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

// Format date helper
function formatDate(dateStr) {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
}

// Escape HTML helper
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Alert data store
const alertsData = {
    1: {
        title: 'Supply Chain Disruption Detected',
        severity: 'critical',
        summary: 'Major supplier in Southeast Asia reporting production halt due to equipment failure. Expected downtime of 2-3 weeks affecting critical component supply.',
        impact: 'Production capacity reduced by 40%. Revenue impact estimated at $2.3M per week. Customer delivery delays expected for 15 accounts.',
        timeline: 'Detected: 2 hours ago. Supplier notification: 1 hour ago. Alternative sourcing initiated: 30 mins ago.',
        affected: 'Manufacturing Line A, Product SKUs 4521-4589, Tier 1 customers in North America region.'
    },
    2: {
        title: 'Quality Control Anomaly',
        severity: 'high',
        summary: 'Batch #4521 showing deviation from standard parameters. Tensile strength readings 12% below threshold on 3 consecutive samples.',
        impact: 'Batch quarantined (5,000 units). Potential recall if root cause not identified. Quality audit triggered.',
        timeline: 'First anomaly: 4 hours ago. Confirmed pattern: 2 hours ago. QC team notified: 1 hour ago.',
        affected: 'Production Line B, Warehouse Section 12, Pending shipments to 8 distributors.'
    },
    3: {
        title: 'Market Volatility Warning',
        severity: 'medium',
        summary: 'Commodity prices showing unusual fluctuation patterns. Steel prices up 8% in 48 hours, aluminum showing similar trajectory.',
        impact: 'Projected cost increase of 5-7% on raw materials. Q2 margin compression risk. Hedging positions require review.',
        timeline: 'Price movement started: 48 hours ago. Threshold alert triggered: 6 hours ago. Trend confirmed: 2 hours ago.',
        affected: 'Procurement contracts expiring Q2, Forward pricing agreements, Budget forecasts for FY2024.'
    }
};

// ============================================
// ALERTS PANEL FUNCTIONALITY
// ============================================

if (alertsToggle) {
    alertsToggle.addEventListener('click', () => {
        if (alertsPanel) alertsPanel.classList.add('active');
        showAlertList();
    });
}

if (alertsClose) {
    alertsClose.addEventListener('click', () => {
        if (alertsPanel) alertsPanel.classList.remove('active');
    });
}

// Close panel when clicking outside bubble
if (alertsPanel) {
    alertsPanel.addEventListener('click', (e) => {
        if (e.target === alertsPanel) {
            alertsPanel.classList.remove('active');
        }
    });
}

// ============================================
// CHAT HISTORY PANEL FUNCTIONALITY
// ============================================

if (historyToggle) {
    historyToggle.addEventListener('click', () => {
        if (historyPanel) historyPanel.classList.add('active');
    });
}

if (historyClose) {
    historyClose.addEventListener('click', () => {
        if (historyPanel) historyPanel.classList.remove('active');
    });
}

if (historyPanel) {
    historyPanel.addEventListener('click', (e) => {
        if (e.target === historyPanel) {
            historyPanel.classList.remove('active');
        }
    });
}

if (historyNewChat) {
    historyNewChat.addEventListener('click', () => {
        // Clear current chat
        if (messages) messages.innerHTML = '';
        conversationHistory = [];
        currentSessionId = null;
        chatActive = false;
        if (logoContainer) logoContainer.classList.remove('chat-active');

        // Create new session
        createNewSession();
        if (historyPanel) historyPanel.classList.remove('active');
    });
}

// ============================================
// SETTINGS PANEL FUNCTIONALITY
// ============================================

// Settings elements
const settingsToggle = document.getElementById('settingsToggle');
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsClose = document.getElementById('settingsClose');
const settingsTabs = document.querySelectorAll('.settings-tab');
const accountTab = document.getElementById('accountTab');
const logoutTab = document.getElementById('logoutTab');
const sendAuthCode = document.getElementById('sendAuthCode');
const authCodeInput = document.getElementById('authCodeInput');
const codeInput = document.getElementById('codeInput');
const verifyCode = document.getElementById('verifyCode');
const authStatus = document.getElementById('authStatus');
const logoutBtn = document.getElementById('logoutBtn');

// User account data (loaded from localStorage after login)
// Note: Password is not stored client-side for security - fetched via authenticated API when needed
let userAccountData = {
    isAdmin: localStorage.getItem('isAdmin') === 'true',
    username: localStorage.getItem('username') || 'user_12345',
    adminEmail: localStorage.getItem('adminEmail') || localStorage.getItem('clientSubdomain')
        ? `admin@${localStorage.getItem('clientSubdomain')}.vigilmail.com`
        : 'admin@company.com'
};

let isAuthenticated = false;
let pendingAuthCode = null;

// Open settings panel
settingsToggle.addEventListener('click', () => {
    settingsOverlay.classList.add('active');
    // Reset to account tab
    switchSettingsTab('account');
    // Reset authentication state when opening
    resetAuthState();
    // Update admin email field visibility based on account type
    updateAdminEmailFieldState();
});

// Update admin email field based on account type
function updateAdminEmailFieldState() {
    const adminEmailField = document.getElementById('adminEmailField');
    const revealAdminEmailBtn = document.getElementById('revealAdminEmail');

    if (!userAccountData.isAdmin) {
        // USER account - grey out and disable reveal button
        adminEmailField.classList.add('admin-restricted');
        revealAdminEmailBtn.classList.add('disabled');
    } else {
        // ADMIN account - enable reveal button
        adminEmailField.classList.remove('admin-restricted');
        revealAdminEmailBtn.classList.remove('disabled');
    }
}

// Close settings panel
settingsClose.addEventListener('click', () => {
    settingsOverlay.classList.remove('active');
});

// Close on overlay click
settingsOverlay.addEventListener('click', (e) => {
    if (e.target === settingsOverlay) {
        settingsOverlay.classList.remove('active');
    }
});

// Tab switching
settingsTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        switchSettingsTab(tabName);
    });
});

function switchSettingsTab(tabName) {
    // Update tab buttons
    settingsTabs.forEach(t => {
        t.classList.toggle('active', t.dataset.tab === tabName);
    });

    // Update tab content
    const adminTabContent = document.getElementById('adminTab');
    accountTab.classList.toggle('active', tabName === 'account');
    logoutTab.classList.toggle('active', tabName === 'logout');
    if (adminTabContent) {
        adminTabContent.classList.toggle('active', tabName === 'admin');
    }
}

// Send authentication code
sendAuthCode.addEventListener('click', async () => {
    sendAuthCode.disabled = true;
    authStatus.textContent = 'Sending authentication code...';
    authStatus.className = 'auth-status pending';

    // Simulate sending email (in production, this would call your backend)
    try {
        // Generate a 6-digit code
        pendingAuthCode = Math.floor(100000 + Math.random() * 900000).toString();

        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        // In production, send the code via email
        // await fetch(`${API_BASE_URL}/auth/send-code`, { ... });

        // Auth code is sent via email - never log to console in production

        authStatus.textContent = 'Authentication code sent to admin email.';
        authStatus.className = 'auth-status success';
        authCodeInput.style.display = 'flex';
        codeInput.focus();

    } catch (error) {
        authStatus.textContent = 'Failed to send code. Please try again.';
        authStatus.className = 'auth-status error';
    } finally {
        sendAuthCode.disabled = false;
    }
});

// Verify authentication code
verifyCode.addEventListener('click', () => {
    verifyAuthCode();
});

codeInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        verifyAuthCode();
    }
});

function verifyAuthCode() {
    const enteredCode = codeInput.value.trim();

    if (enteredCode.length !== 6) {
        authStatus.textContent = 'Please enter a 6-digit code.';
        authStatus.className = 'auth-status error';
        return;
    }

    if (enteredCode === pendingAuthCode) {
        isAuthenticated = true;
        authStatus.textContent = 'Verified! Credentials revealed.';
        authStatus.className = 'auth-status success';
        authCodeInput.style.display = 'none';

        // Reveal credentials
        revealCredentials();
    } else {
        authStatus.textContent = 'Invalid code. Please try again.';
        authStatus.className = 'auth-status error';
        codeInput.value = '';
        codeInput.focus();
    }
}

async function revealCredentials() {
    const usernameValue = document.getElementById('usernameValue');
    const passwordValue = document.getElementById('passwordValue');
    const adminEmailField = document.getElementById('adminEmailField');
    const adminEmailValue = document.getElementById('adminEmailValue');

    // Reveal username
    usernameValue.textContent = userAccountData.username;
    usernameValue.classList.remove('encrypted');
    usernameValue.classList.add('revealed');

    // Fetch password from server (requires authentication)
    try {
        const token = localStorage.getItem('userToken') || localStorage.getItem('adminToken');
        const response = await fetch(`${API_BASE_URL}/auth/get-credentials`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ authCode: pendingAuthCode })
        });

        if (response.ok) {
            const data = await response.json();
            passwordValue.textContent = data.password || '••••••••';
        } else {
            passwordValue.textContent = '(Contact admin)';
        }
    } catch (error) {
        // For demo/offline mode, show placeholder
        passwordValue.textContent = '(Server required)';
    }
    passwordValue.classList.remove('encrypted');
    passwordValue.classList.add('revealed');

    // If admin, also reveal admin email
    if (userAccountData.isAdmin) {
        adminEmailField.classList.remove('admin-restricted');
        adminEmailValue.textContent = userAccountData.adminEmail;
        adminEmailValue.classList.remove('encrypted');
        adminEmailValue.classList.add('revealed');
    }
    // For USER accounts, admin email stays encrypted/greyed out
}

function resetAuthState() {
    isAuthenticated = false;
    pendingAuthCode = null;

    const usernameValue = document.getElementById('usernameValue');
    const passwordValue = document.getElementById('passwordValue');
    const adminEmailField = document.getElementById('adminEmailField');
    const adminEmailValue = document.getElementById('adminEmailValue');
    const revealAdminEmailBtn = document.getElementById('revealAdminEmail');

    // Reset to encrypted state
    usernameValue.textContent = '••••••••••••';
    usernameValue.classList.add('encrypted');
    usernameValue.classList.remove('revealed');

    passwordValue.textContent = '••••••••••••';
    passwordValue.classList.add('encrypted');
    passwordValue.classList.remove('revealed');

    // Reset admin email to encrypted state
    adminEmailValue.textContent = '••••••••••••';
    adminEmailValue.classList.add('encrypted');
    adminEmailValue.classList.remove('revealed');

    // Apply admin-restricted state for non-admin users
    if (!userAccountData.isAdmin) {
        adminEmailField.classList.add('admin-restricted');
        revealAdminEmailBtn.classList.add('disabled');
    } else {
        adminEmailField.classList.remove('admin-restricted');
        revealAdminEmailBtn.classList.remove('disabled');
    }

    // Reset auth UI
    authCodeInput.style.display = 'none';
    codeInput.value = '';
    authStatus.textContent = '';
    authStatus.className = 'auth-status';
}

// Sign out functionality
logoutBtn.addEventListener('click', performSignOut);

function performSignOut() {
    // Clear all session data
    chatSessions = [];
    conversationHistory = [];
    currentSessionId = null;
    chatActive = false;

    // Clear localStorage - including auth tokens
    localStorage.clear();

    // Reset UI
    messages.innerHTML = '';
    logoContainer.classList.remove('chat-active');
    settingsOverlay.classList.remove('active');

    // Redirect to login page
    window.location.href = '../Login/index.html';
}

// ============================================
// DOCUMENT SEARCH FUNCTIONALITY
// ============================================

// Document type icons
const docTypeIcons = {
    risk: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>`,
    solution: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
    </svg>`,
    supplier: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
        <polyline points="9,22 9,12 15,12 15,22"></polyline>
    </svg>`,
    inventory: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="16.5" y1="9.4" x2="7.5" y2="4.21"></line>
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
        <polyline points="3.27,6.96 12,12.01 20.73,6.96"></polyline>
        <line x1="12" y1="22.08" x2="12" y2="12"></line>
    </svg>`
};

// Search documents with debounce
docSearchInput.addEventListener('input', (e) => {
    const query = e.target.value.trim();

    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }

    // Hide overlay if query is empty
    if (!query) {
        docSearchOverlay.classList.remove('active');
        return;
    }

    // Activate chat mode (move logo to top left) when user starts typing
    if (!chatActive && query.length > 0) {
        chatActive = true;
        logoContainer.classList.add('chat-active');
        logo.style.transform = 'none';
    }

    // Debounce search
    searchTimeout = setTimeout(() => {
        if (query.length >= 2) {
            searchDocuments(query);
        }
    }, 300);
});

// Focus search shows overlay if there's content
docSearchInput.addEventListener('focus', () => {
    const query = docSearchInput.value.trim();
    if (query.length >= 2) {
        docSearchOverlay.classList.add('active');
    }
});

// Close search overlay
docSearchClose.addEventListener('click', () => {
    docSearchOverlay.classList.remove('active');
    docSearchInput.value = '';
});

// Close on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        docSearchOverlay.classList.remove('active');
        historyPanel.classList.remove('active');
        alertsPanel.classList.remove('active');
        settingsOverlay.classList.remove('active');
    }
});

// Search documents API call
async function searchDocuments(query) {
    docSearchOverlay.classList.add('active');
    docSearchResults.innerHTML = '<div class="doc-search-loading">Searching...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/documents/search?q=${encodeURIComponent(query)}&limit=20`);
        const data = await response.json();

        if (response.ok && data.success) {
            renderSearchResults(data.results, query);
        } else {
            docSearchResults.innerHTML = `<div class="doc-search-empty">
                <svg class="doc-search-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="11" cy="11" r="8"></circle>
                    <path d="M21 21l-4.35-4.35"></path>
                </svg>
                <div>No results found for "${escapeHtml(query)}"</div>
            </div>`;
        }
    } catch (error) {
        console.error('Search error:', error);
        docSearchResults.innerHTML = `<div class="doc-search-empty">
            <div>Search failed. Please try again.</div>
        </div>`;
    }
}

// Render search results
function renderSearchResults(results, query) {
    if (!results || results.length === 0) {
        docSearchResults.innerHTML = `<div class="doc-search-empty">
            <svg class="doc-search-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <path d="M21 21l-4.35-4.35"></path>
            </svg>
            <div>No results found for "${escapeHtml(query)}"</div>
        </div>`;
        return;
    }

    docSearchResults.innerHTML = results.map(result => {
        const icon = docTypeIcons[result.type] || docTypeIcons.risk;
        let metaTags = `<span class="doc-result-tag type">${result.type}</span>`;

        // Add severity tag for risks
        if (result.metadata?.severity) {
            metaTags += `<span class="doc-result-tag severity-${result.metadata.severity.toLowerCase()}">${result.metadata.severity}</span>`;
        }

        // Add category tag
        if (result.metadata?.category) {
            metaTags += `<span class="doc-result-tag">${result.metadata.category}</span>`;
        }

        // Add SKU for inventory
        if (result.metadata?.sku) {
            metaTags += `<span class="doc-result-tag">SKU: ${result.metadata.sku}</span>`;
        }

        return `
            <div class="doc-result-item" data-id="${result.id}" data-type="${result.type}">
                <div class="doc-result-icon ${result.type}">${icon}</div>
                <div class="doc-result-content">
                    <div class="doc-result-title">${escapeHtml(result.title)}</div>
                    <div class="doc-result-meta">${metaTags}</div>
                </div>
            </div>
        `;
    }).join('');

    // Add click handlers for results
    docSearchResults.querySelectorAll('.doc-result-item').forEach(item => {
        item.addEventListener('click', () => {
            handleDocumentClick(item.dataset.id, item.dataset.type);
        });
    });
}

// Handle document click - analyze in chat
function handleDocumentClick(docId, docType) {
    docSearchOverlay.classList.remove('active');
    docSearchInput.value = '';

    // Activate chat mode
    if (!chatActive) {
        chatActive = true;
        logoContainer.classList.add('chat-active');
        logo.style.transform = 'none';
    }

    // Add message about viewing document
    addMessage(`View ${docType}: ${docId}`, 'user');
    addMessage(`Opening ${docType} details for ID: ${docId}. You can ask questions about this document.`, 'assistant');
}

// Alert item clicks
document.querySelectorAll('.alert-item').forEach(item => {
    item.addEventListener('click', () => {
        const alertId = item.getAttribute('data-alert-id');
        showAlertDetail(alertId);
    });
});

// Back button
alertBack.addEventListener('click', () => {
    showAlertList();
});

// Analyze in Chat button - sends to /api/risks/analyze
alertChatBtn.addEventListener('click', () => {
    if (currentAlert) {
        const alert = alertsData[currentAlert];

        // Close alerts panel
        alertsPanel.classList.remove('active');

        // Activate chat mode
        if (!chatActive) {
            chatActive = true;
            logoContainer.classList.add('chat-active');
            logo.style.transform = 'none';
        }

        // Add user message
        addMessage(`Analyze: ${alert.title}`, 'user');

        // Send to analyze endpoint
        analyzeRisk(alert);
    }
});

function showAlertList() {
    alertList.classList.remove('hidden');
    alertDetailView.classList.remove('active');
    currentAlert = null;
}

function showAlertDetail(alertId) {
    const alert = alertsData[alertId];
    if (!alert) return;

    currentAlert = alertId;

    // Update detail view
    detailDot.className = `alert-dot ${alert.severity}`;
    detailTitle.textContent = alert.title;
    detailSummary.textContent = alert.summary;
    detailImpact.textContent = alert.impact;
    detailTimeline.textContent = alert.timeline;
    detailAffected.textContent = alert.affected;

    // Show detail view, hide list
    alertList.classList.add('hidden');
    alertDetailView.classList.add('active');
}

// ============================================
// 3D LOGO MOUSE TRACKING
// ============================================

let mouseX = 0;
let mouseY = 0;
let logoX = 0;
let logoY = 0;

document.addEventListener('mousemove', (e) => {
    if (chatActive) return;

    const centerX = window.innerWidth / 2;
    const centerY = window.innerHeight / 2;

    mouseX = (e.clientX - centerX) / centerX;
    mouseY = (e.clientY - centerY) / centerY;
});

let animationFrameId = null;

function animateLogo() {
    if (!chatActive && logo) {
        // Slower, more subtle easing
        logoX += (mouseX - logoX) * 0.10;
        logoY += (mouseY - logoY) * 0.10;

        // Reduced rotation and effects for subtlety
        const rotateY = logoX * 20;
        const rotateX = -logoY * 20;
        const translateZ = Math.abs(logoX * logoY) * 30;
        const scale = 1 + Math.abs(logoX * logoY) * 0.08;

        logo.style.transform = `perspective(400px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(${translateZ}px) scale(${scale})`;
        animationFrameId = requestAnimationFrame(animateLogo);
    } else {
        // Stop animation loop when chat is active to save resources
        animationFrameId = null;
    }
}

// Start animation
animateLogo();

// Function to restart animation when needed (e.g., when returning to initial state)
function restartLogoAnimation() {
    if (!animationFrameId && !chatActive) {
        animateLogo();
    }
}

// ============================================
// MESSAGE FADE OUT ON SCROLL
// ============================================

// Debounce helper function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function checkMessageVisibility() {
    const messageElements = messages.querySelectorAll('.message');
    const containerRect = messages.getBoundingClientRect();

    messageElements.forEach(msg => {
        const msgRect = msg.getBoundingClientRect();
        const distanceFromTop = msgRect.top - containerRect.top;

        // Start fading when message is within 60px of top
        if (distanceFromTop < 60 && distanceFromTop > -msgRect.height) {
            const opacity = Math.max(0, distanceFromTop / 60);
            msg.style.opacity = opacity;
            msg.style.transform = `translateY(${(1 - opacity) * -10}px)`;
        } else if (distanceFromTop <= -msgRect.height) {
            // Hide message when fully scrolled out (don't remove - preserves history)
            msg.style.opacity = 0;
            msg.style.visibility = 'hidden';
        } else {
            msg.style.opacity = 1;
            msg.style.transform = 'translateY(0)';
            msg.style.visibility = 'visible';
        }
    });
}

// Use debounced scroll handler for better performance
if (messages) {
    messages.addEventListener('scroll', debounce(checkMessageVisibility, 16)); // ~60fps
}

// ============================================
// ATTACHMENT BUTTONS
// ============================================

if (docBtn) {
    docBtn.addEventListener('click', () => {
        if (docInput) docInput.click();
    });
}

if (photoBtn) {
    photoBtn.addEventListener('click', () => {
        if (photoInput) photoInput.click();
    });
}

if (audioBtn) {
    audioBtn.addEventListener('click', () => {
        if (audioInput) audioInput.click();
    });
}

if (videoBtn) {
    videoBtn.addEventListener('click', () => {
        if (videoInput) videoInput.click();
    });
}

if (docInput) docInput.addEventListener('change', () => handleFileUpload(docInput, 'document'));
if (photoInput) photoInput.addEventListener('change', () => handleFileUpload(photoInput, 'photo'));
if (audioInput) audioInput.addEventListener('change', () => handleFileUpload(audioInput, 'audio'));
if (videoInput) videoInput.addEventListener('change', () => handleFileUpload(videoInput, 'video'));

// ============================================
// SEND MESSAGE & ANALYZE RISK
// ============================================

if (sendBtn) sendBtn.addEventListener('click', sendMessage);

// Expand button toggle
let inputExpanded = false;
const inputWrapper = document.querySelector('.input-wrapper');
const inputSection = document.querySelector('.input-section');
const inputOverlay = document.getElementById('inputOverlay');

function expandInput() {
    inputExpanded = true;
    inputSection.classList.add('expanded');
    inputOverlay.classList.add('active');
    expandBtn.title = 'Collapse input';
    expandBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <polyline points="4 14 10 14 10 20"></polyline>
            <polyline points="20 10 14 10 14 4"></polyline>
            <line x1="14" y1="10" x2="21" y2="3"></line>
            <line x1="3" y1="21" x2="10" y2="14"></line>
        </svg>
    `;
    input.focus();
}

function collapseInput() {
    inputExpanded = false;
    inputSection.classList.remove('expanded');
    inputOverlay.classList.remove('active');
    expandBtn.title = 'Expand input';
    expandBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <polyline points="15 3 21 3 21 9"></polyline>
            <polyline points="9 21 3 21 3 15"></polyline>
            <line x1="21" y1="3" x2="14" y2="10"></line>
            <line x1="3" y1="21" x2="10" y2="14"></line>
        </svg>
    `;
}

if (expandBtn && input && inputWrapper) {
    expandBtn.addEventListener('click', () => {
        if (inputExpanded) {
            collapseInput();
        } else {
            expandInput();
        }
    });
}

// Close expanded input when clicking overlay
if (inputOverlay) {
    inputOverlay.addEventListener('click', () => {
        if (inputExpanded) {
            collapseInput();
        }
    });
}

// Input progress bar - shows scroll position
const inputProgressBar = document.getElementById('inputProgressBar');

function updateInputProgress() {
    if (!input || !inputProgressBar) return;

    const scrollHeight = input.scrollHeight - input.clientHeight;
    if (scrollHeight > 0) {
        const scrollPercent = (input.scrollTop / scrollHeight) * 100;
        inputProgressBar.style.width = scrollPercent + '%';
    } else {
        inputProgressBar.style.width = '0%';
    }
}

if (input) {
    input.addEventListener('scroll', updateInputProgress);
    input.addEventListener('input', updateInputProgress);
}

if (input) {
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 100) + 'px';

        // Activate chat mode (move logo to top left) when user starts typing
        const text = input.value.trim();
        if (!chatActive && text.length > 0) {
            chatActive = true;
            if (logoContainer) logoContainer.classList.add('chat-active');
            if (logo) logo.style.transform = 'none';
        }
    });
}

async function sendMessage() {
    if (!input) return;
    const message = input.value.trim();

    if (!message) return;
    if (message.length < 3) {
        addMessage('Please enter at least 3 characters', 'assistant');
        return;
    }
    if (isLoading) return;

    // Collapse input if expanded
    if (inputExpanded) {
        collapseInput();
    }

    // Activate chat mode
    if (!chatActive) {
        chatActive = true;
        if (logoContainer) logoContainer.classList.add('chat-active');
        if (logo) logo.style.transform = 'none';
    }

    addMessage(message, 'user');
    input.value = '';
    input.style.height = 'auto';

    // Send to analyze endpoint
    await analyzeRisk({ description: message });
}

async function analyzeRisk(alertData, retryCount = 0) {
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 1000; // 1 second base delay

    isLoading = true;
    sendBtn.disabled = true;

    // Show loading message
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE_URL}/risks/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: alertData.description || `${alertData.title}. ${alertData.summary} Impact: ${alertData.impact}`,
                data_type: 'unstructured'
            })
        });

        const data = await response.json();

        // Remove loading message
        removeLoadingMessage(loadingId);

        if (response.ok && data.success) {
            addStructuredResponse(data);
        } else {
            addMessage(data.error || 'Analysis failed. Please try again.', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingId);

        // Retry logic with exponential backoff
        if (retryCount < MAX_RETRIES) {
            const delay = RETRY_DELAY * Math.pow(2, retryCount);
            addMessage(`Connection error. Retrying in ${delay / 1000}s... (${retryCount + 1}/${MAX_RETRIES})`, 'assistant');
            await new Promise(resolve => setTimeout(resolve, delay));
            return analyzeRisk(alertData, retryCount + 1);
        } else {
            addMessage('Connection error. Please check your internet connection and try again.', 'assistant');
        }
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        if (input) input.focus();
    }
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="loading-dots">Analyzing</span>';

    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;

    return id;
}

function removeLoadingMessage(id) {
    const loadingMsg = document.getElementById(id);
    if (loadingMsg) loadingMsg.remove();
}

function addStructuredResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    let html = '';

    // Classification badges
    if (data.classification) {
        const { risk_type, severity, confidence, complexity_score } = data.classification;
        html += `
            <div class="classification-badge">
                <span class="risk-type-badge">${risk_type || 'RISK'}</span>
                <span class="severity-badge ${(severity || '').toLowerCase()}">${severity || 'MEDIUM'}</span>
                <span class="confidence-badge">${Math.round((confidence || 0) * 100)}% confidence</span>
                ${complexity_score ? `<span class="confidence-badge">Complexity: ${complexity_score}/10</span>` : ''}
            </div>
        `;
    }

    // Vigil Summary
    if (data.vigil_summary && Object.keys(data.vigil_summary).length > 0) {
        const summary = data.vigil_summary;

        if (summary.situation) {
            html += `
                <div class="response-section">
                    <div class="response-label">Situation</div>
                    <div class="response-value">${summary.situation}</div>
                </div>
            `;
        }

        if (summary.context) {
            html += `
                <div class="response-section">
                    <div class="response-label">Context</div>
                    <div class="response-value">${summary.context}</div>
                </div>
            `;
        }

        if (summary.approach) {
            html += `
                <div class="response-section">
                    <div class="response-label">Recommended Approach</div>
                    <div class="response-value">${summary.approach}</div>
                </div>
            `;
        }

        if (summary.timeline) {
            html += `
                <div class="response-section">
                    <div class="response-label">Timeline</div>
                    <div class="response-value">${summary.timeline}</div>
                </div>
            `;
        }
    }

    // Narrative fallback
    if ((!data.vigil_summary || Object.keys(data.vigil_summary).length === 0) && data.narrative) {
        html += `
            <div class="response-section">
                <div class="response-label">Analysis</div>
                <div class="response-value">${data.narrative}</div>
            </div>
        `;
    }

    // Detailed Analysis (historical matches, cascading effects, grok intelligence)
    if (data.detailed_analysis) {
        const analysis = data.detailed_analysis;

        if (analysis.historical_matches && analysis.historical_matches.length > 0) {
            html += `<div class="response-section"><div class="response-label">Historical Matches</div><div class="response-value">`;
            analysis.historical_matches.slice(0, 3).forEach(match => {
                html += `<div class="historical-match">
                    <span class="match-similarity">${Math.round((match.similarity || 0) * 100)}% match</span>
                    <span>${match.description || match.text || 'Previous similar risk'}</span>
                </div>`;
            });
            html += `</div></div>`;
        }

        if (analysis.cascading_effects && analysis.cascading_effects.length > 0) {
            html += `<div class="response-section"><div class="response-label">Cascading Effects</div><div class="response-value">`;
            analysis.cascading_effects.forEach(effect => {
                html += `<div style="margin-bottom: 4px;">• ${effect}</div>`;
            });
            html += `</div></div>`;
        }

        if (analysis.grok_intelligence) {
            const grok = analysis.grok_intelligence;
            if (grok.market_insights || grok.industry_context) {
                html += `<div class="response-section"><div class="response-label">External Intelligence</div>
                    <div class="response-value">${grok.market_insights || grok.industry_context || grok.summary || ''}</div>
                </div>`;
            }
        }
    }

    // NEW: Dual-Source Solutions with full details
    if (data.solutions && data.solutions.has_solutions && data.solutions.tiered_solutions) {
        const solutions = data.solutions;

        // Solution summary header
        html += `<div class="response-section">
            <div class="response-label">Solutions (${solutions.summary?.total_solutions || 0} found)</div>
            <div class="solution-sources">
                <span class="source-badge private">${solutions.summary?.from_private_data || 0} from Private Data</span>
                <span class="source-badge external">${solutions.summary?.from_external_intelligence || 0} from External Intelligence</span>
            </div>
        </div>`;

        // Risk context if available
        if (solutions.risk_context) {
            const ctx = solutions.risk_context;
            if (ctx.solution_focus_areas && ctx.solution_focus_areas.length > 0) {
                html += `<div class="response-section">
                    <div class="response-label">Focus Areas</div>
                    <div class="focus-areas">
                        ${ctx.solution_focus_areas.map(area => `<span class="focus-tag">${area.replace('_', ' ')}</span>`).join('')}
                    </div>
                </div>`;
            }
        }

        // Tiered solutions
        solutions.tiered_solutions.forEach(tier => {
            if (tier.solutions && tier.solutions.length > 0) {
                html += `<div class="tier-section">
                    <div class="tier-header">
                        <span class="tier-badge tier-${tier.tier}">${tier.urgency_label || `Tier ${tier.tier}: ${tier.urgency}`}</span>
                        <span class="tier-count">${tier.solution_count} solution${tier.solution_count !== 1 ? 's' : ''}</span>
                    </div>
                    <div class="tier-solutions">`;

                tier.solutions.forEach(sol => {
                    const isRecommended = sol.is_recommended;
                    const sourceClass = sol.source === 'supabase' ? 'private' : 'external';

                    html += `<div class="solution-card ${isRecommended ? 'recommended' : ''}">
                        <div class="solution-header">
                            <span class="solution-title">${sol.title}</span>
                            <span class="source-indicator ${sourceClass}">${sol.source_type}</span>
                        </div>`;

                    // Summary
                    if (sol.summary) {
                        html += `<div class="solution-summary">${sol.summary}</div>`;
                    }

                    // Description
                    if (sol.description && sol.description !== sol.summary) {
                        html += `<div class="solution-description">${sol.description}</div>`;
                    }

                    // Steps
                    if (sol.steps && sol.steps.length > 0) {
                        html += `<div class="solution-steps">`;
                        sol.steps.forEach(step => {
                            html += `<div class="step-item">
                                <span class="step-number">${step.step_number}</span>
                                <div class="step-content">
                                    <div class="step-action">${step.action}</div>
                                    ${step.details ? `<div class="step-details">${step.details}</div>` : ''}
                                    <div class="step-meta">
                                        <span class="step-owner">${step.responsible_party}</span>
                                        ${step.estimated_duration ? `<span class="step-duration">${step.estimated_duration}</span>` : ''}
                                    </div>
                                </div>
                            </div>`;
                        });
                        html += `</div>`;
                    }

                    // Expected outcome and timeline
                    if (sol.expected_outcome || sol.estimated_timeline) {
                        html += `<div class="solution-outcomes">`;
                        if (sol.expected_outcome) {
                            html += `<div class="outcome-item"><span class="outcome-label">Expected Outcome:</span> ${sol.expected_outcome}</div>`;
                        }
                        if (sol.estimated_timeline) {
                            html += `<div class="outcome-item"><span class="outcome-label">Timeline:</span> ${sol.estimated_timeline}</div>`;
                        }
                        html += `</div>`;
                    }

                    // Metrics
                    html += `<div class="solution-metrics">
                        <span class="metric">Confidence: ${Math.round((sol.confidence || 0) * 100)}%</span>
                        <span class="metric">Success: ${Math.round((sol.success_probability || 0) * 100)}%</span>
                        ${sol.solution_category ? `<span class="metric category">${sol.solution_category.replace('_', ' ')}</span>` : ''}
                    </div>`;

                    // Resource details (supplier_details, inventory_details, etc.)
                    if (sol.resource_details && Object.keys(sol.resource_details).length > 0) {
                        html += `<div class="resource-details">`;
                        Object.entries(sol.resource_details).forEach(([key, details]) => {
                            if (details && typeof details === 'object') {
                                const label = key.replace('_details', '').replace('_', ' ');
                                html += `<div class="resource-section">
                                    <span class="resource-label">${label}:</span>
                                    <div class="resource-data">`;
                                Object.entries(details).forEach(([k, v]) => {
                                    if (v !== null && v !== undefined) {
                                        const displayKey = k.replace('_', ' ');
                                        const displayVal = Array.isArray(v) ? v.join(', ') : v;
                                        html += `<span class="resource-item">${displayKey}: ${displayVal}</span>`;
                                    }
                                });
                                html += `</div></div>`;
                            }
                        });
                        html += `</div>`;
                    }

                    // Reference link
                    if (sol.reference) {
                        if (sol.reference.type === 'url') {
                            html += `<a href="${sol.reference.value}" target="_blank" class="reference-link">${sol.reference.label}</a>`;
                        } else if (sol.reference.type === 'document') {
                            html += `<span class="reference-doc">Doc ID: ${sol.reference.value}</span>`;
                        }
                    }

                    html += `</div>`; // Close solution-card
                });

                html += `</div></div>`; // Close tier-solutions and tier-section
            }
        });
    }
    // Legacy solutions fallback
    else if (data.legacy_solutions && data.legacy_solutions.length > 0) {
        html += `<div class="response-section"><div class="response-label">Solutions</div><div class="solutions-list">`;
        data.legacy_solutions.forEach(sol => {
            html += `<div class="solution-item">
                <span class="solution-tier ${sol.tier === 1 ? 'primary' : 'secondary'}">TIER ${sol.tier}</span>
                <span>${sol.title}</span>
                <span class="solution-prob">${Math.round((sol.success_probability || 0) * 100)}%</span>
            </div>`;
        });
        html += `</div></div>`;
    }

    // Alerts - Full display like output prompts (condensed version)
    if (data.alerts && data.alerts.length > 0) {
        html += `<div class="response-section">
            <div class="response-label">Alerts (${data.alerts.length})</div>
            <div class="alerts-container">`;

        data.alerts.forEach(alert => {
            const alertLevel = (alert.alert_level || 'MEDIUM').toLowerCase();
            const alertType = alert.alert_type || 'combined';

            // Build source badges
            let sourceBadges = '';
            if (alert.source) {
                if (alert.source.supabase) sourceBadges += '<span class="alert-source-badge supabase">Supabase</span>';
                if (alert.source.grok_intelligence) sourceBadges += '<span class="alert-source-badge grok">Grok</span>';
                if (alert.source.vigil_summary) sourceBadges += '<span class="alert-source-badge vigil">Vigil</span>';
            }

            // Build data sources info
            let dataSourcesInfo = '';
            if (alert.data_sources) {
                const ds = alert.data_sources;
                if (ds.supabase_risks_analyzed > 0) {
                    dataSourcesInfo += `<span class="data-source-info">${ds.supabase_risks_analyzed} historical risks analyzed</span>`;
                }
            }

            html += `
                <div class="alert-card ${alertLevel}">
                    <div class="alert-card-header">
                        <span class="alert-level-badge ${alertLevel}">${alert.alert_level || 'INFO'}</span>
                        <span class="alert-type-badge">${alertType.replace('_', ' ')}</span>
                        ${sourceBadges}
                    </div>
                    <div class="alert-card-title">${alert.title || 'Alert'}</div>
                    ${alert.summary ? `<div class="alert-card-summary">${alert.summary}</div>` : ''}
                    ${alert.action ? `
                        <div class="alert-card-action">
                            <span class="action-label">Action:</span>
                            <span class="action-text">${alert.action}</span>
                        </div>
                    ` : ''}
                    ${dataSourcesInfo ? `<div class="alert-card-sources">${dataSourcesInfo}</div>` : ''}
                </div>
            `;
        });

        html += `</div></div>`;
    }

    // Follow-up Questions - Clickable suggestions from Vigil
    if (data.follow_up_questions && data.follow_up_questions.length > 0) {
        html += `<div class="response-section follow-up-section">
            <div class="response-label">Suggested Questions</div>
            <div class="follow-up-questions">`;

        data.follow_up_questions.forEach((q, index) => {
            const question = typeof q === 'string' ? q : q.question;
            const category = typeof q === 'object' ? q.category : 'general';
            const hint = typeof q === 'object' ? q.context_hint : '';

            html += `
                <div class="follow-up-question" data-question="${escapeHtml(question)}" data-category="${category}">
                    <span class="follow-up-category ${category}">${category}</span>
                    <span class="follow-up-text">${escapeHtml(question)}</span>
                    ${hint ? `<span class="follow-up-hint">${escapeHtml(hint)}</span>` : ''}
                </div>
            `;
        });

        html += `</div></div>`;
    }

    // Validation info
    if (data.validation_info) {
        const info = data.validation_info;
        html += `<div class="response-section validation-info">
            <div class="response-label">Processing Info</div>
            <div class="validation-badges">
                <span class="validation-badge">${info.data_type_processed}</span>
                ${info.distilbert_vectorized ? '<span class="validation-badge">Vectorized</span>' : ''}
                ${info.grok_integrated ? '<span class="validation-badge">Grok AI</span>' : ''}
                ${info.grok_processed ? '<span class="validation-badge">Grok Intel</span>' : ''}
            </div>
        </div>`;
    }

    contentDiv.innerHTML = html || '<div class="response-value">Analysis complete. No additional details available.</div>';

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;

    conversationHistory.push({
        role: 'assistant',
        content: data,
        timestamp: new Date().toISOString()
    });

    // Store last analysis context for follow-up questions
    window.lastAnalysisContext = {
        risk_type: data.classification?.risk_type,
        severity: data.classification?.severity,
        previous_analysis: data.vigil_summary
    };

    // Add click handlers for follow-up questions
    setTimeout(() => {
        messageDiv.querySelectorAll('.follow-up-question').forEach(questionEl => {
            questionEl.addEventListener('click', () => {
                const question = questionEl.dataset.question;
                if (question) {
                    sendFollowUpQuestion(question);
                }
            });
        });
    }, 100);
}

// Send follow-up question to conversational chat endpoint
async function sendFollowUpQuestion(question) {
    if (isLoading) return;

    // Activate chat mode if not already
    if (!chatActive) {
        chatActive = true;
        logoContainer.classList.add('chat-active');
        logo.style.transform = 'none';
    }

    addMessage(question, 'user');

    isLoading = true;
    sendBtn.disabled = true;

    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: question,
                context: window.lastAnalysisContext || {}
            })
        });

        const data = await response.json();

        removeLoadingMessage(loadingId);

        if (response.ok && data.success) {
            addConversationalResponse(data);
        } else {
            addMessage(data.error || 'Failed to get response. Please try again.', 'assistant');
        }
    } catch (error) {
        console.error('Chat error:', error);
        removeLoadingMessage(loadingId);
        addMessage('Connection error. Please try again.', 'assistant');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        if (input) input.focus();
    }
}

// Add conversational response from Grok
function addConversationalResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    let html = '';

    // Source indicator
    const sourceLabel = data.source === 'grok' ? 'Grok Intelligence' : 'Vigil AI';
    html += `<div class="conversation-source">
        <span class="source-indicator ${data.source}">${sourceLabel}</span>
    </div>`;

    // Main response
    html += `<div class="conversation-response">${escapeHtml(data.response)}</div>`;

    // Data references (user's own data in parentheses style)
    if (data.data_references && data.data_references.length > 0) {
        html += `<div class="data-references">
            <span class="ref-label">Related from your data:</span>
            <div class="ref-list">`;
        data.data_references.slice(0, 3).forEach(ref => {
            html += `<span class="ref-item">${escapeHtml(ref.description.substring(0, 50))}...</span>`;
        });
        html += `</div></div>`;
    }

    // Follow-up suggestions
    if (data.follow_up_suggestions && data.follow_up_suggestions.length > 0) {
        html += `<div class="follow-up-suggestions">`;
        data.follow_up_suggestions.forEach(suggestion => {
            const text = typeof suggestion === 'string' ? suggestion : suggestion.question || suggestion;
            html += `<div class="follow-up-chip" data-question="${escapeHtml(text)}">${escapeHtml(text)}</div>`;
        });
        html += `</div>`;
    }

    contentDiv.innerHTML = html;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;

    // Add click handlers for follow-up chips
    messageDiv.querySelectorAll('.follow-up-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const question = chip.dataset.question;
            if (question) {
                sendFollowUpQuestion(question);
            }
        });
    });

    conversationHistory.push({
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString()
    });

    updateCurrentSession();
}

// Max file size: 10MB
const MAX_FILE_SIZE = 10 * 1024 * 1024;

async function handleFileUpload(inputEl, type) {
    const file = inputEl.files[0];
    if (!file) return;

    // Check file size limit
    if (file.size > MAX_FILE_SIZE) {
        addMessage(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB.`, 'assistant');
        inputEl.value = '';
        return;
    }

    if (!chatActive) {
        chatActive = true;
        logoContainer.classList.add('chat-active');
        logo.style.transform = 'none';
    }

    isLoading = true;
    sendBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);

    addMessage(`Uploaded: ${file.name}`, 'user');

    try {
        const response = await fetch(`${API_BASE_URL}/attachments`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            addMessage(data.response, 'assistant');
        } else {
            addMessage(data.response || 'File processing failed', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Upload failed. Please try again.', 'assistant');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        inputEl.value = '';
    }
}

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;

    conversationHistory.push({
        role,
        content,
        timestamp: new Date().toISOString()
    });

    // Update chat session
    updateCurrentSession();
}

// ============================================
// INITIALIZATION
// ============================================

// Check if user is logged in
const userToken = localStorage.getItem('userToken');
const adminToken = localStorage.getItem('adminToken');
const userRole = localStorage.getItem('userRole');

// Allow localhost access for development (no backend required)
const isLocalDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

if (!userToken && !adminToken && !isLocalDev) {
    // Not logged in, redirect to login
    window.location.href = '../Login/index.html';
} else {
    // Log user info for debugging
    console.log('='.repeat(60));
    console.log('VIGIL INTERFACE - User Session');
    console.log('='.repeat(60));
    console.log('Username:', userAccountData.username);
    console.log('Role:', userRole);
    console.log('Is Admin:', userAccountData.isAdmin);
    console.log('Admin Email:', userAccountData.adminEmail);
    console.log('Client Subdomain:', localStorage.getItem('clientSubdomain'));
    console.log('='.repeat(60));

    // Show admin features if user is admin
    if (userAccountData.isAdmin) {
        // Show Admin Access tab
        const adminTab = document.querySelector('.settings-tab[data-tab="admin"]');
        const adminTabContent = document.getElementById('adminTab');

        if (adminTab) adminTab.style.display = 'block';
        if (adminTabContent) adminTabContent.style.display = 'none'; // Hidden until clicked

        console.log('✓ Admin features enabled');
    }
}

// Load chat sessions on page load
loadChatSessions();

// ============================================
// CUSTOM SCROLLBAR
// ============================================

function updateCustomScrollbar() {
    if (!messages || !customScrollbar || !customScrollbarThumb) return;

    const scrollHeight = messages.scrollHeight;
    const clientHeight = messages.clientHeight;
    const scrollTop = messages.scrollTop;

    if (scrollHeight <= clientHeight) {
        customScrollbar.style.opacity = '0';
        return;
    }

    const trackHeight = customScrollbar.clientHeight;
    const thumbHeight = Math.max(30, (clientHeight / scrollHeight) * trackHeight);
    const maxThumbTop = trackHeight - thumbHeight;
    const thumbTop = (scrollTop / (scrollHeight - clientHeight)) * maxThumbTop;

    customScrollbarThumb.style.height = thumbHeight + 'px';
    customScrollbarThumb.style.top = thumbTop + 'px';
}

// Update scrollbar on messages scroll
if (messages) {
    messages.addEventListener('scroll', updateCustomScrollbar);
}

// Scrollbar drag functionality
let isDraggingScrollbar = false;
let scrollbarStartY = 0;
let scrollbarStartScrollTop = 0;

if (customScrollbarThumb) {
    customScrollbarThumb.addEventListener('mousedown', (e) => {
        isDraggingScrollbar = true;
        scrollbarStartY = e.clientY;
        scrollbarStartScrollTop = messages.scrollTop;
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
}

document.addEventListener('mousemove', (e) => {
    if (!isDraggingScrollbar) return;

    const deltaY = e.clientY - scrollbarStartY;
    const trackHeight = customScrollbar.clientHeight;
    const thumbHeight = customScrollbarThumb.clientHeight;
    const maxThumbTop = trackHeight - thumbHeight;
    const scrollRange = messages.scrollHeight - messages.clientHeight;

    const scrollDelta = (deltaY / maxThumbTop) * scrollRange;
    messages.scrollTop = scrollbarStartScrollTop + scrollDelta;
});

document.addEventListener('mouseup', () => {
    if (isDraggingScrollbar) {
        isDraggingScrollbar = false;
        document.body.style.userSelect = '';
    }
});

// Click on track to jump
if (customScrollbar) {
    customScrollbar.addEventListener('click', (e) => {
        if (e.target === customScrollbarThumb) return;

        const rect = customScrollbar.getBoundingClientRect();
        const clickY = e.clientY - rect.top;
        const trackHeight = customScrollbar.clientHeight;
        const scrollRange = messages.scrollHeight - messages.clientHeight;

        messages.scrollTop = (clickY / trackHeight) * scrollRange;
    });
}

// Initial update and observe for changes
updateCustomScrollbar();

// Update scrollbar when messages are added
const messagesObserver = new MutationObserver(updateCustomScrollbar);
if (messages) {
    messagesObserver.observe(messages, { childList: true, subtree: true });
}

// Update on window resize
window.addEventListener('resize', updateCustomScrollbar);
