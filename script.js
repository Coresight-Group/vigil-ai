const API_BASE_URL = 'http://localhost:5000/api';

const messages = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const attachBtn = document.getElementById('attachBtn');
const attachMenu = document.getElementById('attachMenu');

const docInput = document.getElementById('docInput');
const photoInput = document.getElementById('photoInput');
const audioInput = document.getElementById('audioInput');
const videoInput = document.getElementById('videoInput');

const docBtn = document.getElementById('docBtn');
const photoBtn = document.getElementById('photoBtn');
const audioBtn = document.getElementById('audioBtn');
const videoBtn = document.getElementById('videoBtn');

let isLoading = false;
let conversationHistory = [];

// Attachment menu
attachBtn.addEventListener('click', () => {
    attachMenu.classList.toggle('active');
});

document.addEventListener('click', (e) => {
    if (!e.target.closest('.input-section')) {
        attachMenu.classList.remove('active');
    }
});

// Attachment buttons
docBtn.addEventListener('click', () => {
    docInput.click();
    attachMenu.classList.remove('active');
});

photoBtn.addEventListener('click', () => {
    photoInput.click();
    attachMenu.classList.remove('active');
});

audioBtn.addEventListener('click', () => {
    audioInput.click();
    attachMenu.classList.remove('active');
});

videoBtn.addEventListener('click', () => {
    videoInput.click();
    attachMenu.classList.remove('active');
});

// File uploads
docInput.addEventListener('change', () => handleFileUpload(docInput, 'document'));
photoInput.addEventListener('change', () => handleFileUpload(photoInput, 'photo'));
audioInput.addEventListener('change', () => handleFileUpload(audioInput, 'audio'));
videoInput.addEventListener('change', () => handleFileUpload(videoInput, 'video'));

// Send message
sendBtn.addEventListener('click', sendMessage);

input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 100) + 'px';
});

async function sendMessage() {
    const message = input.value.trim();

    if (!message) return;
    if (message.length < 20) return;
    if (isLoading) return;

    addMessage(message, 'user');
    input.value = '';
    input.style.height = 'auto';
    isLoading = true;
    sendBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        if (response.ok) {
            addMessage(data.response, 'assistant');
        } else {
            addMessage(data.response || 'Error processing message', 'assistant');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Connection error. Please try again.', 'assistant');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

async function handleFileUpload(input, type) {
    const file = input.files[0];

    if (!file) return;

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
        input.value = '';
    }
}

function addMessage(content, role) {
    if (messages.firstChild?.classList.contains('welcome')) {
        messages.innerHTML = '';
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    const now = new Date();
    timeDiv.textContent = now.toLocaleTimeString('en-US', {
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
        timestamp: now.toISOString()
    });
}
