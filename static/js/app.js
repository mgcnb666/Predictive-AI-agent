// ===== Configuration =====
const CONFIG_STORAGE_KEY = 'predictionai_config';
const API_BASE_URL = window.location.origin;

// Default configuration
const DEFAULT_CONFIG = {
    search: {
        serperApiKey: '',
        jinaApiKey: '',
        searchProvider: 'serper',
        reranker: 'jina'
    },
    llm: {
        litellmModelId: 'openrouter/google/gemini-2.0-flash-001'
    },
    openrouter: {
        openrouterApiKey: '',
        openrouterBaseUrl: 'https://openrouter.ai/api/v1',
        openrouterModel: 'google/gemini-2.0-flash-001',
        yourSiteUrl: '',
        yourSiteName: 'Prediction AI Agent'
    }
};

// ===== State Management =====
class AppState {
    constructor() {
        this.config = this.loadConfig();
    }

    loadConfig() {
        const stored = localStorage.getItem(CONFIG_STORAGE_KEY);
        if (stored) {
            try {
                return JSON.parse(stored);
            } catch (e) {
                console.error('Failed to parse config:', e);
            }
        }
        return JSON.parse(JSON.stringify(DEFAULT_CONFIG));
    }

    saveConfig() {
        localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(this.config));
    }

    resetConfig() {
        this.config = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
        this.saveConfig();
    }

    getApiHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.config.search.serperApiKey) {
            headers['X-Serper-Api-Key'] = this.config.search.serperApiKey;
        }
        if (this.config.search.jinaApiKey) {
            headers['X-Jina-Api-Key'] = this.config.search.jinaApiKey;
        }
        if (this.config.openrouter.openrouterApiKey) {
            headers['X-OpenRouter-Api-Key'] = this.config.openrouter.openrouterApiKey;
        }
        if (this.config.llm.litellmModelId) {
            headers['X-LiteLLM-Model'] = this.config.llm.litellmModelId;
        }
        if (this.config.search.searchProvider) {
            headers['X-Search-Provider'] = this.config.search.searchProvider;
        }
        if (this.config.search.reranker) {
            headers['X-Reranker'] = this.config.search.reranker;
        }

        return headers;
    }
}

const appState = new AppState();

// ===== API Functions =====
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = appState.getApiHeaders();

    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...headers,
                ...options.headers
            }
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || error.detail || 'Request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Request failed:', error);
        throw error;
    }
}

async function predict(data) {
    return apiRequest('/api/v1/predict', {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

async function checkHealth() {
    return apiRequest('/health');
}

// ===== Chat UI Functions =====
function showLoading() {
    document.getElementById('chatLoading').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('chatLoading').style.display = 'none';
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addUserMessage(text) {
    const chatMessages = document.getElementById('chatMessages');
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-avatar">U</div>
        <div class="message-content">
            <div class="message-bubble">${text}</div>
            <div class="message-time">${time}</div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function addBotMessage(content) {
    const chatMessages = document.getElementById('chatMessages');
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-bubble">
                ${content}
            </div>
            <div class="message-time">${time}</div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function showError(message) {
    addBotMessage(`<div class="error-message">${message}</div>`);
}

function renderPredictionResult(result) {
    // Handle the actual response format from UniversalPredictionAgent
    const domain = result.domain || 'general';
    const timestamp = result.timestamp;

    // Extract analysis and confidence from the top-level result
    const analysis = result.analysis || result.summary || '';
    const confidence = result.confidence || 0.5;

    // Determine confidence level
    let confidenceBadge;
    if (confidence >= 0.7) {
        confidenceBadge = '<span class="confidence-badge">High Confidence</span>';
    } else if (confidence >= 0.5) {
        confidenceBadge = '<span class="confidence-badge">Medium Confidence</span>';
    } else {
        confidenceBadge = '<span class="confidence-badge">Low Confidence</span>';
    }

    let html = `
        <div class="prediction-result">
            <div class="result-header">
                <div class="result-title">${getDomainName(domain)}</div>
                ${confidenceBadge}
            </div>
    `;

    // Analysis section
    if (analysis) {
        html += `
            <div class="result-section">
                <div class="result-section-title">📊 Prediction Analysis</div>
                <div class="result-text">${formatAnalysis(analysis)}</div>
            </div>
        `;
    }

    // Confidence and domain info
    html += `
        <div class="result-section">
            <div class="result-grid">
                <div class="result-item">
                    <strong>Confidence</strong>
                    <span>${(confidence * 100).toFixed(1)}%</span>
                </div>
                ${timestamp ? `
                <div class="result-item">
                    <strong>Analysis Time</strong>
                    <span>${new Date(timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;

    // Key factors
    if (result.factors && Array.isArray(result.factors) && result.factors.length > 0) {
        html += `
            <div class="result-section">
                <div class="result-section-title">🔑 Key Factors</div>
                <ul class="result-list">
                    ${result.factors.map(factor => `<li>${factor}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Weather forecast
    if (result.forecast) {
        html += renderForecast(result.forecast);
    }

    // Warnings
    if (result.warnings && Array.isArray(result.warnings) && result.warnings.length > 0) {
        html += `
            <div class="result-section">
                <div class="result-section-title">⚠️ Warnings</div>
                <ul class="result-list">
                    ${result.warnings.map(warning => `<li>${warning}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    html += '</div>';
    return html;
}

function getDomainName(domain) {
    const names = {
        'general': 'General Prediction',
        'sports': 'Sports',
        'weather': 'Weather',
        'election': 'Election'
    };
    return names[domain] || domain;
}

function formatAnalysis(analysis) {
    // If analysis is already formatted HTML, return it
    if (analysis.includes('<')) return analysis;

    // Convert newlines to <br> and preserve formatting
    return analysis
        .split('\n')
        .map(line => line.trim())
        .filter(line => line)
        .map(line => {
            // Bold headers
            if (line.endsWith(':') || line.match(/^#+\s/)) {
                return `<p><strong>${line}</strong></p>`;
            }
            // List items
            if (line.match(/^[-*•]\s/)) {
                return `<li>${line.substring(2)}</li>`;
            }
            return `<p>${line}</p>`;
        })
        .join('');
}


function renderForecast(forecast) {
    if (!forecast || typeof forecast !== 'object') return '';

    let html = '<div class="result-section"><div class="result-section-title">🌤️ Weather Details</div><div class="result-grid">';

    if (forecast.temperature_range) {
        const temp = forecast.temperature_range;
        html += `
            <div class="result-item">
                <strong>Temperature Range</strong>
                <span>${temp.low || 0}°C - ${temp.high || 0}°C</span>
            </div>
        `;
    }

    if (forecast.precipitation_prob !== undefined) {
        html += `
            <div class="result-item">
                <strong>Precipitation Probability</strong>
                <span>${(forecast.precipitation_prob * 100).toFixed(0)}%</span>
            </div>
        `;
    }

    if (forecast.weather_condition) {
        html += `
            <div class="result-item">
                <strong>Weather Condition</strong>
                <span>${forecast.weather_condition}</span>
            </div>
        `;
    }

    if (forecast.wind_speed) {
        const wind = forecast.wind_speed;
        html += `
            <div class="result-item">
                <strong>Wind Speed</strong>
                <span>${wind.speed || 0} ${wind.unit || 'km/h'}</span>
            </div>
        `;
    }

    if (forecast.humidity !== undefined) {
        html += `
            <div class="result-item">
                <strong>Humidity</strong>
                <span>${(forecast.humidity * 100).toFixed(0)}%</span>
            </div>
        `;
    }

    html += '</div></div>';
    return html;
}

// ===== Event Handlers =====
document.addEventListener('DOMContentLoaded', function() {
    // Check API health on load
    checkHealth().then(() => {
        const statusDot = document.getElementById('statusDot');
        statusDot.classList.add('online');
    }).catch(() => {
        const statusDot = document.getElementById('statusDot');
        statusDot.classList.add('offline');
    });

    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Chat form submit
    document.getElementById('chatForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const question = chatInput.value.trim();
        const domain = 'general';  // Always use general domain
        const useSearch = document.getElementById('useSearch').checked;

        if (!question) {
            return;
        }

        // Add user message to chat
        addUserMessage(question);

        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Prepare parameters
        const params = {
            query: question,
            question: question
        };

        const data = {
            domain,
            params,
            use_search: useSearch
        };

        showLoading();

        try {
            const result = await predict(data);
            hideLoading();

            // Render and add bot message with prediction result
            const resultHTML = renderPredictionResult(result);
            addBotMessage(resultHTML);
        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    });

    // Settings modal
    document.getElementById('settingsBtn').addEventListener('click', function() {
        loadSettingsToForm();
        document.getElementById('settingsModal').classList.add('active');
    });

    document.getElementById('closeSettings').addEventListener('click', function() {
        document.getElementById('settingsModal').classList.remove('active');
    });

    document.getElementById('saveSettings').addEventListener('click', function() {
        saveSettingsFromForm();
        document.getElementById('settingsModal').classList.remove('active');
        addBotMessage('<div class="success-message">✅ Settings saved successfully!</div>');
    });

    document.getElementById('resetSettings').addEventListener('click', function() {
        if (confirm('Are you sure you want to reset all settings to default values?')) {
            appState.resetConfig();
            loadSettingsToForm();
            alert('Settings have been reset to default values');
        }
    });

    // Settings tabs
    document.querySelectorAll('.settings-tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const setting = this.dataset.setting;

            document.querySelectorAll('.settings-tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            document.querySelectorAll('.settings-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`${setting}-settings`).classList.add('active');
        });
    });

    // Model chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', function() {
            const model = this.dataset.model;
            document.getElementById('litellmModelId').value = model;
        });
    });

    // Close modals on outside click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.remove('active');
            }
        });
    });
});

// ===== Settings Functions =====
function loadSettingsToForm() {
    const config = appState.config;

    // Search settings
    document.getElementById('serperApiKey').value = config.search.serperApiKey || '';
    document.getElementById('jinaApiKey').value = config.search.jinaApiKey || '';
    document.getElementById('searchProvider').value = config.search.searchProvider || 'serper';
    document.getElementById('reranker').value = config.search.reranker || 'jina';

    // LLM settings
    document.getElementById('litellmModelId').value = config.llm.litellmModelId || '';

    // OpenRouter settings
    document.getElementById('openrouterApiKey').value = config.openrouter.openrouterApiKey || '';
    document.getElementById('openrouterBaseUrl').value = config.openrouter.openrouterBaseUrl || '';
    document.getElementById('openrouterModel').value = config.openrouter.openrouterModel || '';
    document.getElementById('yourSiteUrl').value = config.openrouter.yourSiteUrl || '';
    document.getElementById('yourSiteName').value = config.openrouter.yourSiteName || '';
}

function saveSettingsFromForm() {
    appState.config = {
        search: {
            serperApiKey: document.getElementById('serperApiKey').value,
            jinaApiKey: document.getElementById('jinaApiKey').value,
            searchProvider: document.getElementById('searchProvider').value,
            reranker: document.getElementById('reranker').value
        },
        llm: {
            litellmModelId: document.getElementById('litellmModelId').value
        },
        openrouter: {
            openrouterApiKey: document.getElementById('openrouterApiKey').value,
            openrouterBaseUrl: document.getElementById('openrouterBaseUrl').value,
            openrouterModel: document.getElementById('openrouterModel').value,
            yourSiteUrl: document.getElementById('yourSiteUrl').value,
            yourSiteName: document.getElementById('yourSiteName').value
        }
    };

    appState.saveConfig();
}
