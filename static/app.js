// ── MeetNote Frontend Application Logic ─────────────────────────

let currentMode = 'url';
let selectedFile = null;
let currentResult = null;

// Switch between YouTube URL and File Upload
function switchMode(mode) {
  currentMode = mode;
  document.getElementById('mode-url').classList.toggle('active', mode === 'url');
  document.getElementById('mode-file').classList.toggle('active', mode === 'file');
  document.getElementById('url-container').style.display = mode === 'url' ? 'flex' : 'none';
  document.getElementById('file-container').style.display = mode === 'file' ? 'flex' : 'none';
}

// File Drag and Drop Handlers
const dropzone = document.getElementById('dropzone');
if (dropzone) {
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      selectedFile = e.dataTransfer.files[0];
      document.getElementById('dropzone-label').textContent = `✓ ${selectedFile.name}`;
    }
  });
}

function handleFileSelected(input) {
  if (input.files.length > 0) {
    selectedFile = input.files[0];
    document.getElementById('dropzone-label').textContent = `✓ ${selectedFile.name}`;
  }
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

  event.target.classList.add('active');
  const targetPane = document.getElementById(`tab-${tabId}`);
  if (targetPane) targetPane.classList.add('active');
}

// Update Animated Phase Tracker
function setPhase(phaseNum, percent) {
  document.getElementById('progress-percent').textContent = `${percent}%`;
  document.getElementById('progress-bar').style.width = `${percent}%`;

  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById(`phase-${i}`);
    if (!el) continue;
    if (i < phaseNum) {
      el.className = 'phase-item done';
    } else if (i === phaseNum) {
      el.className = 'phase-item active';
    } else {
      el.className = 'phase-item';
    }
  }
}

// Generate Notes Execution
async function generateNotes() {
  const btn = document.getElementById('btn-generate');
  const language = document.getElementById('lang-select').value;
  const model = document.getElementById('model-select').value;

  if (currentMode === 'url') {
    const url = document.getElementById('yt-url').value.trim();
    if (!url) {
      alert('Please enter a YouTube video URL.');
      return;
    }
  } else {
    if (!selectedFile) {
      alert('Please select or drag an audio/video file.');
      return;
    }
  }

  btn.disabled = true;
  document.getElementById('hero-section').style.display = 'none';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'flex';

  // Animate Phases
  setPhase(1, 15);
  setTimeout(() => setPhase(2, 35), 1000);
  setTimeout(() => setPhase(3, 60), 2500);

  try {
    let response;
    if (currentMode === 'url') {
      const url = document.getElementById('yt-url').value.trim();
      response = await fetch('/api/process-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, language, model })
      });
    } else {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('language', language);
      formData.append('model', model);

      response = await fetch('/api/upload-file', {
        method: 'POST',
        body: formData
      });
    }

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Failed to process media.');
    }

    setPhase(4, 85);
    const data = await response.json();
    currentResult = data;

    setPhase(5, 100);
    setTimeout(() => {
      document.getElementById('progress-section').style.display = 'none';
      renderResults(data);
      btn.disabled = false;
    }, 600);

  } catch (error) {
    btn.disabled = false;
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('hero-section').style.display = 'flex';
    alert(`Error: ${error.message}`);
  }
}

// Render Dashboard Data
function renderResults(data) {
  document.getElementById('result-title').textContent = data.title || 'Meeting Summary';
  document.getElementById('metric-words').textContent = (data.word_count || 0).toLocaleString();
  document.getElementById('metric-time').textContent = `~${Math.max(1, Math.floor((data.word_count || 0) / 150))} min`;

  document.getElementById('content-summary').textContent = data.summary || 'No summary available.';
  document.getElementById('content-decisions').textContent = data.key_decisions || 'No decisions recorded.';
  document.getElementById('content-actions').textContent = data.action_items || 'No action items identified.';
  document.getElementById('content-questions').textContent = data.open_questions || 'No open questions.';
  document.getElementById('content-transcript').textContent = data.transcript || '';

  document.getElementById('results-section').style.display = 'flex';
}

// Send Interactive Chat Query
async function sendChatMessage() {
  const input = document.getElementById('chat-query');
  const query = input.value.trim();
  if (!query || !currentResult) return;

  input.value = '';
  const messagesContainer = document.getElementById('chat-messages');

  // Append user bubble
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble user';
  userBubble.textContent = query;
  messagesContainer.appendChild(userBubble);

  // Append loading assistant bubble
  const assistantBubble = document.createElement('div');
  assistantBubble.className = 'chat-bubble assistant';
  assistantBubble.textContent = 'Searching transcript...';
  messagesContainer.appendChild(assistantBubble);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transcript: currentResult.transcript,
        question: query
      })
    });

    if (!res.ok) throw new Error('Failed to query knowledge base.');
    const data = await res.json();
    assistantBubble.textContent = data.answer;
  } catch (err) {
    assistantBubble.textContent = `Error: ${err.message}`;
  }

  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Download PDF / Markdown Exports
async function downloadExport(format) {
  if (!currentResult) return;

  const endpoint = format === 'pdf' ? '/api/export-pdf' : '/api/export-md';
  const filename = `${(currentResult.title || 'MeetNote').replace(/\s+/g, '_')}_notes.${format}`;

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: currentResult.title,
        summary: currentResult.summary,
        action_items: currentResult.action_items,
        key_decisions: currentResult.key_decisions,
        open_questions: currentResult.open_questions,
        transcript: currentResult.transcript
      })
    });

    if (!res.ok) throw new Error('Download failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Export error: ${err.message}`);
  }
}

// Copy Transcript to Clipboard
function copyTranscript() {
  if (!currentResult || !currentResult.transcript) return;
  navigator.clipboard.writeText(currentResult.transcript);
  alert('Transcript copied to clipboard!');
}
