const form = document.getElementById('summary-form');
const textarea = document.getElementById('text-input');
const statusEl = document.getElementById('status');
const summaryEl = document.getElementById('summary-output');
const submitBtn = document.getElementById('submit-btn');

function setStatus(message, isError) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? '#cc0000' : '#333333';
  statusEl.classList.remove('hidden');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = textarea.value.trim();
  if (!text) {
    setStatus('Please enter text to summarise.', true);
    return;
  }

  submitBtn.disabled = true;
  setStatus('Generating summary, please wait...');
  summaryEl.classList.add('hidden');

  try {
    const response = await fetch('/post_summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ textString: text }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.error || 'Request failed');
    }

    const data = await response.json();
    summaryEl.textContent = data.summary || 'No summary returned.';
    summaryEl.classList.remove('hidden');
    setStatus('Summary generated successfully.');
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    submitBtn.disabled = false;
  }
});
