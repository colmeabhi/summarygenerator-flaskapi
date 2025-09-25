# Summary Generator Flask API

This project exposes a Flask-based HTTP API that summarises free-form text. It combines classic NLP feature extraction (tokenisation, TF-ISF, NER, proper noun density, etc.) with a PCA-based feature enhancement system that scores sentences and selects the most representative subset. The API returns the extracted summary as JSON.

The codebase is split into focused modules under `summarygenerator/`:

- `app.py` – Flask entry point that exposes the `/post_summary` POST endpoint.
- `summarygenerator/entity.py` – Named-entity extraction helpers built on NLTK.
- `summarygenerator/text_features.py` – Text cleaning, tokenisation, sentence scoring and utility functions.
- `summarygenerator/rbm_simple.py` – PCA-based feature enhancement system (scikit-learn-backed).
- `summarygenerator/summary.py` – Orchestrates the full summarisation workflow.
- `summarygenerator/resources.py` – Centralised NLP resources (stopwords, regexes, NLTK downloads).
- `model_features.py` – Compatibility layer that re-exports the modular API for legacy imports.

## Prerequisites

- Python 3.9+ (the scientific stack is compatible with modern Python versions).
- Recommended: virtual environment (`venv`, `conda`, or `pipenv`).

## Local Setup

```bash
git clone https://github.com/colmeabhi/rbm-summarization-model.git
cd rbm-summarization-model

# Create and activate a virtual environment (example using venv)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

On first run, NLTK will download the required corpora (`averaged_perceptron_tagger`, `punkt`, `maxent_ne_chunker`, `words`, and `stopwords`). The download is triggered automatically at import time; ensure the process has permission to write to `~/nltk_data`.

## Running the API

```bash
# Launch the Flask development server (runs on port 5001 by default)
python3 app.py

# ...or run on a different port using the Flask CLI
python3 -m flask --app app run --port 5002
```

The server logs will indicate if the default port is already in use. Choose an alternative port via `--port` as shown above.

## Requesting a Summary

Endpoint: `POST /post_summary`

Request body (JSON):

```json
{
  "textString": "Highways are lines of communication, touching life on many levels..."
}
```

Using `curl`:

```bash
curl -X POST http://localhost:5001/post_summary \
     -H 'Content-Type: application/json' \
     -d '{"textString": "Your paragraph text here."}'
```

Response:

```json
{
  "summary": "The extracted summary sentences concatenated together."
}
```

## Project Notes

- The PCA-based feature enhancement processes the feature matrix on every request. Processing time depends on input size and typically completes quickly.
- `summarygenerator/resources.py` captures global state (stopword lists, compiled regexes, current working directory).
- The project uses modern versions of NumPy, pandas, and scikit-learn for optimal compatibility and performance.

## Testing & Development

- Run the Flask app and send sample requests as shown above.
- Monitor the console output to see feature extraction and processing details.
- If you adjust NLTK usage, remember to download any additional corpora in `summarygenerator/resources.py`.

Contributions and improvements to the summarisation pipeline or the API structure are welcome.

