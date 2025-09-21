# Summary Generator Flask API

This project exposes a Flask-based HTTP API that summarises free-form text. It combines classic NLP feature extraction (tokenisation, TF-ISF, NER, proper noun density, etc.) with a Theano-based Restricted Boltzmann Machine (RBM) that scores sentences and selects the most representative subset. The API returns the extracted summary and additionally writes it to `outputs/Summary.txt` on disk.

The codebase is split into focused modules under `summarygenerator/`:

- `app.py` – Flask entry point that exposes the `/json-example` POST endpoint.
- `summarygenerator/entity.py` – Named-entity extraction helpers built on NLTK.
- `summarygenerator/text_features.py` – Text cleaning, tokenisation, sentence scoring and utility functions.
- `summarygenerator/rbm.py` – RBM implementation and training routine (Theano-backed).
- `summarygenerator/summary.py` – Orchestrates the full summarisation workflow.
- `summarygenerator/resources.py` – Centralised NLP resources (stopwords, regexes, NLTK downloads).
- `model_features.py` – Compatibility layer that re-exports the modular API for legacy imports.

## Prerequisites

- Python 3.9 (the pinned scientific stack targets 3.9; other versions may require dependency updates).
- A working C/Fortran toolchain (Xcode Command Line Tools on macOS or build-essential on Linux) for Theano/NumPy builds.
- Recommended: virtual environment (`venv`, `conda`, or `pipenv`).

## Local Setup

```bash
git clone https://github.com/colmeabhi/summarygenerator-flaskapi.git
cd summarygenerator-flaskapi

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
# Launch the Flask development server on the default port (5000)
python3 app.py

# ...or run on a different port using the Flask CLI
python3 -m flask --app app run --port 5001
```

The server logs will indicate if the default port is already in use. Choose an alternative port via `--port` as shown above.

## Requesting a Summary

Endpoint: `POST /json-example`

Request body (JSON):

```json
{
  "textString": "Highways are lines of communication, touching life on many levels..."
}
```

Using `curl`:

```bash
curl -X POST http://localhost:5000/json-example \
     -H 'Content-Type: application/json' \
     -d '{"textString": "Your paragraph text here."}'
```

Response:

```
The given text is : <summary sentences>
```

The same summary is also written to `outputs/Summary.txt` for later review.

## Project Notes

- The RBM retrains on every request using the generated feature matrix. Depending on input size, this may take several seconds.
- `summarygenerator/resources.py` captures global state (stopword lists, compiled regexes, current working directory). If you change the working directory while the app runs, ensure you still have write access to `outputs/Summary.txt`.
- The project currently targets older versions of NumPy, pandas, scikit-learn, and Theano (per `requirements.txt`) for compatibility. Upgrading these libraries will likely require code changes.

## Testing & Development

- Run the Flask app and send sample requests as shown above.
- Inspect `outputs/Summary.txt` to validate sentence ordering.
- If you adjust NLTK usage, remember to download any additional corpora in `summarygenerator/resources.py`.

Contributions and improvements to the summarisation pipeline or the API structure are welcome.

