# Movie Review Intelligence & Sentiment Analyzer

NLP/ML pipeline that classifies movie reviews as **positive / neutral / negative**, analyses opinions per
aspect (acting, storyline, direction, cinematography), retrieves supporting evidence with **RAG**, and
exports a structured **PDF / DOCX** report.

| Layer | Responsibility | Code |
|---|---|---|
| NLP | cleaning, sentence splitting, aspect tagging | `src/preprocess.py`, `src/aspects.py` |
| ML | 3-class sentiment (NB / LogReg / SVM / DistilBERT) | `src/train_baselines.py`, `src/train_bert.py` |
| RAG | sentence-transformer embeddings + FAISS retrieval | `src/rag.py` |
| LLM | writes the report from retrieved evidence (optional) | `src/report.py` |

## Setup (VS Code)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv`.

## Run
```bash
# 1. Baselines (fast, CPU). Downloads SST-5 and collapses it to 3 classes.
python -m src.train_baselines

# 2. (Optional, GPU recommended) fine-tune DistilBERT
python -m src.train_bert --epochs 3

# 3. App
streamlit run app.py
```
Metrics (accuracy, macro precision/recall/F1) go to `reports/*_metrics.csv`, with per-model
classification reports and confusion matrices as `.txt` / `.png` in `reports/`.

The app uses BERT automatically if `models/bert` exists, otherwise the best baseline.

### LLM-written reports (optional)
```bash
export ANTHROPIC_API_KEY=...     # Windows PowerShell: $env:ANTHROPIC_API_KEY="..."
```
Tick the checkbox in the sidebar. Without a key the report is built from a deterministic template,
so everything works offline. Override the model with `ANTHROPIC_MODEL`.

## Using your own reviews
CSV with a `text` column (and optionally `movie`). See `data/sample_reviews.csv`.

## Notes for your write-up
- SST-5 is sentence-level, so the sentiment model is tuned for short text. Whole reviews are classified
  as-is (truncated at 256 tokens for BERT); aspect sentences match the training distribution best.
- Expect the **neutral** class to be the hardest. Discuss it using the confusion matrices.
- Aspect detection is keyword-based (transparent, easy to explain). A natural extension is a
  trained ABSA model.
- Ideas for later: IMDB-50k as a larger RAG corpus, persisting FAISS indexes per movie, scraping/TMDB API.
**CI/CD Status:** Automated testing active via Jenkins Pipeline.