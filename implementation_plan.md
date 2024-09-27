# CohortCraft — Implementation Plan

Customer propensity and segmentation modeling toolkit. A reproducible Python pipeline that
generates (or ingests) customer + transaction data, engineers features with SQL, trains
conversion propensity models (logistic regression, XGBoost) evaluated with AUC-ROC, segments
customers with K-means, and estimates lifetime value with survival analysis across
marketing cohorts.

**Stack:** Python 3.10+, pandas, NumPy, scikit-learn, XGBoost, SQLite (via `sqlite3` + SQL
files), lifelines (Kaplan-Meier / Cox PH), matplotlib, pytest.

## Key design decisions
- **Synthetic data with known structure**: a seeded generator produces customers across
  acquisition-channel/month cohorts, transactions, campaign touches, conversion labels and
  churn times, so every model has signal to find and results are reproducible.
- **SQL feature store**: raw tables are loaded into SQLite; RFM, engagement and cohort features
  are computed by plain `.sql` files, keeping feature logic portable to a warehouse.
- **Model registry by config**: models are defined in a small registry so the trainer,
  evaluator and CLI iterate over them uniformly.
- **Survival-based CLV**: CLV = margin-per-period x sum of discounted survival probabilities
  S(t) from a Cox model, estimated per customer and aggregated per cohort and segment.

## Phase 1 — Scaffolding, synthetic data, SQL feature store
- `pyproject.toml`, `requirements.txt`, `cohortcraft/__init__.py`, `cohortcraft/config.py`
- `cohortcraft/data/synthetic.py` — seeded customer/transaction/campaign generator
- `cohortcraft/data/db.py` — SQLite load/query helpers
- `sql/schema.sql`, `sql/features.sql`, `sql/cohorts.sql`
- `cohortcraft/features.py` — build feature matrix from SQL

## Phase 2 — Propensity modeling
- `cohortcraft/propensity/models.py` — LR pipeline (scaling + one-hot) and XGBoost model registry
- `cohortcraft/propensity/train.py` — stratified split, CV, fit, persist with joblib
- `cohortcraft/propensity/evaluate.py` — AUC-ROC, PR-AUC, Brier, lift/decile table, ROC plot

## Phase 3 — Customer segmentation
- `cohortcraft/segmentation/kmeans.py` — feature scaling, k selection (elbow + silhouette), fit
- `cohortcraft/segmentation/profile.py` — segment profiles and heuristic segment naming

## Phase 4 — Survival analysis & lifetime value
- `cohortcraft/survival/lifetime.py` — Kaplan-Meier per cohort, Cox PH model, log-rank tests
- `cohortcraft/survival/clv.py` — discounted survival-based CLV per customer/cohort/segment

## Phase 5 — Pipeline, CLI, reporting, tests
- `cohortcraft/pipeline.py`, `cohortcraft/cli.py` — end-to-end run, artifacts to `outputs/`
- `cohortcraft/report.py` — markdown summary report
- `tests/` — unit tests for generator, SQL features, models, segmentation, CLV
- `.env.example`, `.gitignore`, `README.md` with architecture diagram
