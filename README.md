<div align="center">

# ⚽ FIFA World Cup 2026 Analytics Hub

**An interactive data-science platform for World Cup match prediction, 48-team tournament simulation, and player style clustering — built on transparent, honestly-labeled models instead of black-box guesses.**

[Live Demo](http://localhost:8501) · [Report Bug](https://github.com) · [Request Feature](https://github.com)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-ff4b4b?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest%20%2F%20KMeans-f7931e?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3f4f75?logo=plotly&logoColor=white)

</div>

---

## Overview

**World Cup 2026 Analytics Hub** combines three data-science mini-projects — match outcome prediction, player-style clustering, and tournament simulation — into one Streamlit app with a pitch-green, gold-accented "official tournament" UI.

Every model in this project is built to be **honest about its own limits**: predictions come with a visible hold-out accuracy score, training data is clearly labeled as illustrative/synthetic rather than passed off as a live feed, and every bracket or scoreline shown is explicitly flagged as a model projection — not a real result.

---

## Interactive Feature Preview

<div align="center">

![Match Predictor and Knockout Bracket](./src/assets/images/predictor-preview.png)

*Match Outcome Predictor, 48-team Group Draw, and the model-simulated knockout bracket*

</div>

---

## Key Features

* **Match Outcome Predictor** — a `RandomForestClassifier` trained on Elo rating, FIFA rank, squad market value, head-to-head record, and venue, returning a home/draw/away probability split plus an expected-goals (xG) estimate.
* **Model Transparency Panel** — the sidebar shows the model's actual hold-out test accuracy (vs. the ~33% random-guess baseline for a 3-way outcome) instead of implying false confidence.
* **48-Team Group Draw & Full Tournament Simulator** — a proper 4-pot seeded draw across 12 groups of 4, a full round-robin group stage, top-2 + best-8-third-place qualification logic, and a Round-of-32 → Final knockout bracket rendered as an animated, Champions-League-style visual.
* **Player Style Clustering** — K-Means clustering over six playing-style attributes (pace, shooting, passing, dribbling, defending, physical), with clusters auto-labeled ("Deep-Lying Playmaker," "Ball-Winning Defender," etc.) from their centroid rather than shown as raw numbers.
* **Click-to-Inspect Radar Chart** — click any player point on the cluster scatter plot to open a live Plotly radar chart of that player's full attribute profile.
* **Beat the AI** — a gamified mode where you pick your own bracket and see how many picks the model agrees with.
* **Synthetic Data Engine** — match history is generated from a strength-weighted Poisson goal model (not random noise), so the ML model has real signal to learn from even without a live data feed.

---

## Tech Stack

| Category | Technology & Specification |
| :--- | :--- |
| **App Framework** | Streamlit (multi-tab layout, `st.cache_resource` / `st.cache_data`, `on_select` chart interactivity) |
| **Modeling** | scikit-learn (`RandomForestClassifier`, `KMeans`, `StandardScaler`, `LabelEncoder`) |
| **Data Handling** | pandas, NumPy (Poisson-based synthetic match generation, Elo rating updates) |
| **Visualization** | Plotly Express / Graph Objects (interactive scatter + radar charts), custom HTML/CSS bracket renderer via `streamlit.components.v1` |
| **Styling** | Custom CSS (pitch-green / gold "official tournament" theme, Google Fonts `Inter`) |

---

## Project Structure

```text
worldcup/
├── app.py                          # Streamlit entry point — tabs, theming, bracket renderer
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── teams_data.py               # 48-team illustrative dataset, flags, seeded group draw
└── src/
    ├── match_engine/
    │   └── predictor.py            # WorldCupMatchPredictor: synthetic data, Elo, RF model, bracket sim
    ├── player_clustering/
    │   └── clusterer.py            # PlayerStyleClusterer: KMeans, auto-labeling, scatter + radar charts
    └── sentiment_tracker/
        └── tracker.py              # Disabled — kept for future use, see Roadmap below
```

---

## Model & Data Transparency Notes

This project intentionally does **not** pretend to use real, live football data:

- `data/teams_data.py` — 48 illustrative national teams with hand-set rank / market value / Elo / country code. Not sourced from a live FIFA ranking feed.
- `predictor.py::generate_synthetic_history()` — match results are simulated via a strength-weighted Poisson model, not scraped from a real results database.
- `clusterer.py::SAMPLE_PLAYERS` — 20 illustrative player profiles with approximate 0–100 style ratings, not sourced from an official stats provider.
- Every prediction, bracket, and scoreline in the UI is labeled a **model projection**, and the sidebar always shows the model's real hold-out accuracy.

See [Roadmap](#roadmap--future-work) below for how to swap each of these for real data sources.

---

## Getting Started

```bash
# Clone and enter the project
git clone https://github.com/mhdhamka/worldcup-analytics.git
cd worldcup-analytics-hub

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Roadmap / Future Work

### Real Data Integration
- [ ] Replace `generate_synthetic_history()` with a real historical results dataset (e.g. a Kaggle international-football-results export) so Elo ratings and head-to-head stats reflect actual matches.
- [ ] Wire in a live FIFA World Ranking feed (official API or a maintained CSV mirror) in place of the hand-set `TEAMS_DB` ranks.
- [ ] Swap `SAMPLE_PLAYERS` for a real per-90 stats export (FBref, Opta, or StatsBomb open data) so clustering reflects actual season performance.
- [ ] Implement FIFA's official cross-group Round-of-32 qualifier rules in `build_round_of_32()`, replacing the simplified illustrative seeding.

### Modeling Improvements
- [ ] Benchmark `RandomForestClassifier` against gradient-boosted alternatives (XGBoost, LightGBM, CatBoost).
- [ ] Move from a single train/test split to k-fold cross-validation for a more reliable accuracy estimate.
- [ ] Calibrate predicted probabilities (`CalibratedClassifierCV`) so the win/draw/loss percentages are better aligned with real-world frequencies.
- [ ] Add feature importance / SHAP explanations to the Match Predictor tab for interpretability.
- [ ] Track and display prediction accuracy against real completed matches, if/when a live data feed is added.

### Sentiment Tracker Revival
- [ ] Replace the disabled `transformers` + `tweepy` pipeline with a lightweight lexicon-based sentiment scorer (no heavy model download, no paid API tier required).
- [ ] Build a simulated live match-event feed (goal / red card / VAR review triggers a sentiment spike) for a "fun and interactive" experience without needing real tweet access.
- [ ] If reviving real social listening, evaluate current-generation X/Bluesky/Reddit APIs and their access tiers before re-adding `tweepy`/`transformers` to `requirements.txt`.

### Deployment & Engineering
- [ ] Containerize with Docker for reproducible deployment.
- [ ] Deploy to Streamlit Community Cloud or Hugging Face Spaces for a public live demo link.
- [ ] Add a GitHub Actions CI pipeline (lint with `ruff`/`flake8`, run unit tests on push).
- [ ] Add `pytest` unit tests for `predictor.py` (Elo math, outcome labeling, bracket logic) and `clusterer.py` (cluster labeling heuristics).
- [ ] Expose the predictor as a small FastAPI service so other apps/scripts can query predictions without going through the Streamlit UI.

### UX & Gamification
- [ ] Persist "Beat the AI" scores (SQLite or a hosted DB) to build a real leaderboard across sessions/users.
- [ ] Add a shareable results card/image for tournament simulation outcomes.
- [ ] Mobile-responsive layout pass and accessibility audit (contrast ratios, screen-reader labels on custom HTML components).
- [ ] Internationalization (i18n) for team names and UI copy.

---

## Contributing

Issues and pull requests are welcome. If you're picking up one of the Roadmap items above, please open an issue first so effort isn't duplicated.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/real-data-integration`)
3. Commit your changes
4. Open a pull request describing what changed and why

---

## License

Distributed under the MIT License.

---

<div align="center">

If you found this project interesting, consider giving it a star ⭐

Crafted by **[@mhdhamka](https://github.com/mhdhamka)**

</div>